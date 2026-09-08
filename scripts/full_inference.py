import torch
import numpy as np
from PIL import Image
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
import sys
sys.path.insert(0, r'C:\SRM_Project\SwinIR')
from models.network_swinir import SwinIR

# Load SwinIR model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SwinIR(
    upscale=4, in_chans=3, img_size=32, window_size=8,
    img_range=1., depths=[6,6,6,6,6,6], embed_dim=180,
    num_heads=[6,6,6,6,6,6], mlp_ratio=2,
    upsampler='nearest+conv', resi_connection='1conv'
).to(device)

# Load trained weights
ckpt = r'C:\SRM_Project\BasicSR\experiments\SwinIR_Satellite_SR_x4\models\net_g_5000.pth'
weights = torch.load(ckpt, map_location=device)
model.load_state_dict(weights['params'], strict=False)
model.eval()
print('SwinIR model loaded!')

# Load Sentinel-2 patch
img = Image.open(r'C:\SRM_Project\data\sentinel2\LR\patch_urban.png').convert('RGB')
img_tensor = torch.from_numpy(
    np.array(img).astype(np.float32) / 255.
).permute(2,0,1).unsqueeze(0).to(device)

# Run SwinIR SR
with torch.no_grad():
    sr_tensor = model(img_tensor)

sr_np = sr_tensor.squeeze().permute(1,2,0).cpu().numpy()
sr_np = np.clip(sr_np * 255, 0, 255).astype(np.uint8)
sr_image = Image.fromarray(sr_np)
sr_image.save(r'C:\SRM_Project\data\sentinel2_SR_output.png')
print('SR output saved!')

# Run SegFormer on SR output
processor = SegformerImageProcessor.from_pretrained(
    "nvidia/segformer-b2-finetuned-ade-512-512"
)
seg_model = SegformerForSemanticSegmentation.from_pretrained(
    "nvidia/segformer-b2-finetuned-ade-512-512"
)
seg_model.eval()

inputs = processor(images=sr_image, return_tensors="pt")
with torch.no_grad():
    outputs = seg_model(**inputs)

logits = outputs.logits
upsampled = torch.nn.functional.interpolate(
    logits, size=sr_image.size[::-1],
    mode='bilinear', align_corners=False
)
seg_map = upsampled.argmax(dim=1)[0].numpy()

# Color map
color_map = np.zeros((*seg_map.shape, 3), dtype=np.uint8)
vegetation = [4, 9, 17, 21, 66]
roads      = [6, 11, 52]
water      = [21, 26, 60]
buildings  = [0, 1, 25]
soil       = [13, 29, 94]

for c in vegetation: color_map[seg_map==c] = [34,139,34]
for c in roads:      color_map[seg_map==c] = [128,128,128]
for c in water:      color_map[seg_map==c] = [0,100,255]
for c in buildings:  color_map[seg_map==c] = [255,69,0]
for c in soil:       color_map[seg_map==c] = [139,90,43]

seg_image = Image.fromarray(color_map)
seg_image.save(r'C:\SRM_Project\data\sentinel2_segmentation.png')

# Final overlay
overlay = (sr_np * 0.6 + color_map * 0.4).astype(np.uint8)
Image.fromarray(overlay).save(r'C:\SRM_Project\data\sentinel2_overlay.png')
print('All outputs saved!')
print('Files saved:')
print('  SR Output  -> sentinel2_SR_output.png')
print('  Seg Map    -> sentinel2_segmentation.png')
print('  Overlay    -> sentinel2_overlay.png')