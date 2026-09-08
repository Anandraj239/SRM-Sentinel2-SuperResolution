import torch
import numpy as np
from PIL import Image
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

# Load model
processor = SegformerImageProcessor.from_pretrained("nvidia/segformer-b2-finetuned-ade-512-512")
model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b2-finetuned-ade-512-512")
model.eval()

# Load SR image
sr_image = Image.open(r'C:\SRM_Project\BasicSR\experiments\SwinIR_Satellite_SR_x4\visualization\img1_0080\img1_0080_5000.png').convert('RGB')

# Run inference
inputs = processor(images=sr_image, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)

# Get segmentation map
logits = outputs.logits
upsampled = torch.nn.functional.interpolate(
    logits, size=sr_image.size[::-1],
    mode='bilinear', align_corners=False
)
seg_map = upsampled.argmax(dim=1)[0].numpy()

# Color map for visualization
color_map = np.zeros((*seg_map.shape, 3), dtype=np.uint8)

# Map ADE20K classes to colors
vegetation_classes = [4, 9, 17, 21, 66]    # grass, field, tree, plant
road_classes       = [6, 11, 52]            # road, sidewalk, path
water_classes      = [21, 26, 60]           # water, sea, river
building_classes   = [0, 1, 25]             # wall, building, house
soil_classes       = [13, 29, 94]           # earth, dirt, land

for cls in vegetation_classes:
    color_map[seg_map == cls] = [34, 139, 34]   # Green

for cls in road_classes:
    color_map[seg_map == cls] = [128, 128, 128]  # Grey

for cls in water_classes:
    color_map[seg_map == cls] = [0, 100, 255]    # Blue

for cls in building_classes:
    color_map[seg_map == cls] = [255, 69, 0]     # Red

for cls in soil_classes:
    color_map[seg_map == cls] = [139, 90, 43]    # Brown

# Save segmentation map
seg_image = Image.fromarray(color_map)
seg_image.save(r'C:\SRM_Project\data\segmentation_map.png')
print('Segmentation map saved!')

# Save overlay (SR + segmentation blended)
sr_array = np.array(sr_image)
overlay  = (sr_array * 0.6 + color_map * 0.4).astype(np.uint8)
overlay_image = Image.fromarray(overlay)
overlay_image.save(r'C:\SRM_Project\data\segmentation_overlay.png')
print('Overlay saved!')