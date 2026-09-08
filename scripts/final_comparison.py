import numpy as np
from PIL import Image, ImageDraw

size = (512, 512)

lr  = Image.open(r'C:\SRM_Project\data\val\LR\img1_0080.png').convert('RGB').resize(size, Image.NEAREST)
sr  = Image.open(r'C:\SRM_Project\BasicSR\experiments\SwinIR_Satellite_SR_x4\visualization\img1_0080\img1_0080_5000.png').convert('RGB').resize(size)
seg = Image.open(r'C:\SRM_Project\data\segmentation_map.png').convert('RGB').resize(size)
ovr = Image.open(r'C:\SRM_Project\data\segmentation_overlay.png').convert('RGB').resize(size)

gap = 20
w = size[0]*4 + gap*3
h = size[1] + 80
final = Image.new('RGB', (w, h), (30, 30, 30))

final.paste(lr,  (0,              80))
final.paste(sr,  (size[0]+gap,    80))
final.paste(seg, (size[0]*2+gap*2,80))
final.paste(ovr, (size[0]*3+gap*3,80))

draw = ImageDraw.Draw(final)
draw.text((100,  20), 'LR Input (10m)',       fill=(255,100,100))
draw.text((size[0]+gap+50,  20), 'SwinIR SR (2.5m)', fill=(100,255,100))
draw.text((size[0]*2+gap*2+50, 20), 'SegFormer Map', fill=(100,200,255))
draw.text((size[0]*3+gap*3+50, 20), 'SR + Segmentation', fill=(255,200,100))

draw.text((50, 50), 'Pixelated | Blurry',     fill=(200,200,200))
draw.text((size[0]+gap+20, 50), 'PSNR:24.69dB SSIM:0.707', fill=(200,200,200))
draw.text((size[0]*2+gap*2+50, 50), 'Land Cover Classes', fill=(200,200,200))
draw.text((size[0]*3+gap*3+20, 50), 'Full Pipeline Output', fill=(200,200,200))

final.save(r'C:\SRM_Project\data\final_pipeline_result.png')
print('Final 4-panel image saved!')