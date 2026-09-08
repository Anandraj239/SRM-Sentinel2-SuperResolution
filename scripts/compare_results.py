import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Paths
lr_path  = r'C:\SRM_Project\data\val\LR\img1_0080.png'
sr_path  = r'C:\SRM_Project\BasicSR\experiments\SwinIR_Satellite_SR_x4\visualization\img1_0080\img1_0080_5000.png'
hr_path  = r'C:\SRM_Project\data\val\HR\img1_0080.png'
out_path = r'C:\SRM_Project\data\comparison_result.png'

# Load images
lr = Image.open(lr_path).convert('RGB')
sr = Image.open(sr_path).convert('RGB')
hr = Image.open(hr_path).convert('RGB')

# Resize all to same size for comparison
size = (512, 512)
lr_big = lr.resize(size, Image.NEAREST)
sr_res = sr.resize(size, Image.LANCZOS)
hr_res = hr.resize(size, Image.LANCZOS)

# Create side by side image
total_w = size[0] * 3 + 40
total_h = size[1] + 60
comparison = Image.new('RGB', (total_w, total_h), (255, 255, 255))

# Paste images
comparison.paste(lr_big, (0, 60))
comparison.paste(sr_res, (size[0] + 20, 60))
comparison.paste(hr_res, (size[0] * 2 + 40, 60))

# Add labels
draw = ImageDraw.Draw(comparison)
draw.text((180, 10), 'LR Input (10m)',     fill=(255,0,0),   font=None)
draw.text((size[0]+180, 10), 'SwinIR SR (2.5m)', fill=(0,150,0), font=None)
draw.text((size[0]*2+200, 10), 'HR Reference',   fill=(0,0,255),  font=None)

# Add metrics
draw.text((10, 35), 'Size: 128x128', fill=(100,100,100))
draw.text((size[0]+10, 35), 'PSNR: 24.69 dB | SSIM: 0.707 | Size: 512x512', fill=(100,100,100))
draw.text((size[0]*2+50, 35), 'Size: 512x512', fill=(100,100,100))

comparison.save(out_path)
print('Comparison saved!')