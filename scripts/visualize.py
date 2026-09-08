import rasterio
import numpy as np
from PIL import Image

with rasterio.open(r'C:\SRM_Project\data\sentinel2\LR\sentinel2_TCI.jp2') as src:
    red   = src.read(1)
    green = src.read(2)
    blue  = src.read(3)
    print(f'Image size: {src.width} x {src.height} pixels')
    print(f'Resolution: {src.res} metres')

rgb = np.stack([red, green, blue], axis=-1)
rgb = ((rgb - rgb.min()) / (rgb.max() - rgb.min()) * 255).astype(np.uint8)
img = Image.fromarray(rgb)
img.save(r'C:\SRM_Project\data\sentinel2\LR\sentinel2_preview.png')
print('Done! Image saved.')