import rasterio
import numpy as np
from PIL import Image

with rasterio.open(r'C:\SRM_Project\data\sentinel2\LR\sentinel2_TCI.jp2') as src:
    print(f'Full image size: {src.width} x {src.height}')
    
    # Crop urban area patch (top-left where city is visible)
    window = rasterio.windows.Window(500, 500, 512, 512)
    red   = src.read(1, window=window)
    green = src.read(2, window=window)
    blue  = src.read(3, window=window)

rgb = np.stack([red, green, blue], axis=-1)
rgb = ((rgb - rgb.min()) / (rgb.max() - rgb.min()) * 255).astype(np.uint8)
img = Image.fromarray(rgb)
img.save(r'C:\SRM_Project\data\sentinel2\LR\patch_urban.png')
print('Urban patch saved! Size: 512x512 at 10m = 5.12km x 5.12km area')