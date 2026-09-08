import rasterio
import numpy as np
from PIL import Image
import os

# Create folders
os.makedirs(r'C:\SRM_Project\data\train\HR', exist_ok=True)
os.makedirs(r'C:\SRM_Project\data\train\LR', exist_ok=True)
os.makedirs(r'C:\SRM_Project\data\val\HR', exist_ok=True)
os.makedirs(r'C:\SRM_Project\data\val\LR', exist_ok=True)

def extract_patches(tci_path, prefix, start_idx=0, num_patches=100):
    print(f'Processing {tci_path}...')
    with rasterio.open(tci_path) as src:
        print(f'Image size: {src.width} x {src.height}')
        width, height = src.width, src.height

    patch_size = 512
    count = 0
    idx = start_idx

    for row in range(0, height - patch_size, patch_size):
        for col in range(0, width - patch_size, patch_size):
            if count >= num_patches:
                break

            with rasterio.open(tci_path) as src:
                window = rasterio.windows.Window(col, row, patch_size, patch_size)
                red   = src.read(1, window=window)
                green = src.read(2, window=window)
                blue  = src.read(3, window=window)

            rgb = np.stack([red, green, blue], axis=-1)

            # Skip dark/empty patches
            if rgb.mean() < 10:
                continue

            # Normalize
            rgb = ((rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8) * 255).astype(np.uint8)

            # HR = original 512x512
            hr_img = Image.fromarray(rgb)

            # LR = downscale 4x to 128x128
            lr_img = hr_img.resize((128, 128), Image.BICUBIC)

            # Save
            if count < 80:  # 80% train
                hr_img.save(f'C:\\SRM_Project\\data\\train\\HR\\{prefix}_{idx:04d}.png')
                lr_img.save(f'C:\\SRM_Project\\data\\train\\LR\\{prefix}_{idx:04d}.png')
            else:           # 20% val
                hr_img.save(f'C:\\SRM_Project\\data\\val\\HR\\{prefix}_{idx:04d}.png')
                lr_img.save(f'C:\\SRM_Project\\data\\val\\LR\\{prefix}_{idx:04d}.png')

            count += 1
            idx += 1
            print(f'  Saved patch {count}/{num_patches}')

        if count >= num_patches:
            break

    print(f'Done! {count} patches extracted.')
    return idx

# Process both images
next_idx = extract_patches(
    r'C:\SRM_Project\data\sentinel2\LR\sentinel2_TCI.jp2',
    prefix='img1',
    start_idx=0,
    num_patches=100
)

extract_patches(
    r'C:\SRM_Project\data\sentinel2\LR2\sentinel2_TCI_2.jp2',
    prefix='img2',
    start_idx=next_idx,
    num_patches=100
)

train_hr = len(os.listdir(r'C:\SRM_Project\data\train\HR'))
train_lr = len(os.listdir(r'C:\SRM_Project\data\train\LR'))
val_hr   = len(os.listdir(r'C:\SRM_Project\data\val\HR'))
val_lr   = len(os.listdir(r'C:\SRM_Project\data\val\LR'))

print('\nDataset Summary:')
print(f'Train HR: {train_hr} images')
print(f'Train LR: {train_lr} images')
print(f'Val HR:   {val_hr} images')
print(f'Val LR:   {val_lr} images')