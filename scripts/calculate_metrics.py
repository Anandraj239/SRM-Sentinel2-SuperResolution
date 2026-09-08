import torch
import lpips
import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import os

print('=' * 60)
print('   COMPLETE METRICS EVALUATION')
print('=' * 60)

# Load LPIPS
loss_fn = lpips.LPIPS(net='alex')

# Paths
val_lr = r'C:\SRM_Project\data\val\LR'
val_hr = r'C:\SRM_Project\data\val\HR'
val_sr = r'C:\SRM_Project\BasicSR\experiments\SwinIR_Satellite_SR_x4\visualization'

psnr_list  = []
ssim_list  = []
lpips_list = []

files = sorted(os.listdir(val_hr))
print(f'Evaluating {len(files)} validation images...')
print('-' * 60)

for fname in files:
    hr_path = os.path.join(val_hr, fname)
    name    = fname.replace('.png','')
    sr_path = os.path.join(val_sr, name, f'{name}_5000.png')

    if not os.path.exists(sr_path):
        continue

    # Load images
    hr = np.array(Image.open(hr_path).convert('RGB'))
    sr = np.array(Image.open(sr_path).convert('RGB').resize(
        (hr.shape[1], hr.shape[0]), Image.LANCZOS))

    # PSNR
    p = psnr(hr, sr, data_range=255)
    psnr_list.append(p)

    # SSIM
    s = ssim(hr, sr, channel_axis=2, data_range=255)
    ssim_list.append(s)

    # LPIPS
    hr_t = torch.from_numpy(hr).permute(2,0,1).float().unsqueeze(0) / 127.5 - 1
    sr_t = torch.from_numpy(sr).permute(2,0,1).float().unsqueeze(0) / 127.5 - 1
    l = loss_fn(hr_t, sr_t).item()
    lpips_list.append(l)

    print(f'{fname}: PSNR={p:.2f} SSIM={s:.4f} LPIPS={l:.4f}')

print('-' * 60)
print(f'Average PSNR  : {np.mean(psnr_list):.4f} dB')
print(f'Average SSIM  : {np.mean(ssim_list):.4f}')
print(f'Average LPIPS : {np.mean(lpips_list):.4f} (lower is better)')
print('=' * 60)

# Save metrics
with open(r'C:\SRM_Project\data\metrics_report.txt', 'w') as f:
    f.write('METRICS REPORT\n')
    f.write('=' * 40 + '\n')
    f.write(f'Average PSNR  : {np.mean(psnr_list):.4f} dB\n')
    f.write(f'Average SSIM  : {np.mean(ssim_list):.4f}\n')
    f.write(f'Average LPIPS : {np.mean(lpips_list):.4f}\n')
    f.write(f'Total Images  : {len(psnr_list)}\n')
print('Metrics saved to metrics_report.txt')