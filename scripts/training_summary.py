import os
import re
from datetime import datetime
import torch

log_path = r'C:\SRM_Project\BasicSR\experiments\SwinIR_Satellite_SR_x4\train_SwinIR_Satellite_SR_x4_20260906_135813.log'

print('=' * 60)
print('   SWINIR SATELLITE SR - TRAINING SUMMARY')
print('=' * 60)

if torch.cuda.is_available():
    gpu = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f'Hardware    : {gpu}')
    print(f'VRAM        : {vram:.1f} GB')
else:
    print('Hardware    : CPU')

print(f'Framework   : PyTorch {torch.__version__}')
print(f'Model       : SwinIR-Medium (11.7M parameters)')
print(f'Scale       : 4x (128x128 -> 512x512)')
print(f'Dataset     : Sentinel-2 L2A (Copernicus)')
print(f'Train pairs : 160 images')
print(f'Val pairs   : 40 images')
print('-' * 60)

psnr_list = []
ssim_list = []
loss_list = []
start_time = None
end_time = None

with open(log_path, 'r') as f:
    lines = f.readlines()

for line in lines:
    if 'Start training' in line:
        ts = line[:23]
        try:
            start_time = datetime.strptime(ts, '%Y-%m-%d %H:%M:%S,%f')
        except:
            pass
    if 'End of training' in line:
        ts = line[:23]
        try:
            end_time = datetime.strptime(ts, '%Y-%m-%d %H:%M:%S,%f')
        except:
            pass
    if 'psnr:' in line and 'Best:' in line:
        match = re.search(r'psnr:\s*([\d.]+)', line)
        if match:
            psnr_list.append(float(match.group(1)))
    if 'ssim:' in line and 'Best:' in line:
        match = re.search(r'ssim:\s*([\d.]+)', line)
        if match:
            ssim_list.append(float(match.group(1)))
    if 'l_pix:' in line:
        match = re.search(r'l_pix:\s*([\d.e+-]+)', line)
        if match:
            loss_list.append(float(match.group(1)))

if start_time and end_time:
    total_secs = (end_time - start_time).total_seconds()
    print(f'Total Time  : {int(total_secs//60)} min {int(total_secs%60)} sec')
    print(f'Start       : {start_time.strftime("%Y-%m-%d %H:%M:%S")}')
    print(f'End         : {end_time.strftime("%Y-%m-%d %H:%M:%S")}')

print('-' * 60)
print('VALIDATION METRICS:')
print(f'{"Checkpoint":<15} {"PSNR (dB)":<15} {"SSIM"}')
print('-' * 40)
checkpoints = [500,1000,1500,2000,2500,3000,3500,4000,4500,5000]
for i,(p,s) in enumerate(zip(psnr_list, ssim_list)):
    ckpt = checkpoints[i] if i < len(checkpoints) else i+1
    print(f'{ckpt:<15} {p:<15.4f} {s:.4f}')

print('-' * 60)
if psnr_list:
    print(f'Best PSNR   : {max(psnr_list):.4f} dB')
    print(f'Best SSIM   : {max(ssim_list):.4f}')
if loss_list:
    print(f'Final Loss  : {loss_list[-1]:.6f}')
    print(f'Initial Loss: {loss_list[0]:.6f}')
    imp = ((loss_list[0]-loss_list[-1])/loss_list[0])*100
    print(f'Improvement : {imp:.1f}%')
print('=' * 60)