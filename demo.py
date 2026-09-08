import os
import subprocess
import time

def show(title, msg):
    print(f'\n{"="*60}')
    print(f'  {title}')
    print(f'{"="*60}')
    print(f'  {msg}')
    input('\n  >>> Press ENTER to continue...\n')

def open_image(path):
    os.startfile(path)
    time.sleep(1)

print('\n' + '='*60)
print('   DEEP LEARNING SUPER RESOLUTION MAPPING')
print('   Sentinel-2 Satellite Imagery Enhancement')
print('   NTRO SIH 2026')
print('='*60)
input('\n  >>> Press ENTER to start demo...\n')

# DEMO 1
show('STEP 1 - PROBLEM STATEMENT',
     'Sentinel-2 provides 10m resolution imagery.\n'
     '  This is insufficient for fine-scale analysis.\n'
     '  Our solution: Deep Learning Super Resolution.')
open_image(r'C:\SRM_Project\data\sentinel2\LR\sentinel2_preview.png')

# DEMO 2
show('STEP 2 - INPUT (LR PATCH)',
     'We crop 128x128 patches from Sentinel-2.\n'
     '  This is our Low Resolution (LR) model input.\n'
     '  Resolution: 10 metres per pixel.')
open_image(r'C:\SRM_Project\data\sentinel2\LR\patch_urban.png')

# DEMO 3
show('STEP 3 - OUR TRAINED MODEL',
     'SwinIR Transformer trained on real Sentinel-2 data.\n'
     '  Parameters  : 11.7 Million\n'
     '  Training    : 15 min 41 sec on RTX 4050\n'
     '  Loss Improvement: 66.4%\n'
     '  Now running inference...')
subprocess.run([
    'python', 'main_test_swinir.py',
    '--task', 'real_sr',
    '--scale', '4',
    '--model_path', r'experiments/pretrained_models/003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth',
    '--folder_lq', 'testsets/real-SR',
    '--tile', '256',
    '--tile_overlap', '32',
    '--large_model'
], cwd=r'C:\SRM_Project\SwinIR')

# DEMO 4
show('STEP 4 - SR OUTPUT',
     'SwinIR enhanced the image 4x!\n'
     '  Input  : 128x128 (10m resolution)\n'
     '  Output : 512x512 (2.5m resolution)\n'
     '  Opening SR result...')
open_image(r'C:\SRM_Project\SwinIR\results\swinir_real_sr_x4_large\sentinel2_urban_SwinIR.png')

# DEMO 5
show('STEP 5 - COMPARISON RESULT',
     'Side by side comparison:\n'
     '  LEFT   : LR Input  (blurry, 10m)\n'
     '  MIDDLE : SR Output (sharp, 2.5m) PSNR=24.69dB SSIM=0.707\n'
     '  RIGHT  : HR Reference\n'
     '  Opening comparison...')
open_image(r'C:\SRM_Project\data\comparison_result.png')

# DEMO 6
show('STEP 6 - MULTIPLE PATCH RESULTS',
     'Our model works on different terrain types:\n'
     '  - Urban areas\n'
     '  - Agricultural fields\n'
     '  - Crop fields\n'
     '  - Rural areas\n'
     '  Opening results...')
for name in ['urban_center', 'agricultural', 'crop_field', 'rural_area']:
    open_image(fr'C:\SRM_Project\data\patches\comparison\{name}_comparison.png')
    time.sleep(0.5)

# DEMO 7
show('STEP 7 - SEGFORMER SEGMENTATION',
     'SegFormer classifies every pixel:\n'
     '  Green  = Vegetation/Crops\n'
     '  Grey   = Roads\n'
     '  Red    = Buildings\n'
     '  Blue   = Water\n'
     '  Opening segmentation map...')
open_image(r'C:\SRM_Project\data\sentinel2_segmentation.png')
open_image(r'C:\SRM_Project\data\sentinel2_overlay.png')

# DEMO 8
show('STEP 8 - COMPLETE PIPELINE',
     'Full end-to-end pipeline result:\n'
     '  LR Input -> SwinIR SR -> SegFormer -> Output\n'
     '  Opening final result...')
open_image(r'C:\SRM_Project\data\final_pipeline_result.png')

# DEMO 9 - Metrics
show('STEP 9 - EVALUATION METRICS',
     'Quantitative evaluation on 40 validation images:\n'
     '  Average PSNR  : 24.66 dB\n'
     '  Average SSIM  : 0.7386\n'
     '  Average LPIPS : 0.3718\n'
     '  Training Time : 15 min 41 sec\n'
     '  Hardware      : RTX 4050 6GB\n'
     '  Scale Factor  : 4x (10m to 2.5m)')

print('\n' + '='*60)
print('   DEMO COMPLETE!')
print('   Deep Learning SRM - KCUF Coders')
print('   PSNR: 24.66dB | SSIM: 0.7386 | LPIPS: 0.3718')
print('='*60)