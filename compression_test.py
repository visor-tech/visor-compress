# demo: compressing a visor image
# pip install -e /home/xyy/code/py/visor-py

import os
import sys
import time
import numpy as np
import visor.image as vsr_img
import numcodecs
from numcodecs_ffmpeg import ffmpeg_codec

def compress_dataset(visor_path, img_tags, compressor, overwrite=False):
    """Compress whole visor image dataset"""
    # TODO: with filter

    vimg_r = vsr_img.open_vsr(visor_path, 'r')
    img_info = vimg_r.info
    selected = vimg_r.image_files['raw']

    vimg_c = vsr_img.open_vsr(visor_path, 'w')

    # loop through all the slices
    for slice_fc in selected:
        print('Processing', slice_fc['path'])
        slice_name = slice_fc['path'].split('.')[0]
        res_id = 0  # TODO: loop over resolutions
        varr = vimg_r.read(
            img_type='raw',
            zarr_file=f'{slice_name}.zarr',
            resolution=res_id)

        # metadata
        arr_info = varr.info
        arr_info['multiscales'][0]['datasets'] = [varr.info['multiscales'][0]['datasets'][res_id]]
        arr_info['sources'] = [slice_fc]

        slice_name_c = f'{img_tags["person"]}_{slice_name}_{img_tags["date"]}'

        vimg_c.write(
            arr = varr.array, img_type = 'compr',
            file = slice_name_c, resolution = res_id,
            img_info=img_info, arr_info=arr_info, selected=selected,
            compressor=compressor,
            overwrite=overwrite)

def get_total_size(directory):
    total_size = 0
    n_files = 0
    for dirpath, dirnames, filenames in os.walk(directory):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            total_size += os.path.getsize(file_path)
            n_files += 1
    return (total_size, n_files)

def evaluation(visor_path, t_elapsed):
    # compare file
    file_stat_orig = get_total_size(os.path.join(visor_path, 'visor_raw_images'))
    file_stat_cmpr = get_total_size(os.path.join(visor_path, 'visor_compr_images'))

    print('Data size:')
    print(f'  Original  : {file_stat_orig[0]} bytes, {file_stat_orig[1]} files'
          f', average size: {file_stat_orig[0] / file_stat_orig[1]:.0f} bytes')
    print(f'  Compressed: {file_stat_cmpr[0]} bytes, {file_stat_cmpr[1]} files'
          f', average size: {file_stat_cmpr[0] / file_stat_cmpr[1]:.0f} bytes')
    print(f'  Compression ratio (before/after): {file_stat_orig[0] / file_stat_cmpr[0]:.2f}')
    print(f'Time cost:\n'
          f'  wall time: {t_elapsed:.2f} s,\n'
          f'  per file : {t_elapsed / file_stat_orig[1]:.2g} s average')
    print(f'Equivalent throughput:\n'
          f'  input  side {file_stat_orig[0] / t_elapsed / 1024 / 1024:.3g} MiB/s\n'
          f'  output side {file_stat_cmpr[0] / t_elapsed / 1024 / 1024:.3g} MiB/s')

    """ ffmpeg
    Original file size: 5749452852 bytes, 1540 files
    Compressed file size: 21643933 bytes, 1539 files
    Compression ratio: 0.00
    Time cost: 332.84 s
    
    ffmpeg crf 28 threads all
    Data size:
        Original  : 5749452852 bytes, 1540 files, average size: 3733411 bytes
        Compressed: 21597809 bytes, 1539 files, average size: 14034 bytes
        Compression ratio (after/before): 0.00376
    Time cost:
        wall time: 325.28 s,
        per file : 0.21 s average
    Equivalent throughput:
        input  side 16.9Data size:
        output side 0.0633 MiB/s

    {'crf': 28, 'preset': 'fast', 'threads': 1}
    Original  : 5749452852 bytes, 1540 files, average size: 3733411 bytes
      Compressed: 21593293 bytes, 1539 files, average size: 14031 bytes
      Compression ratio (after/before): 0.00376
    Time cost:
      wall time: 313.24 s,
      per file : 0.2 s average
    Equivalent throughput:
      input  side 17.5 MiB/s
      output side 0.0657 MiB/s MiB/s
    
    
    """

def calculate_ssim(img1, img2):
    """Calculate SSIM between two images with support for multi-dimensional arrays."""
    if img1.dtype == np.uint16:
        max_pixel = 65535.0
    elif img1.dtype == np.uint8:
        max_pixel = 255.0
    else:
        raise ValueError("Unsupported image dtype. Only uint8 and uint16 are supported.")

    # Convert to float32, preserving range
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)

    # Constants as per the SSIM paper
    k1 = 0.01
    k2 = 0.03
    # Dynamic range
    L = max_pixel
    c1 = (k1 * L) ** 2
    c2 = (k2 * L) ** 2

    # Calculate means over all dimensions
    mu1 = np.mean(img1).compute()
    mu2 = np.mean(img2).compute()

    # Calculate variances and covariance over all dimensions
    var1 = np.mean((img1 - mu1) ** 2).compute()
    var2 = np.mean((img2 - mu2) ** 2).compute()
    covar = np.mean((img1 - mu1) * (img2 - mu2)).compute()

    # Calculate SSIM
    # ((2*mu1*mu2 + C1)*(2*covar + C2))/((mu1**2 + mu2**2 + C1)*(var1 + var2 + C2))
    numerator = (2 * mu1 * mu2 + c1) * (2 * covar + c2)
    denominator = (mu1**2 + mu2**2 + c1) * (var1 + var2 + c2)
    
    ssim = numerator / denominator
    return ssim

def calculate_psnr(img1, img2):
    """Calculate PSNR between two images."""
    # Original version used: max_pixel = np.max(img1)
    # Using fixed value 255 to match OpenCV's implementation for 8-bit images
    if img1.dtype == np.uint16:
        max_pixel = 65535.0
    elif img1.dtype == np.uint8:
        max_pixel = 255.0
    else:
        raise ValueError("Unsupported image dtype. Only uint8 and uint16 are supported.")

    # Convert to float32 (matching OpenCV's internal implementation)
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)
    
    # Calculate MSE over all dimensions
    mse = np.mean((img1 - img2) ** 2)
    
    if mse == 0:
        return float('inf')
    
    # use 
    # max_pixel = img1.max()
    
    # Calculate PSNR (same formula as OpenCV)
    psnr = 20 * np.log10(max_pixel) - 10 * np.log10(mse)
    return psnr

def test_err(visor_path):
    """Test error between raw and compressed images using PSNR."""
    # Open both raw and compressed images
    vimg_raw = vsr_img.open_vsr(visor_path, 'r')
    selected = vimg_raw.image_files['raw']
    
    # Statistics storage
    psnr_values = []
    ssim_values = []
    
    # Loop through all slices
    for slice_fc in selected:
        slice_name = slice_fc['path'].split('.')[0]
        res_id = 0  # Using first resolution level
        
        # Read raw image
        varr_raw = vimg_raw.read(
            img_type='raw',
            zarr_file=f'{slice_name}.zarr',
            resolution=res_id)
        raw_array = varr_raw.array
        
        # Read compressed image
        hostname = os.uname().nodename
        date = time.strftime("%Y%m%d")
        slice_name_c = f'{hostname}_{slice_name}_{date}'
        try:
            varr_comp = vimg_raw.read(
                img_type='compr',
                zarr_file=f'{slice_name_c}.zarr',
                resolution=res_id)
            comp_array = varr_comp.array
            
            # Calculate PSNR and SSIM
            psnr = calculate_psnr(raw_array, comp_array).compute()
            ssim = calculate_ssim(raw_array, comp_array)
            psnr_values.append(psnr)
            ssim_values.append(ssim)
            print(f'PSNR for {slice_name}: {psnr:.2f} dB, SSIM: {ssim:.4f}')
        except Exception as e:
            print(f"Error processing {slice_name_c}: {e}")
    
    if psnr_values:
        print("\nImage quality evaluation:")
        print("Image size: ", raw_array.shape)
        print("\nPSNR Statistics:")
        print(f"Mean PSNR: {np.mean(psnr_values):.2f} dB")
        print(f"Min PSNR: {np.min(psnr_values):.2f} dB")
        print(f"Max PSNR: {np.max(psnr_values):.2f} dB")
        print(f"Std PSNR: {np.std(psnr_values):.2f} dB")
        print("\nSSIM Statistics:")
        print(f"Mean SSIM: {np.mean(ssim_values):.4f}")
        print(f"Min SSIM: {np.min(ssim_values):.4f}")
        print(f"Max SSIM: {np.max(ssim_values):.4f}")
        print(f"Std SSIM: {np.std(ssim_values):.4f}")

if __name__ == '__main__':
    visor_path = r'/share/data/VISoR_TestData/zarr_test/vsr/N1779_cut1_xyy.vsr'

    if 1:
        compressor = ffmpeg_codec(crf=28, preset='fast', threads=1)
    else:
        compressor = numcodecs.Blosc(cname='zstd', clevel=1)
    overwrite = True

    img_tags = {
        'person': os.uname().nodename, # hostname
        'date': time.strftime("%Y%m%d") # current date
    }

    print(f'Compressor: {compressor}')
    t1 = time.time()
    #compress_dataset(visor_path, img_tags, compressor, overwrite)
    t2 = time.time()
    print("File path:", visor_path)
    evaluation(visor_path, t2-t1)
    test_err(visor_path)

"""
File path: /share/data/VISoR_TestData/zarr_test/vsr/N1779_cut1_xyy.vsr
Data size:
  Original  : 5749452852 bytes, 1540 files, average size: 3733411 bytes
  Compressed: 21593293 bytes, 1539 files, average size: 14031 bytes
  Compression ratio (before/after): 266.26

Image quality evaluation:
Image size:  (4, 1, 2858, 788, 2048)

PSNR Statistics:
Mean PSNR: 43.76 dB
Min PSNR: 43.76 dB
Max PSNR: 43.76 dB
Std PSNR: 0.00 dB

SSIM Statistics:
Mean SSIM: 0.8435
Min SSIM: 0.8435
Max SSIM: 0.8435
Std SSIM: 0.0000
"""