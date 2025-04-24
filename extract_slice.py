import sys
import visor.image as vsr_img

visor_path = r'/share/data/VISoR_Data/vsr/N1779.vsr'
visor_target_path = r'/share/data/VISoR_TestData/zarr_test/vsr/N1779_cut1_xyy.vsr'

vimg = vsr_img.open_vsr(visor_path, 'r')
print(vimg.info)

slice_idx = 19

# Read raw image of slice_1_10x from zarr file
# arr is a visor.Array object
varr = vimg.read(img_type='raw',
                 zarr_file=vimg.image_files['raw'][slice_idx]['path'],
                 resolution=0)

print(vars(varr).keys())

print(varr.info)

if 0:
    sys.exit(0)

img_info = vimg.info
arr_info = varr.info
selected = [vimg.image_files['raw'][slice_idx]]

vimg_t = vsr_img.open_vsr(visor_target_path, 'w')
vimg_t.write(arr = varr.array, img_type = 'raw',
             file = 'slice_20_10x', resolution = 0,
             img_info=img_info, arr_info=arr_info, selected=selected)
