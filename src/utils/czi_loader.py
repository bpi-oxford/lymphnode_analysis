
from aicsimageio import AICSImage
import tifffile as tiff
import os
import numpy as np

# Load the .czi file
image_path = r'/users/kir-fritzsche/aif490/devel/tissue_analysis/CARE/cycleCARE/data/validation/overview_volume_10ms_10pc647_2pc488-03__deskew_cgt.czi'
# Get an AICSImage object
img = AICSImage(image_path)  # selects the first scene found


# Pull only a specific chunk in-memory
lazy_c0 = img.get_image_dask_data("TCZYX" , C=0) # Select the first channel explicitly
print(lazy_c0.shape)  # returns 4D dask array

shape  = lazy_c0.shape
tlim = slice(0, shape[0])
zlim = slice(79, 79 + 256)
ylim = slice(960+256, 960+512)
xlim = slice(1248+256, 1248+512)



# Slice the already-loaded dask array (axes: T, Z, Y, X)
lazy_c0_crop = lazy_c0[tlim, :, zlim, ylim, xlim]
print(lazy_c0_crop.shape)  # returns 4D dask array
crop = lazy_c0_crop.compute() 
print(crop.shape)

# Save cropped data as a tiff stack
output_dir = r'/users/kir-fritzsche/aif490/devel/tissue_analysis/lymphnode_analysis/data2validate/crop2_ordered'
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, f'overview_volume_10ms_10pc647_2pc488-03__deskew_cgt_actin{tlim.start}-{tlim.stop}_z{zlim.start}-{zlim.stop}_y{ylim.start}-{ylim.stop}_x{xlim.start}-{xlim.stop}.tiff')
tiff.imwrite(output_path, crop.astype(np.uint16))


