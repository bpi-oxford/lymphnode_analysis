
from aicsimageio import AICSImage
import tifffile as tiff
import os
import subprocess
import numpy as np

# Load the .czi file
#image_path = r"/mnt/Ceph/annavschepers/Lattice_data/2024-11-21_mouse_LNs_lifeact_aCD3_B220/compressed_deskewed/b2-2a_2c_pos6-01_deskew_cgt.czi"
image_path = r'/mnt/Ceph/annavschepers/Lattice_data/2024-11-21_mouse_LNs_lifeact_aCD3_B220/compressed_deskewed/b2-6a_overview_pos1-01_deskew_cgt.czi'
# Get an AICSImage object
img = AICSImage(image_path)  # selects the first scene found


# Pull only a specific chunk in-memory
lazy_c0 = img.get_image_dask_data("TCZYX" , C=0) # Select the first channel explicitly
print(lazy_c0)  # returns 4D dask array

T_range = [1] #np.arange(80)
print(T_range)

for T in T_range:
    print('processing' + str(T))

    z_first = 70
    z_last = 200
    x_first = 1191
    x_width = 256
    y_first = 615
    y_width = 256
    T_first = 0 #T
    T_last =  80 #T+1

    zlim = slice(z_first , z_last)
    xlim = slice(x_first, x_first + x_width)
    ylim = slice(y_first, y_first + y_width)
    tlim = slice(T_first , T_last)
    print(ylim)



    lazy_c0_crop = img.get_image_dask_data("TCZYX", T = tlim , Z=zlim , Y=ylim, X=xlim)  # returns out-of-memory 4D dask array
    print(lazy_c0_crop)  # returns 4D dask array
    crop = lazy_c0_crop.compute() 
    print(crop.shape)
    #crop = crop[0, 0, :, :, :]

    # Save the cropped image as a TIFF file in the current directory
    output_filename = os.path.basename(image_path).replace(".czi", "_lymphnode2_crop1_t" + str(T) + ".tiff")
    tiff.imwrite(output_filename, crop)

    # Define the external directory to copy the file to
    external_dir = "/mnt/Work/Group Fritzsche/Ed"

    # Use subprocess to copy the file to the external directory
    subprocess.run(["cp", output_filename, external_dir], check=True)


