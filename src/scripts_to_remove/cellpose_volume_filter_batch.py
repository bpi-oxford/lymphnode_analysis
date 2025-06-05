import tifffile as tiff
import numpy as np
import os
from scripts_to_remove.cellpose_volume_analyser import calculate_volumes , filter_volumes
from natsort import natsorted
from multiprocessing import Pool, cpu_count

'''
This scripts filters segmentation masks based on a volume limit. Designed to be used on 
a direcotry that contains segmentation masks in .tif format - in individual frame z-stacks.
Now I believe this is redundant as modifying the min_size parameter in the cellpose model should do this.
'''

def filter_batch(directory, vol_limit_list):
    image_paths = natsorted([os.path.join(directory, f) for f in os.listdir(directory) if f.endswith("seg_masks.tif")])
    print(f"Located {len(image_paths)} images.")


    z_stack = tiff.imread(image_paths[0])
    shape = z_stack.shape
    filtered_stack =  np.zeros( (len(image_paths) , shape[0] , shape[1] , shape[2]))
    for image_path in image_paths:
            print(image_path)
            z_stack = tiff.imread(image_path)
            # Calculate volumes
            volumes = calculate_volumes(z_stack, voxel_size=0.145**3)
            filtered_masks = filter_volumes(z_stack, volumes, vol_limit_list)
            output_file_path = image_path[0:-4] + '_minsize_' + str(vol_limit_list[0]) + '_filtered.tif'
            tiff.imwrite(output_file_path, filtered_masks)
            filtered_stack[counter,...] = filtered_masks
            counter +=1
            print(f"Processed {counter} images.")

    return filtered_stack

def filter_file(filepath, vol_limi_list):
    z_stack = tiff.imread(filepath)
    volumes = calculate_volumes(z_stack, voxel_size=0.145**3)
    filtered_masks = filter_volumes(z_stack, volumes, vol_limit_list)
    return filtered_masks

def process_image(args):
    image_path, vol_limit_list = args
    print('processing' + image_path)
    z_stack = tiff.imread(image_path)
    # Calculate volumes
    volumes = calculate_volumes(z_stack, voxel_size=0.145**3)
    filtered_masks = filter_volumes(z_stack, volumes, vol_limit_list)
    output_file_path = image_path[0:-4] + '_minsize_' + str(vol_limit_list[0]) + '_filtered.tif'
    tiff.imwrite(output_file_path, filtered_masks)
    print(f"Processed {image_path}")
    return filtered_masks

def filter_batch_MPI(image_paths, vol_limit_list):
   
    print(f"Located {len(image_paths)} images.")

    with Pool(8) as pool:
        args = [(image_path, vol_limit_list) for image_path in image_paths]
        filtered_stacks = pool.map(process_image, args)

    return np.array(filtered_stacks)

if __name__ == "__main__":
    directory = r"/home/edwheeler/Documents/cropped_region_3_mixed"
    image_paths = natsorted([os.path.join(directory, f) for f in os.listdir(directory) if f.endswith("seg_masks.tif")])
    image_paths = image_paths[0:5]
    vol_limit_list = [10, 100000]  # Example volume limits in um^3
    filtered_stack = filter_batch_MPI(image_paths, vol_limit_list)
    print(filtered_stack.shape)
