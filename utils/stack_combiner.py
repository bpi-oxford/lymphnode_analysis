import tifffile as tiff
import os
from natsort import natsorted
import numpy as np

def find_matching_files(directory):
    matching_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if "gamma1t" in file and "seg_masks" in file:
                matching_files.append(os.path.join(root, file))
    return matching_files


if __name__ == "__main__":
    directory = r"/home/edwheeler/Documents/cropped_region_1/raw_video"
    matching_files = find_matching_files(directory)
    matching_files = natsorted(matching_files)  # Sort files in natural order using natsort
    stacks = [tiff.imread(file) for file in matching_files]
    print(f"Loaded {len(stacks)} stacks.")

    file_1 = tiff.imread(matching_files[0])
    shape = file_1.shape
    num_t_steps = len(stacks)

    master_stack = np.zeros((num_t_steps, shape[0], shape[1],shape[2]), dtype=np.uint16)
    for t in range(num_t_steps):
        master_stack[t, ...] = stacks[t]
    
    tiff.imwrite(r"/home/edwheeler/Documents/cropped_region_1/raw_video/combined_gamma1_stack.tif", master_stack.astype(np.uint16))