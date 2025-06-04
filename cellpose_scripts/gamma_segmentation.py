'''
Try and use the gamma transform to provide multiple segmentation outputs for ultrack to use
'''
import numpy as np
import tifffile as tiff
import pandas as pd
import matplotlib.pyplot as plt
from cellpose_scripts.cellpose_batch import process_video_with_multiprocessing

def gamma_transform(image, gamma):
    """
    Apply gamma transformation to the image.
    """
    # Normalize the image to [0, 1]
    image_normalized = (image - np.min(image)) / (np.max(image) - np.min(image))
    
    # Apply gamma transformation
    transformed_image = np.power(image_normalized, gamma)
    
    # Rescale back to original range
    transformed_image = transformed_image * (np.max(image) - np.min(image)) + np.min(image)
    
    return transformed_image


if __name__ == "__main__":
    # Load the image
    image_path = r"/home/edwheeler/Documents/cropped_region_2_motile/b2-2a_2c_pos6-01_deskew_cgt_cropped_for_segmentation_motile_region.tif"
    image = tiff.imread(image_path)
    custom_model_path = r"/home/edwheeler/Documents/training_data/train/models/CP_20250430_181517"

    num_t_steps = image.shape[0]
    # Define gamma values
    gamma_values = [1.25, 1.5, 1.75]

    for gamma in gamma_values:
        gamma_image = gamma_transform(image, gamma)
        gamma_path = image_path[0:-4] + 'gamma' + str(gamma) + '.tif'

        gamma_masks = process_video_with_multiprocessing(gamma_image , gamma_path, custom_model_path)
        print(gamma_masks[0].shape)
        print(gamma_masks)

        save_gamma_path = gamma_path[0:-4] + '_seg_masks_stack.tif'
        tiff.imwrite(save_gamma_path , gamma_masks.astype(np.uint16))
        print(f"Gamma transformed masks saved to {save_gamma_path}")

