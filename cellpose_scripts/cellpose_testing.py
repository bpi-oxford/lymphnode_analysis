import os
from cellpose import models, io
import numpy as np
from skimage.io import imsave

if __name__ == "__main__":
    # Define paths
    input_image_path = r'/home/edwheeler/Documents/cropped_region_1/raw_frames/b2-2a_2c_pos6-01_deskew_cgt-cropped_for_segmentationt0.tif'  # Replace with the path to your input image
    output_masks_dir = r'/mnt/Work/Group Fritzsche/Ed/'  # Replace with the path to your output directory

    # Ensure output directory exists
    os.makedirs(output_masks_dir, exist_ok=True)

    # Load the image
    image = io.imread(input_image_path)
    print(image.shape)
    # Initialize Cellpose model
    custom_model_path = r"/home/edwheeler/Documents/training_data/train/models/CP_20250430_181517"
    model = models.CellposeModel(gpu=True, pretrained_model  = custom_model_path)  # Use 'cyto' or 'nuclei' depending on your data
    
    # Perform segmentation
    masks, flows, styles = model.eval(image, do_3D = True) #, flow3D_smooth=10) #, stitch_threshold =0.1)  # Adjust channels if needed

    # Save masks to the output directory
    mask_output_path = os.path.join(output_masks_dir, 'segmented_masks.tif')
    im_output_path   = os.path.join(output_masks_dir, 'input_image.tif')
    imsave(mask_output_path, masks.astype(np.uint16))  # Save masks as 16-bit TIFF
    imsave(im_output_path  , image)

    print(f"Masks saved to {mask_output_path}")