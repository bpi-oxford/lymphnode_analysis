from cellpose import models
import numpy as np
import tifffile as tiff

def run_segmentation(image, nuclei_image, custom_model_path , diameter = 30):
    """
    Runs segmentation on a 2-channel image using Cellpose.

    Parameters:
        image: raw image .
        nuclei_image: 'nuclei' image - i.e. a cell interior segmentation attempt.
        model_path (str): Path to the Cellpose model.
        diameter (float or None): Estimated diameter of objects to segment. If None, Cellpose will auto-estimate.

    Returns:
        masks (numpy.ndarray): Segmentation masks for the image.
    """
    # First combine the two images into a 2-channel image
    combined_image = np.stack((image, nuclei_image), axis=0)

    # Load the Cellpose model
    model = models.CellposeModel(gpu=True, pretrained_model=custom_model_path)
    masks, outlines, flows = model.eval(combined_image, flow3D_smooth=2, batch_size=8, do_3D=True, diameter=diameter, min_size=500, channels = [1,2] ) #,z_axis = 0, channels=[0,0])

    return masks

def filter_secondary_segmentation(masks, label_list):
    # Create a new mask initialized to zero
    filtered_masks = np.zeros_like(masks)

    # Iterate through the label list and keep only the specified labels
    for label in label_list:
        filtered_masks[masks == label] = label

    return filtered_masks

if __name__ == "__main__":
    # Example usage
    raw_image_path = r'/home/edwheeler/Documents/cropped_region_1/raw_video/b2-2a_2c_pos6-01_deskew_cgt-cropped_for_segmentation.tif'
    raw_image = tiff.imread(raw_image_path)
    print(raw_image.shape)
    seg1_image_path = r'/home/edwheeler/Documents/tissue_analysis_project/tracking_benchmarking/outputs/relabelled_crop1_ultrack15noblur_reshape.tif'
    seg1_image = tiff.imread(seg1_image_path)
    custom_model_path = r'/home/edwheeler/Documents/cyto_model_test/train_input/train/models/models/CP_20250521_114246'
    output_dir = r'/mnt/Work/Group Fritzsche/Ed/'   

    print(seg1_image.shape)
    print(np.max(seg1_image))
    print(np.max(seg1_image))
    
    label_list = [53,79]

    seg2 = run_segmentation(raw_image, seg1_image, custom_model_path, diameter=30)
    print(seg2.shape)

    filtered_seg2 = filter_secondary_segmentation(seg2, label_list)
    print(filtered_seg2.shape)

    output_file_path = output_dir + 'secondary_segmentation_masks_filtered.tif'
    tiff.imwrite(output_file_path, filtered_seg2.astype(np.uint16))


