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
    # Ensure the image and nuclei_image have the same shape
    if image.shape != nuclei_image.shape:
        print(f"Shape mismatch: image shape {image.shape}, nuclei_image shape {nuclei_image.shape}")
    assert image.shape == nuclei_image.shape, "The image and nuclei_image must have the same shape."
    # Ensure the image and nuclei_image have the same shape
    assert image.shape == nuclei_image.shape, "The image and nuclei_image must have the same shape."

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

def interface_over_time(raw_video, seg_video, custom_model_path,label_list):
    """
    Extracts interfaces over time from a 4D raw video and segmentation video.

    Parameters:
        raw_video (numpy.ndarray): 4D array of raw_data.
        seg_video (numpy.ndarray): 4D array of segmentation masks.
        label_list (list): List of labels to extract interfaces for.

    Returns:
        interfaces (numpy.ndarray): 4D array of cells in the label list segmented over time.
    """

    #Initially, find the time points in the seg video where the labels are present
    timepoints_with_labels = []
    for t in range(seg_video.shape[0]):
        current_frame = seg_video[t]
        if all(label in np.unique(current_frame) for label in label_list):
            timepoints_with_labels.append(t)

    segmentation_stack = seg_video[timepoints_with_labels]
    raw_video = raw_video[timepoints_with_labels]

    print('time points identified with labels:', timepoints_with_labels)
    # Initialize an empty list to store interfaces
    filtered_mask_stack = np.zeros_like(segmentation_stack)

    # Iterate through each time point in the segmentation stack
    for t in timepoints_with_labels:
        print(f"Processing time point {t} ")
        current_seg= segmentation_stack[t]
        current_raw = raw_video[t]
        all_masks = run_segmentation(current_raw, current_seg, custom_model_path=custom_model_path, diameter=30)  # Assuming custom_model_path is defined elsewhere
        filtered_masks = filter_secondary_segmentation(all_masks, label_list)
        filtered_mask_stack[t] = filtered_masks
        

    return filtered_mask_stack

if __name__ == "__main__":
    # Example usage
    raw_image_path = r'/home/edwheeler/Documents/raw_data/cropped_region_1/raw_video/b2-2a_2c_pos6-01_deskew_cgt-cropped_for_segmentation.tif'
    raw_image = tiff.imread(raw_image_path)
    raw_image = raw_image[:, 0:120, ...] #the segmentation is only 120 in z
    print(raw_image.shape)
    seg1_image_path = r'/home/edwheeler/Documents/tissue_analysis_project/tracking_benchmarking/outputs/relabelled_crop1_ultrack15noblur_reshape.tif'
    seg1_image = tiff.imread(seg1_image_path)
    custom_model_path = r'/home/edwheeler/Documents/cyto_model_test/train_input/train/models/models/CP_20250521_114246'
    output_dir = r'/mnt/Work/Group Fritzsche/Ed/'   
    label_list = [53,79]

    seg2 = interface_over_time(raw_video=raw_image, seg_video=seg1_image, custom_model_path=custom_model_path, label_list=label_list)

    output_file_path = output_dir + 'secondary_segmentation_masks_filtered.tif'
    tiff.imwrite(output_file_path, seg2.astype(np.uint16))


