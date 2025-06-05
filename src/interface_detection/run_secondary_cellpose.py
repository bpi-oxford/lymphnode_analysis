from cellpose import models
import numpy as np
import tifffile as tiff

def run_segmentation(image, nuclei_image, custom_model_path , diameter = 30):
    """
    Runs segmentation on a 2-channel image using Cellpose.
    Check the diameter?

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

    # First combine the two images into a 2-channel image
    combined_image = np.stack((image, nuclei_image), axis=0)

    # Load the Cellpose model
    model = models.CellposeModel(gpu=True, pretrained_model=custom_model_path)
    masks, outlines, flows = model.eval(combined_image, flow3D_smooth=2, batch_size=8, do_3D=True, diameter=diameter, min_size=500, channels = [1,2] ) #,z_axis = 0, channels=[0,0])
    tiff.imwrite(r'/mnt/Work/Group Fritzsche/Ed/segmentation_masks.tif', masks.astype(np.uint16))
    return masks

def filter_secondary_segmentation(masks, nuclei_image, label_list):

    # Ensure the masks and nuclei_image have the same shape
    if masks.shape != nuclei_image.shape:
        raise ValueError(f"Shape mismatch: masks shape {masks.shape}, nuclei_image shape {nuclei_image.shape}")

    # Create a new mask initialized to zero
    filtered_nuclei_masks = np.zeros_like(nuclei_image)

    # Iterate through the label list and keep only the specified labels
    for label in label_list:
        filtered_nuclei_masks[nuclei_image == label] = label
    

    #now find the corresponding masks in the segmentation masks
    cyto_labels = []
    for label in np.unique(filtered_nuclei_masks):
        if label == 0:
            continue
        else:
            print(f"Processing label {label}")
            # Find the indices of the centre of the current label in the filtered nuclei masks
            coords = np.argwhere(filtered_nuclei_masks == label)
            if coords.size == 0:
                continue
            centroid = coords.mean(axis=0).astype(int)
            # Get the corresponding label in the masks
            cyto_labels.append(masks[centroid[0], centroid[1], centroid[2]])
    
    #now filter the masks to only include the cyto_labels
    filtered_masks = np.zeros_like(masks)
    for cyto_label in cyto_labels:
        filtered_masks[masks == cyto_label] = cyto_label

    print(cyto_labels)

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

    segmentation_stack = seg_video #[timepoints_with_labels, ...]
    raw_video = raw_video #[timepoints_with_labels, ...]

    print('time points identified with all labels:', timepoints_with_labels)
    # Initialize an empty list to store interfaces
    filtered_mask_stack = np.zeros_like(segmentation_stack)

    # Iterate through each time point in the segmentation stack
    for t in timepoints_with_labels:
        print(f"Processing time point {t} ")
        current_seg= segmentation_stack[t]
        current_raw = raw_video[t]
        all_masks = run_segmentation(current_raw, current_seg, custom_model_path=custom_model_path, diameter=30)  # Assuming custom_model_path is defined elsewhere

        filtered_masks = filter_secondary_segmentation(all_masks, current_seg, label_list)
        # Calculate the centroid of the filtered masks
        unique_labels = np.unique(filtered_masks)
        for label in unique_labels:
            if label == 0:  # Skip background
                continue
            coords = np.argwhere(filtered_masks == label)
            centroid = coords.mean(axis=0)
            print(f"Time {t}, Label {label}, Centroid: {centroid}")
        filtered_mask_stack[t] = filtered_masks
        

    return filtered_mask_stack

if __name__ == "__main__":
    # Example usage
    raw_image_path = r'/home/edwheeler/Documents/raw_data/cropped_region_1/raw_video/b2-2a_2c_pos6-01_deskew_cgt-cropped_for_segmentation.tif'
    raw_image = tiff.imread(raw_image_path)
    raw_image = raw_image[0:4, 0:120, ...] #the segmentation is only 120 in z
    print(raw_image.shape)
    seg1_image_path = r'/home/edwheeler/Documents/tissue_analysis_project/tracking_benchmarking/outputs/relabelled_crop1_ultrack15noblur_reshape.tif'
    seg1_image = tiff.imread(seg1_image_path)
    seg1_image = seg1_image[0:4, 0:120, ...] #the segmentation is only 120 in z
    print(seg1_image.shape)
    custom_model_path = r'/home/edwheeler/Documents/cyto_model_test/train_input/train/models/models/CP_20250521_114246'
    output_dir = r'/mnt/Work/Group Fritzsche/Ed/'   
    label_list = [53,79]

    seg2 = interface_over_time(raw_video=raw_image, seg_video=seg1_image, custom_model_path=custom_model_path, label_list=label_list)

    output_file_path = output_dir + 'secondary_segmentation_masks_filtered.tif'
    tiff.imwrite(output_file_path, seg2.astype(np.uint16))


