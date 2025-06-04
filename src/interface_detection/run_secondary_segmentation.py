from cellpose import models
import numpy as np
import tifffile as tiff
from skimage.measure import regionprops

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
    masks, outlines, flows = model.eval(combined_image, flow3D_smooth=2, batch_size=8, do_3D=True, diameter=diameter, min_size=25, channels = [1,2] ) #,z_axis = 0, channels=[0,0])

    return masks

def focus_on_label(raw_image, labelled_image, label, padding):
    '''
    finds bounding box of label and crops the images down to that size to make segmentation easier,
    with an added padding around the bounding box.
    '''
    # Create a binary mask for the specified label
    single_labelled_image = np.zeros_like(labelled_image, dtype=np.uint8)
    single_labelled_image[labelled_image == label] = 1
    print(np.max(single_labelled_image))
    # Get properties of labeled regions
    props = regionprops(single_labelled_image)
    for prop in props:
        if prop.label == 1:  # Since the image is already binarized, the label will always be 1
            print('cell found')
            print(prop.bbox)
            min_depth, min_row, min_col, max_depth, max_row, max_col = prop.bbox
        else:
            print('no cell found with that label!')
            return None , None, None, None
    
    # Apply padding
    min_depth = max(min_depth - padding, 0)
    min_row = max(min_row - padding, 0)
    min_col = max(min_col - padding, 0)
    max_depth = min(max_depth + padding, raw_image.shape[0])
    max_row = min(max_row + padding, raw_image.shape[1])
    max_col = min(max_col + padding, raw_image.shape[2])

    cropped_raw_image = raw_image[min_depth:max_depth, min_row:max_row, min_col:max_col]
    cropped_label_image = labelled_image[min_depth:max_depth, min_row:max_row, min_col:max_col]

    # Calculate the centroid of the region
    centroid = prop.centroid
    bbox = ( min_depth, min_row, min_col, max_depth, max_row, max_col)
    return cropped_raw_image, cropped_label_image ,bbox, centroid


def focus_on_two_cells(raw_image , labelled_image, label_list, padding):
    # Create a binary mask for the labels in label_list
    filtered_labelled_image = np.zeros_like(labelled_image, dtype=np.uint8)
    for label in label_list:
        filtered_labelled_image[labelled_image == label] = label

    # Update the labelled_image to only include the specified labels
    labelled_image[:] = filtered_labelled_image

    # Create a binary mask for the labels in label_list
    combined_labelled_image = np.zeros_like(labelled_image, dtype=np.uint8)
    for label in label_list:
        combined_labelled_image[labelled_image == label] = 1

    # Get properties of labeled regions
    props = regionprops(combined_labelled_image)
    print(len(props))
    if len(props) == 0:
        print('No regions found for the given labels!')
        return None, None, None, None

    # Combine bounding boxes of all regions
    min_depth = min(prop.bbox[0] for prop in props)
    min_row   = min(prop.bbox[1] for prop in props)
    min_col   = min(prop.bbox[2] for prop in props)
    max_depth = max(prop.bbox[3] for prop in props)
    max_row   = max(prop.bbox[4] for prop in props)
    max_col   = max(prop.bbox[5] for prop in props)

    # Apply padding taking edges into account
    min_depth = max(min_depth - padding, 0)
    min_row   = max(min_row - padding, 0)
    min_col   = max(min_col - padding, 0)
    max_depth = min(max_depth + padding, raw_image.shape[0])
    max_row   = min(max_row + padding, raw_image.shape[1])
    max_col   = min(max_col + padding, raw_image.shape[2])
    # Crop the raw and labelled images around the combined bounding box
    cropped_raw_image   = raw_image[min_depth:max_depth, min_row:max_row, min_col:max_col]
    cropped_label_image = labelled_image[min_depth:max_depth, min_row:max_row, min_col:max_col]

    # Calculate the centroid of the combined region
    centroids = [prop.centroid for prop in props]
    combined_centroid = tuple(np.mean(centroids, axis=0))
    bbox = (min_depth, min_row, min_col, max_depth, max_row, max_col)
    return cropped_raw_image, cropped_label_image, bbox, combined_centroid


def return_crop_to_image(large_image , crop , bbox):
    """
    Places a cropped region back into its original position in the larger image.

    Parameters:
        large_image (numpy.ndarray): The original large image.
        crop (numpy.ndarray): The cropped region to be placed back.
        bbox (tuple): The bounding box of the cropped region in the format 
                      (min_depth, min_row, min_col, max_depth, max_row, max_col).
        centroid (tuple): The centroid of the region (optional, not used here).

    Returns:
        updated_image (numpy.ndarray): The large image with the cropped region placed back.
    """
    min_depth, min_row, min_col, max_depth, max_row, max_col = bbox

    # Ensure the crop fits within the bounding box
    crop_depth, crop_height, crop_width = crop.shape
    assert crop_depth == (max_depth - min_depth), "Crop depth does not match bounding box depth."
    assert crop_height == (max_row - min_row), "Crop height does not match bounding box height."
    assert crop_width == (max_col - min_col), "Crop width does not match bounding box width."

    # Place the crop back into the large image
    updated_image = large_image.copy()
    updated_image[min_depth:max_depth, min_row:max_row, min_col:max_col] = crop

    return updated_image


def extract_central_cell_from_mask(labelled_image):
        """
        Extracts the central cell from a labelled image by identifying the label closest to the center.

        Parameters:
            labelled_image (numpy.ndarray): The labelled image.

        Returns:
            central_cell_mask (numpy.ndarray): A binary mask containing only the central cell.
        """
        # Get the center of the image
        center = np.array(labelled_image.shape) // 2

        # Get properties of labeled regions
        props = regionprops(labelled_image)

        # Find the label closest to the center
        '''
        min_distance = float('inf')
        central_label = None
        for prop in props:
            centroid = np.array(prop.centroid)
            distance = np.linalg.norm(centroid - center)
            if distance < min_distance:
                min_distance = distance
                central_label = prop.label
        '''

        central_label = labelled_image[center[0], center[1], center[2]]

        # Create a binary mask for the central cell
        central_cell_mask = np.zeros_like(labelled_image, dtype=np.uint8)
        central_cell_mask[labelled_image == central_label] = 1
        return central_cell_mask

def cells_from_label_list(labelled_image, returned_crop ,label_list ):
    '''
    Identifies the two cells of interest from the labelled image and the segmented crop
    that has been returned to the same strucutre as the labelled_image
    '''

    # Convert labelled_image into a uint16 array
    labelled_image = labelled_image.astype(np.uint16)
    filtered_crop = np.zeros_like(returned_crop, dtype=np.uint16)
    labels_to_keep = []
    for label in label_list:
        single_labelled_image = (labelled_image == label).astype(np.uint8) * label
        print(np.max(single_labelled_image))
        # Find the centroid of the region in labelled_image that corresponds to the label
        props = regionprops(single_labelled_image)
        for prop in props:
            if prop.label == label:
                centroid = prop.centroid
                print(centroid)

        # Check if a region in returned_crop corresponds to this centroid - 
        # Find the label in returned_crop that corresponds to the centroid
        label_at_centroid = returned_crop[int(centroid[0]), int(centroid[1]), int(centroid[2])]
        if label_at_centroid !=0:
            labels_to_keep.append(label_at_centroid)
            filtered_crop[returned_crop == label_at_centroid] = label



        print('keeping' + str((labels_to_keep)))
    return filtered_crop
        

def extract_cell_from_video(raw_vid , seg_vid , label , custom_model_path, padding):
    num_t_points   = raw_image.shape[0]
    seg_crop_dict  = {}
    bbox_dict      = {}
    centroids_dict = {}
    empty_vid = np.zeros_like(raw_vid)
    for i in range(num_t_points):
        print(i)
        raw_vid_frame = raw_vid[i,...]
        seg_vid_frame = seg_vid[i,...]
        crop_raw_image  , crop_label_image , bbox, centroid= focus_on_label(raw_vid_frame , seg_vid_frame, label, padding) 
        if crop_label_image is not None:
            mask = run_segmentation(crop_raw_image , crop_label_image,  custom_model_path)
            filtered_mask = extract_central_cell_from_mask(mask)
            seg_crop_dict[i] = filtered_mask
            centroids_dict[i] = centroid
            bbox_dict[i] = bbox
            

    # Determine the maximum dimensions of all masks
    max_z_range =  max(mask.shape[0] for mask in seg_crop_dict.values())
    max_y_range =  max(mask.shape[1] for mask in seg_crop_dict.values())
    max_x_range =  max(mask.shape[2] for mask in seg_crop_dict.values())

    # Create an empty array to hold the combined masks
    combined_seg = np.zeros((num_t_points,  max_z_range,  max_y_range,  max_x_range), dtype=np.uint8)
    combined_masks = np.zeros((num_t_points,  max_z_range,  max_y_range,  max_x_range), dtype=np.uint8)
    combined_raw   = np.zeros((num_t_points,  max_z_range,  max_y_range,  max_x_range), dtype=np.uint8)

    # Place each mask into the combined array, centered within the larger dimensions
    for i in range((num_t_points)):
        #filtered seg
        filtered_seg_frame = return_crop_to_image(empty_vid[i,...] , seg_crop_dict[i] , bbox_dict[i])
        raw_vid_frame = raw_vid[i,...]
        seg_vid_frame = seg_vid[i,...]
        # Apply the bounding box to the raw_data
        centroid = centroids_dict[i]
        x_centroid = int(centroid[2])
        y_centroid = int(centroid[1])
        z_centroid = int(centroid[0])

        # Calculate the region around the centroid
        half_x_range = max_x_range // 2
        half_y_range = max_y_range // 2
        half_z_range = max_z_range // 2

        z_start = max(0, z_centroid - half_z_range)
        z_end = z_start + max_z_range
        if z_end > raw_vid.shape[1]:
            z_end = raw_vid.shape[1]
            z_start = z_end - max_z_range

        y_start = max(0, y_centroid - half_y_range)
        y_end = y_start + max_y_range
        if y_end > raw_vid.shape[2]:
            y_end = raw_vid.shape[2]
            y_start = y_end - max_y_range

        x_start = max(0, x_centroid - half_x_range)
        x_end = x_start + max_x_range
        if x_end > raw_vid.shape[3]:
            x_end = raw_vid.shape[3]
            x_start = x_end - max_x_range

        combined_raw[i , ...]   = raw_vid_frame[ z_start:z_end,y_start:y_end, x_start:x_end]
        combined_masks[i , ...] = filtered_seg_frame[ z_start:z_end,y_start:y_end, x_start:x_end] #single_
        combined_seg[i,...]     = seg_vid_frame[ z_start:z_end,y_start:y_end, x_start:x_end]

    # Save the combined masks as a single image
    output_dir = r'/mnt/Work/Group Fritzsche/Ed/'
    tiff.imwrite(output_dir + 'combined_central_masks.tif', combined_masks)
    tiff.imwrite(output_dir + 'combined_seg.tif' , combined_seg)
    tiff.imwrite(output_dir + 'combined_raw.tif' , combined_raw)
    return



def two_cell_interface(raw_vid, seg_vid, label_list , custom_model_path, padding, output_dir):
    '''
    first i need to locate the 'interface' position between the cells. I could consider this to be the 
    midpoint between the centroids. If this distance is too large then error?

    Inputs
    -------------------------------------------------------------------------------
    raw_vid: 4D numpy array of shape (t, z, y, x) representing the raw video data.
    seg_vid: 4D numpy array of shape (t, z, y, x) representing the segmented video data.
    label_list: List of labels to focus on for the interface detection. (please be length 2)
    custom_model_path: Path to the custom Cellpose model for segmentation. - should end in cyto for the whole cytoplasm model
    padding: Padding to apply around the bounding box of the detected cells - i think best to keep this reasonbly high to reduce errors.
    output_dir: Directory where the output files will be saved.

    Outputs
    -------------------------------------------------------------------------------
    saves the following files to the output_dir:
    - combined_interface_masks.tif: 4D numpy array of the interface masks for each time point. i.e. the same as seg_masks but with the two cells of interest only
    - combined_seg.tif: 4D numpy array of the segmentation masks for each time point. That follow the 
    Needs to be rewritten!!!!
    '''



    num_t_points     = raw_image.shape[0]
    seg_crop_dict    = {}
    filter_crop_dict = {}
    bbox_dict        = {}
    midpoints_dict   = {}
    empty_vid = np.zeros_like(raw_vid)
    for i in range(num_t_points):
        print(i)
        raw_vid_frame = raw_vid[i,...]
        seg_vid_frame = seg_vid[i,...]
        crop_raw_image  , crop_label_image , bbox, centroid= focus_on_two_cells(raw_vid_frame , seg_vid_frame, label_list, padding) 
        if crop_label_image is not None:
            print(crop_raw_image.shape)
            print(crop_label_image.shape)
            mask                = run_segmentation(crop_raw_image , crop_label_image,  custom_model_path)
            mask_returned       = return_crop_to_image(empty_vid[i,...] , mask , bbox)
            filtered_mask       = cells_from_label_list(seg_vid[i,...] , mask_returned , label_list)
            seg_crop_dict[i]    = mask
            filter_crop_dict[i] = filtered_mask
            midpoints_dict[i] = centroid
            bbox_dict[i] = bbox
            

    # Determine the maximum dimensions of all masks
    max_z_range =  max(mask.shape[0] for mask in seg_crop_dict.values())
    max_y_range =  max(mask.shape[1] for mask in seg_crop_dict.values())
    max_x_range =  max(mask.shape[2] for mask in seg_crop_dict.values())

    # Create an empty array to hold the combined masks
    combined_seg   = np.zeros((num_t_points,  max_z_range,  max_y_range,  max_x_range), dtype=np.uint8)
    combined_interface_masks = np.zeros((num_t_points,  max_z_range,  max_y_range,  max_x_range), dtype=np.uint8)
    combined_raw   = np.zeros((num_t_points,  max_z_range,  max_y_range,  max_x_range), dtype=np.uint8)

    # Place each mask into the combined array, centered within the larger dimensions
    for i in range((num_t_points)):
        #filtered seg
        filtered_seg_frame =  filter_crop_dict[i]
        raw_vid_frame = raw_vid[i,...]
        seg_vid_frame = seg_vid[i,...]

        # Apply the bounding box to the raw_data
        midpoint = midpoints_dict[i]
        x_centroid = int(midpoint[2])
        y_centroid = int(midpoint[1])
        z_centroid = int(midpoint[0])

        # Calculate the region around the centroid
        half_x_range = max_x_range // 2
        half_y_range = max_y_range // 2
        half_z_range = max_z_range // 2

        z_start = max(0, z_centroid - half_z_range)
        z_end = z_start + max_z_range
        if z_end > raw_vid.shape[1]:
            z_end = raw_vid.shape[1]
            z_start = z_end - max_z_range

        y_start = max(0, y_centroid - half_y_range)
        y_end = y_start + max_y_range
        if y_end > raw_vid.shape[2]:
            y_end = raw_vid.shape[2]
            y_start = y_end - max_y_range

        x_start = max(0, x_centroid - half_x_range)
        x_end = x_start + max_x_range
        if x_end > raw_vid.shape[3]:
            x_end = raw_vid.shape[3]
            x_start = x_end - max_x_range

        combined_raw  [i , ...] = raw_vid_frame     [ z_start:z_end,y_start:y_end, x_start:x_end]
        combined_interface_masks[i , ...] = filtered_seg_frame[ z_start:z_end,y_start:y_end, x_start:x_end]
        combined_seg  [i , ...] = seg_vid_frame     [ z_start:z_end,y_start:y_end, x_start:x_end]

    # Save the combined masks as a single image
    tiff.imwrite(output_dir + 'combined_interface_masks.tif', combined_interface_masks)
    tiff.imwrite(output_dir + 'combined_seg.tif'          , combined_seg  )
    tiff.imwrite(output_dir + 'combined_raw.tif'          , combined_raw  )



def create_composite(raw_vid, label_vid):
    # Ensure the raw video and label video have the same dimensions
    assert raw_vid.shape == label_vid.shape, "Raw video and label video must have the same dimensions."

    # Stack the raw video and label video along a new axis to create a 2-channel video
    composite_video = np.stack((raw_vid, label_vid), axis=1)

    # Save the composite video as a TIFF file
    output_dir = r'/mnt/Work/Group Fritzsche/Ed/'
    tiff.imwrite(output_dir + 'composite_video.tif', composite_video)
    return





    
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

    two_cell_interface(raw_image , seg1_image, label_list , custom_model_path ,padding=30, output_dir = r'/mnt/Work/Group Fritzsche/Ed/')