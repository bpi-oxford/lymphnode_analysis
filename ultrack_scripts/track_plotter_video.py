'''
code to produce a video from a track and raw_data and labelled image
'''
import tifffile as tiff
import pandas as pd
import numpy  as np
from skimage.measure import regionprops
from skimage import measure
import matplotlib.pyplot as plt

def label_to_contour(single_labelled_image):
    contours = measure.find_contours(single_labelled_image, level=0.5)
    contour_image = np.zeros_like(single_labelled_image, dtype=np.uint8)

    for contour in contours:
        contour = np.round(contour).astype(int)
        for point in contour:
            y, x = point
            if 0 <= y < contour_image.shape[0] and 0 <= x < contour_image.shape[1]:
                contour_image[y, x] = 1

    return contour_image

if __name__ == "__main__":
    tracks_df = pd.read_csv(r'/home/edwheeler/Documents/code/ultrack_scripts/outputs/crop_2_tracks.csv')
    labelled_mask = tiff.imread(r"/home/edwheeler/Documents/code/ultrack_scripts/outputs/crop_2_relabelled_masks.tif")
    raw_data = tiff.imread(r"//home/edwheeler/Documents/cropped_region_2_motile/b2-2a_2c_pos6-01_deskew_cgt_cropped_for_segmentation_motile_region.tif")

    #pick a track
    track_id = 18
    track_data = tracks_df[tracks_df['track_id'] == track_id]

    x = track_data['x'].values
    y = track_data['y'].values
    z = track_data['z'].values
    t = track_data['t'].values
    print(t)

    #collect the labelled image
    single_labelled_mask = (labelled_mask == track_id).astype(int)
    bbox_dict = {}
    for i in range(len(t)):
        time = t[i]
        # Find the region properties for the specific track
        single_labelled_mask_frame = single_labelled_mask[time , ...]
        regions = regionprops(single_labelled_mask_frame)
        if regions:
            region = regions[0]  # Assuming there's only one region for the track_id
            centroid = region.centroid
            bbox = region.bbox
            print(f"Centroid: {centroid}, Bounding Box: {bbox}")
        else:
            print("No region found for the given track_id.")
        
        bbox_dict[time] = bbox

    # Find the maximum range of x and y in the bbox dictionary
    x_ranges = []
    y_ranges = []

    for time, box in bbox_dict.items():
        z_min, y_min, x_min, z_max, y_max, x_max = box
        x_ranges.append(x_max - x_min)
        y_ranges.append(y_max - y_min)

    max_x_range = max(x_ranges) + 20
    max_y_range = max(y_ranges) + 20

    print(f"Maximum X range: {max_x_range}")
    print(f"Maximum Y range: {max_y_range}")

    centroid_vid_raw   = np.zeros ( (len(t) , max_y_range , max_x_range))
    centroid_vid_label = np.zeros ( (len(t) , max_y_range , max_x_range))

    for i in range(len(t)):
        time = t[i]
        # Apply the bounding box to the raw_data
        z_min, y_min, x_min, z_max, y_max, x_max = bbox_dict[time]
        z_centroid = int((z_min + z_max) / 2.0)
        y_centroid = int((y_min + y_max) / 2.0)
        x_centroid = int((x_min + x_max) / 2.0)

        # Calculate the region around the centroid
        half_x_range = max_x_range // 2
        half_y_range = max_y_range // 2

        y_start = max(0, y_centroid - half_y_range)
        y_end = y_start + max_y_range
        if y_end > raw_data.shape[2]:
            y_end = raw_data.shape[2]
            y_start = y_end - max_y_range

        x_start = max(0, x_centroid - half_x_range)
        x_end = x_start + max_x_range
        if x_end > raw_data.shape[3]:
            x_end = raw_data.shape[3]
            x_start = x_end - max_x_range

        contour_frame = label_to_contour(single_labelled_mask[time, z_centroid,...])
        centroid_vid_raw[i , ...] = raw_data[time, z_centroid,y_start:y_end, x_start:x_end]
        centroid_vid_label[i , ...] = contour_frame[ y_start:y_end, x_start:x_end] #single_labelled_mask[time, z_centroid, y_start:y_end, x_start:x_end]
    
    centroid_vid = np.zeros((2, len(t) , max_y_range , max_x_range) )
    centroid_vid[0, ...] = centroid_vid_raw
    centroid_vid[1, ...] = centroid_vid_label
    output_path = r'/home/edwheeler/Documents/cropped_region_2_motile/figures/' + str(track_id) + '.tif'
    # Convert the data to uint8 to ensure compatibility with ImageJ
    centroid_vid_uint16 = (centroid_vid / centroid_vid.max() * 65535).astype(np.uint16)
    # Swap the axes to correct the channel and time dimensions
    centroid_vid_uint16 = np.transpose(centroid_vid_uint16, (1, 0, 2, 3))
    tiff.imwrite(output_path, centroid_vid_uint16, imagej=True)


