import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from helper_function_relabel_segmentation_masks import relabel_segmentation_masks
from helper_function_relabel_segmentation_masks_MPI import relabel_segmentation_masks_MPI
import tifffile as tiff

def length_finder(track_df):
    unique_tracks = track_df['track_id'].unique()
    length_arr = np.zeros(len(unique_tracks))
    for i in range(len(unique_tracks)):
        track_id = unique_tracks[i]
        track_df_i = track_df[track_df['track_id'] == track_id]
        length_arr[i] = len(track_df_i)
    return length_arr

if __name__ == "__main__":
    tracks_df  = pd.read_csv(r'/home/edwheeler/Documents/code/ultrack_scripts/outputs/crop_2_tracks.csv')
    #masks = tiff.imread(r"/home/edwheeler/Documents/cropped_region_2_motile/b2-2a_2c_pos6-01_deskew_cgt_cropped_for_segmentation_motile_region_seg_masks_minsize_10_filtered_stack.tif")
    #masks = relabel_segmentation_masks_MPI(masks , tracks_df, 200000)
    #tiff.imwrite('relabelled_masks.tif', masks)
    length_arr = length_finder(tracks_df)
    plt.hist(length_arr, bins=40)
    plt.xlabel('Length of track (frames)', fontweight='bold')
    plt.ylabel('Number of tracks', fontweight='bold')
    plt.title('Length of tracks', fontweight='bold')
    plt.xticks(fontweight='bold')
    plt.yticks(fontweight='bold')
    plt.show()
