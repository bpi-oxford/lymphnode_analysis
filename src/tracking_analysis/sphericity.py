import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tifffile as tiff
from Velocity_from_ultrack import avg_velocity_from_track
from scipy.ndimage import label, find_objects
from skimage.measure import regionprops
from skimage.measure import marching_cubes, mesh_surface_area

def sphericity_from_single_label(labelled_mask, track_df, voxel_size=0.145**3):
    """
    Calculate the sphericity of a label.
    
    Parameters:
    label (np.ndarray): 3D binary array representing the label.
    
    Returns:
    float: Sphericity of the label.
    """

    T = track_df['t'].values
    Z = labelled_mask.shape[1]
    Y = labelled_mask.shape[2]
    X = labelled_mask.shape[3]
    print(T, Z, Y, X)
    sphericity = np.zeros(len(T))
    velocity = np.zeros(len(T))
    for i in range(len(T)):
        time = T[i]
        frame = labelled_mask[time, :, :, :]
        vol = np.sum(frame) * voxel_size
        if vol > 0:
            #use regions to find the cell
            regions = regionprops(frame)
            #raise error if more than one region is found
            if len(regions) > 1:
                raise ValueError("More than one region found in the labelled frame.")
            
            bbox = regions[0].bbox
            #boolean array for the region
            region_mask = np.zeros(frame.shape, dtype=bool)
            region_mask[bbox[0]:bbox[3], bbox[1]:bbox[4], bbox[2]:bbox[5]] = True

            # Use marching cubes to find the surface area
            verts, faces, _, _ = marching_cubes(frame, level=0, mask=region_mask)
            SA = mesh_surface_area(verts, faces) * voxel_size**(2/3)
            sphericity[i] = (np.pi **(1.0/3.0)) * ((6 * vol) ** (2.0/3.0)) * 1 / (SA)
            print("Sphericity: ", sphericity[i])
        if i > 0:
            #calculate the velocity
            pos0 = np.array([track_df['x'].values[i-1], track_df['y'].values[i-1], track_df['z'].values[i-1]])
            pos = np.array([track_df['x'].values[i], track_df['y'].values[i], track_df['z'].values[i]])
            time_diff = T[i] - T[i-1]
            velocity[i] = np.sqrt(np.sum((pos - pos0) ** 2)) / time_diff 

    return sphericity , velocity

def sphericity_from_all_labels(labelled_mask_path, track_df_path, voxel_size=0.145**3):
    """
    Calculate the sphericity of all labels in the labelled mask.
    
    Parameters:
    labelled_mask (np.ndarray): 4D array representing the labelled masks.
    
    Returns:
    pd.DataFrame: DataFrame containing sphericity values for each label.
    """
    track_df = pd.read_csv(track_df_path)
    #track_df = track_df[track_df['track_id'] < 10]
    labelled_mask = tiff.imread(labelled_mask_path)
    track_ids = track_df['track_id'].unique()
    sphericity = np.zeros((len(track_ids), len(labelled_mask)))
    velocity = np.zeros((len(track_ids), len(labelled_mask)))
    #first find the longest track:
    longest_track_length = 0
    for track_id in track_ids:
        track_length = len(track_df[track_df['track_id'] == track_id])
        if track_length > longest_track_length:
            longest_track_length = track_length   
    print("Longest track length: ", longest_track_length)

    for i in range(len(track_ids)):
        track_id = track_ids[i]
        single_track_df = track_df[track_df['track_id'] == track_id]
        track_length = len(single_track_df)
        single_labelled_mask = (labelled_mask ==track_id).astype(int)
        sphericity[i, 0:track_length] ,  velocity[i, 0:track_length]  = sphericity_from_single_label(single_labelled_mask, single_track_df)

    return sphericity , velocity

def volume_over_time(segmentation ,track_df, label):
    """
    Calculate the volume of a label over time.
    
    Parameters:
    segmentation (np.ndarray): 4D array representing the labelled masks.
    label (int): The label to calculate the volume for.
    
    Returns:
    np.ndarray: Volume of the label over time.
    """
    single_track_df = track_df[track_df['track_id'] == label]
    print(single_track_df)
    volume_arr = np.zeros(len(single_track_df))
    time_arr   = np.zeros(len(single_track_df))
    for i in range(len(single_track_df)):
        t=  single_track_df['t'].values[i].astype(int)
        seg_frame = segmentation[t, :, :, :]
        label_segmentation = (seg_frame == label).astype(int)
        volume_arr[i] = np.sum(label_segmentation > 0 , axis=(0,1,2))
        time_arr[i] = t
    return volume_arr , time_arr

if __name__ == "__main__":

    track_df_path = r'/home/edwheeler/Documents/tissue_analysis_project/tracking_benchmarking/outputs/tracks_crop1_15.csv'
    labelled_mask_path = r'/home/edwheeler/Documents/tissue_analysis_project/tracking_benchmarking/outputs/relabelled_crop1_ultrack15_concatenated.tif'

    labelled_mask = tiff.imread(labelled_mask_path)
    track_df = pd.read_csv(track_df_path)

    track_id = 70
    single_track_df = track_df[track_df['track_id'] == track_id]
    volume , time = volume_over_time(labelled_mask ,track_df, track_id)
    print("Volume: ", volume)
    plt.plot(time , volume)
    plt.title("Volume over time")
    plt.xlabel("Time")
    plt.ylabel("Volume")
    plt.show()
    #sphericity , velocity = sphericity_from_all_labels(labelled_mask_path, tracd_df_path)