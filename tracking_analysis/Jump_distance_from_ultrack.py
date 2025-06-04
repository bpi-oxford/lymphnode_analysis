import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tifffile as tiff

def jump_distance_single_track(single_track_df , pixel_size=0.145, time_interval=12.76):
    """
    Calculate the jump distance for a single track.
    
    Parameters:
    single_track_df (pd.DataFrame): DataFrame containing x, y, z coordinates and time for a single track.
    
    Returns:
    pd.DataFrame: DataFrame containing time intervals and corresponding jump distances.
    """
    # Calculate displacements
    x = single_track_df['x'].values * pixel_size
    y = single_track_df['y'].values * pixel_size
    z = single_track_df['z'].values * pixel_size
    t = single_track_df['t'].values * time_interval

    jump_distance = np.zeros(len(t)-1)

    for i in range(1, len(t)):
        pos0 = np.array([x[i-1], y[i-1], z[i-1]])
        pos = np.array([x[i], y[i], z[i]])
        jump_distance[i-1] = np.sqrt(np.sum((pos - pos0) ** 2))
    
    return jump_distance

def jump_distance_all_tracks(track_df_path):
    """
    Calculate the jump distance for all tracks in the DataFrame.
    
    Parameters:
    track_df (pd.DataFrame): DataFrame containing x, y, z coordinates and time for multiple tracks.
    
    Returns: jump_distance dataframe. Note that the tracks are padded with zeros for frames when the label
    is not present - so need to remove these zeros before analysis
    """
    track_df = pd.read_csv(track_df_path)
    track_ids = track_df['track_id'].unique()
    longest_track_length = 0
    for track_id in track_ids:
        track_length = len(track_df[track_df['track_id'] == track_id])
        if track_length > longest_track_length:
            longest_track_length = track_length
    print("Longest track length: ", longest_track_length)


    jump_distance_matrix = np.zeros((longest_track_length , len(track_ids),))

    for i in range(len(track_ids)):
        track_id = track_ids[i]
        single_track_df = track_df[track_df['track_id'] == track_id]
        jump_distance_single_track_values = jump_distance_single_track(single_track_df)
        jump_distance_matrix[:len(jump_distance_single_track_values),i] = jump_distance_single_track_values

    return jump_distance_matrix
