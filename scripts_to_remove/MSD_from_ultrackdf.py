import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def calculate_SD(single_track_df):
    """
    Calculate the Mean Squared Displacement (MSD) for a single track.
    
    Parameters:
    single_track_df (pd.DataFrame): DataFrame containing x, y, z coordinates and time for a single track.
    
    Returns:
    pd.DataFrame: DataFrame containing time intervals and corresponding MSD values.
    """
    # Calculate displacements
    x = single_track_df['x'].values
    y = single_track_df['y'].values
    z = single_track_df['z'].values
    t = single_track_df['t'].values

    SD = np.zeros(len(t))
    pos0 = np.array([x[0], y[0], z[0]])

    for i in range(len(t)):
        pos = np.array([x[i], y[i], z[i]])
        sq_disp = (np.sum((pos - pos0) ** 2))
        SD[i] = sq_disp
    
    single_track_df['SD'] = SD
    return single_track_df

def calculate_SD_all_tracks(track_df):
    """
    Calculate the  Squared Displacement (SD) for all tracks in the DataFrame.
    
    Parameters:
    track_df (pd.DataFrame): DataFrame containing x, y, z coordinates and time for multiple tracks.
    
    Returns: MSD dataframe
    """

    track_ids = track_df['track_id'].unique()
    #first find the longest track:
    longest_track_length = 0
    for track_id in track_ids:
        track_length = len(track_df[track_df['track_id'] == track_id])
        if track_length > longest_track_length:
            longest_track_length = track_length
    print("Longest track length: ", longest_track_length)

    SD = np.zeros( (longest_track_length, len(track_ids)))

    for i in range(len(track_ids)):
        track_id = track_ids[i]
        single_track_df = track_df[track_df['track_id'] == track_id]
        SD_single_track = calculate_SD(single_track_df)['SD'].values
        SD[0:len(SD_single_track), i] = SD_single_track

    return SD

def SD_to_MSD(SD):
    shape = SD.shape
    MSD = np.zeros((shape[0], ))
    MSD_errors = np.zeros((shape[0], ))

    for i in range(shape[1]):
        plt.plot(SD[:, i], label=f'Track {i+1}')
    plt.show()
    for i in range(shape[0]):
        SD_at_time_i = SD[i, :]
        SD_at_time_i_nonzeros = SD_at_time_i[SD_at_time_i != 0]
        MSD[i] = np.mean(SD_at_time_i_nonzeros)
        stddev = np.std(SD_at_time_i_nonzeros)
        MSD_errors[i] = stddev / np.sqrt(len(SD_at_time_i_nonzeros))

    plt.plot(MSD, label='MSD')
    plt.fill_between(range(len(MSD)), MSD - MSD_errors, MSD + MSD_errors, color='gray', alpha=0.5, label='Error')
    plt.legend()
    plt.show()




if __name__ == "__main__":
    ultrack_df_path = r"/home/edwheeler/Documents/code/ultrack_scripts/cropped_1_opening_frame_tracks.csv"
    track_df = pd.read_csv(ultrack_df_path)
    SD = calculate_SD_all_tracks(track_df)
    MSD = SD_to_MSD(SD)
