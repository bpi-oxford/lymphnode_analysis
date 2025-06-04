import trackpy as tp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def Deff_from_singletrack(single_track_df, pixel_size=0.145, time_interval=12.76):
    """
    Calculate the effective diffusion coefficient (Deff) for a single track.
    
    Parameters:
    single_track_df (pd.DataFrame): DataFrame containing x, y, z coordinates and time for a single track.
    
    Returns:
    float: Effective diffusion coefficient (Deff).
    """
    MSD = tp.imsd(single_track_df, mpp=pixel_size, fps=1/time_interval)
    print(MSD)
    plt.plot(MSD.index, MSD.iloc[:, 0], label='MSD')
    plt.xlabel('Time (s)')
    plt.ylabel('MSD (um^2)')
    plt.title('Mean Squared Displacement (MSD)')
    plt.show()
    return 

def prepare_ultrack_df(ultrack_df):
    trackpy_df =ultrack_df[['track_id', 't','x', 'y', 'z']]
    trackpy_df.rename(columns={'track_id': 'particle' , 't' : 'frame'}, inplace=True)
    return trackpy_df

if __name__ == "__main__":
    # Load the data
    ultrack_df_path_crop1 = r"/home/edwheeler/Documents/code/ultrack_scripts/outputs/crop_1_tracks.csv"
    ultrack_df_path_crop2 = r"/home/edwheeler/Documents/code/ultrack_scripts/outputs/crop_2_tracks.csv"
    ultrack_df_crop1 = pd.read_csv(ultrack_df_path_crop1)
    ultrack_df_crop2 = pd.read_csv(ultrack_df_path_crop2)
    ultrack_df_crop1 = prepare_ultrack_df(ultrack_df_crop1)
    ultrack_df_crop2 = prepare_ultrack_df(ultrack_df_crop2)

    Deff_from_singletrack(ultrack_df_crop2[ultrack_df_crop2['particle'] == 203])

