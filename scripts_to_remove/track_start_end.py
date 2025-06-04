'''
Script that implements ultracks tracks_ends and tracks_starts functions
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ultrack.tracks.gap_closing import tracks_starts , tracks_ends

def end_positions_df(ultrack_df):
    end_df = tracks_ends(ultrack_df)
    return end_df

def start_positions_df(ultrack_df):
    start_df = tracks_starts(ultrack_df)
    return start_df

def tracks_within_central_volume(ultrack_df , xborder , yborder ,z_border):
    '''
    function to detect the tracks that do not enter the border region
    '''
    track_ids = np.unique(ultrack_df['track_id'])
    central_tracks = []
    for track_id in track_ids:
        track_data = ultrack_df[ultrack_df['track_id'] == track_id]
        if (track_data['x'].min() > xborder[0] and track_data['x'].max() < xborder[1] and
            track_data['y'].min() > yborder[0] and track_data['y'].max() < yborder[1] and
            track_data['z'].min() > z_border[0] and track_data['z'].max() < z_border[1]):
            central_tracks.append(track_id)
    
    central_tracks_df = ultrack_df[ultrack_df['track_id'].isin(central_tracks)]
    return central_tracks_df


if __name__ == "__main__":
    track_path = r'/home/edwheeler/Documents/tissue_analysis_project/tracking_benchmarking/outputs/tracks/tracks_for_plotting/tracks_crop2_new.csv'
    tracks_df = pd.read_csv(track_path)
    output_directory = r'/mnt/Work/Group Fritzsche/Ed'
    

    xborder = [0 , 512]
    yborder = [0 , 512]
    zborder = [0 , 182]

    central_tracks = tracks_within_central_volume(tracks_df ,
                                                    xborder,
                                                    yborder,
                                                    zborder)
    print(central_tracks)


    track_start_df = tracks_starts(tracks_df)
    track_end_df   = tracks_ends  (tracks_df)

    print(track_end_df)

    plt.hist(track_start_df['z'])
    plt.show()
    output_path = f"{output_directory}/track_length_withz_plot.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')