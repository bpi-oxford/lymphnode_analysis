'''
run scripts from the benchmarking module

I want to test effect of closing and max_neighbours on lengths and end/start positions
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tracking_benchmarking.track_closing import close_gaps
from tracking_benchmarking.track_start_end import start_positions_df , end_positions_df, tracks_within_central_volume
from tracking_benchmarking.track_length_finder import length_finder

def volume_over_time(segmentation , label):
    label_segmentation = (segmentation == label).astype(int)
    volume = np.sum(label_segmentation > 0)
    return volume


if __name__ == "__main__":
    tracks_path = r'/home/edwheeler/Documents/tissue_analysis_project/tracking_benchmarking/outputs/tracks_crop1_15.csv'
    tracks_df = pd.read_csv(tracks_path)
    start_positions_df(tracks_df)

    xy_edge = 0
    z_edge_bottom  = 10
    z_edge_top     = 100

    xborder = [xy_edge , 512-xy_edge]
    yborder = [xy_edge , 512-xy_edge]
    zborder = [z_edge_bottom , z_edge_top]

    max_gap = 1
    max_radius = 0

    closed_tracks_df = close_gaps(tracks_df ,max_gap=max_gap, max_radius=max_radius)

    central_tracks = tracks_within_central_volume(closed_tracks_df ,
                                                  xborder,
                                                  yborder,
                                                  zborder)
    
    central_track_lengths = length_finder(central_tracks)

    plt.hist(central_track_lengths)
    plt.show()

