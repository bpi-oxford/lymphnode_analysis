''''
Testing ultracks close_tracks_gaps function
'''
import pandas as pd
import matplotlib.pyplot as plt
from ultrack.tracks import close_tracks_gaps


def close_gaps(df , max_gap , max_radius ):
    return close_tracks_gaps(tracks_df=df, max_gap = max_gap, max_radius=max_radius)

if __name__ == "__main__":
    tracks_path = r'/home/edwheeler/Documents/code/ultrack_scripts/outputs/crop_2_tracks.csv'
    tracks = pd.read_csv(tracks_path)
