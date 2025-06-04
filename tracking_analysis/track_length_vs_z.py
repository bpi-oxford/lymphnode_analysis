import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    # Load the dataframe of tracks
    track_path = r'/home/edwheeler/Documents/tissue_analysis_project/tracking_benchmarking/outputs/tracks/tracks_for_plotting/tracks_crop2_new.csv'
    track_df = pd.read_csv(track_path)
    output_directory = r'/mnt/Work/Group Fritzsche/Ed'

    track_ids = track_df['track_id'].unique()
    track_zs      = np.zeros((len(track_ids) , ))
    track_lengths = np.zeros((len(track_ids) , ))
    for i in range(len(track_ids)):
        track_id = track_ids[i]
        track_df_i = track_df[track_df['track_id'] == track_id]
        z = track_df_i['z'].values
        track_zs[i] = np.mean(z)
        track_lengths[i] = len(track_df_i)

    plt.scatter(track_lengths,track_zs, alpha=0.5)
    plt.xlabel('Track Length')
    plt.ylabel('Track Z')
    plt.show()

    output_path = f"{output_directory}/track_length_withz_plot.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')

    