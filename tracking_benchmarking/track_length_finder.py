''''
simple function to return the lengths of the tracks in a ultrack df
'''
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

def length_finder(track_df):
    unique_tracks = track_df['track_id'].unique()
    length_arr = np.zeros(len(unique_tracks))
    for i in range(len(unique_tracks)):
        track_id = unique_tracks[i]
        track_df_i = track_df[track_df['track_id'] == track_id]
        length_arr[i] = len(track_df_i)
    return length_arr

def plot_track_lengths(ultrack_df_path_list , label_list, output_directory, mpp=0.145 , fps=1/12.6):
    '''
    -Inputs: 
    ultrack_df_path_list: list of ultrack dataframes which will be separately plotted
    label_list          : list of the DFs labels for the legend. Must be same length as ultrack_df_path_list
    output_directory    : where the plot will be saved to
    mpp , fps           : imaging parameters
    '''

    assert len(ultrack_df_path_list) == len(label_list), "ultrack_df_path_list and label_list must be the same length"

    plt.figure(figsize=(10, 6))
    for path in ultrack_df_path_list:
        tracks = pd.read_csv(path)
        lengths = length_finder(tracks)
        label = label_list[ultrack_df_path_list.index(path)]
        plt.hist(lengths, label=label, density=True, alpha=0.7 , bins = 25)


    plt.ylabel('Frequency Density', fontsize=14, fontweight='bold')
    plt.xlabel('Track Lengths', fontsize=14, fontweight='bold')
    plt.xticks(fontsize=12, fontweight='bold')
    plt.yticks(fontsize=12, fontweight='bold')
    plt.title('Histogram of Track Lengths', fontsize=16, fontweight='bold')
    plt.legend()
    plt.show()

    # Save the plot to the output directory
    output_path = f"{output_directory}/track_length_plot.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_path}")
    return


    
    

if __name__ == "__main__":
    # Load the data
    ultrack_track_directory = r'/home/edwheeler/Documents/tissue_analysis_project/tracking_benchmarking/outputs/tracks/tracks_for_plotting/'
    # Get the list of all .csv files in the directory
    ultrack_df_path_list = [
        os.path.join(ultrack_track_directory, f)
        for f in os.listdir(ultrack_track_directory)
        if f.endswith('.csv')
    ]
    labels = ['old' , 'new']
    # Call the plot_eMSD function with the list of files
    output_directory = r'/mnt/Work/Group Fritzsche/Ed'

    plot_track_lengths(ultrack_df_path_list, labels, output_directory)