import trackpy as tp
import pandas as pd
import numpy as np
import tifffile as tiff
import os
import matplotlib.pyplot as plt

'''
functions to calculate and plot the mean square displacement (MSD)
and its ensemble version (eMSD) using trackpy functionality
'''

def MSD_to_ensembleMSD(MSD):
    '''
    Input: MSD - output from trackpy: a dataframe of MSD vs displacement for each track

    Output: Combines the tracks through calculating their mean at each lag time. Finds the 
    standard mean of the error also.
    '''
    shape = MSD.shape
    eMSD = np.zeros((shape[0], ))
    eMSD_errors = np.zeros((shape[0], ))

    for i in range(shape[0]):
        SD_at_time_i = MSD.iloc[i, :]
        SD_at_time_i_nonzeros = SD_at_time_i[SD_at_time_i != 0]
        eMSD[i] = np.mean(SD_at_time_i_nonzeros)
        stddev = np.std(SD_at_time_i_nonzeros)
        eMSD_errors[i] = stddev / np.sqrt(len(SD_at_time_i_nonzeros))
    return eMSD , eMSD_errors

def find_eMSD(ultrack_df_path, mpp=0.145 , fps=1/12.6):
    '''
    -Input: path to ultrack_df
    mpp: microns per pixel -defaults to 0.145
    fps: frames per second - defaults to 1/12.6

    -Output: ensemble MSD and the error in a 1D array form
    '''
    ultrack_df = pd.read_csv(ultrack_df_path)
    trackpy_df =ultrack_df[['track_id', 't','x', 'y', 'z']]
    print(trackpy_df.head())
    trackpy_df.rename(columns={'track_id': 'particle' , 't' : 'frame'}, inplace=True)
    MSD3d  = tp.imsd(trackpy_df, mpp, fps)
    print(MSD3d.head())
    eMSD , eMSD_errors = MSD_to_ensembleMSD(MSD3d)
    return eMSD , eMSD_errors

def plot_eMSD(ultrack_df_path_list , label_list, output_directory, mpp=0.145 , fps=1/12.6):
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
        eMSD, eMSD_errors = find_eMSD(path , mpp , fps)
        frames = np.arange(len(eMSD))
        mins = frames * 1/60 * 1/fps
        # Plot the eMSD
        label = label_list[ultrack_df_path_list.index(path)]
        plt.plot(mins, eMSD, label=label)
        plt.fill_between(mins, eMSD - eMSD_errors, eMSD + eMSD_errors, color='gray', alpha=0.5)


    plt.ylabel(r'eMSD ($\mu m^2$)', fontsize=14, fontweight='bold')
    plt.xlabel('Time (mins)', fontsize=14, fontweight='bold')
    plt.xticks(fontsize=12, fontweight='bold')
    plt.yticks(fontsize=12, fontweight='bold')
    plt.title('Ensemble Mean Squared Displacement (eMSD)', fontsize=16, fontweight='bold')
    plt.legend()
    plt.show()

    # Save the plot to the output directory
    output_path = f"{output_directory}/eMSD_plot.png"
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

    plot_eMSD(ultrack_df_path_list, labels, output_directory)
