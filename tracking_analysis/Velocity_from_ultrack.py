import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tifffile as tiff
import os
import seaborn as sns

def avg_velocity_from_track(single_track_df, pixel_size=0.145, time_interval=12.76):
    # Sort the dataframe by time values
    single_track_df = single_track_df.sort_values(by='t').reset_index(drop=True)
    #print(single_track_df)

    x = single_track_df['x'].values * pixel_size
    y = single_track_df['y'].values * pixel_size
    z = single_track_df['z'].values * pixel_size
    t = single_track_df['t'].values * time_interval


    # Calculate the average velocity
    velocity = np.zeros(len(t))
    for i in range(1,len(t)):
        pos0 = np.array([x[i-1], y[i-1], z[i-1]])
        pos = np.array([x[i], y[i], z[i]])
        time_diff = t[i] - t[i-1]
        velocity[i] = np.sqrt(np.sum((pos - pos0) ** 2)) / time_diff #np.sqrt( (pos[0] - pos0[0]) **2 + (pos[1] - pos0[1]) **2 + (pos[2] - pos0[2]) **2 )/ time_diff  
        #print(velocity[i])
    avg_velocity = np.mean(velocity[1:])
    return avg_velocity

def avg_velocity_all_tracks(track_df):
    """
    Calculate the average velocity for all tracks in the DataFrame.
    
    Parameters:
    track_df (pd.DataFrame): DataFrame containing x, y, z coordinates and time for multiple tracks.
    
    Returns: avg_velocity dataframe
    """

    track_ids = track_df['track_id'].unique()
    avg_velocity_ensemble = np.zeros(len(track_ids))
    lengths = np.zeros(len(track_ids))

    for i in range(len(track_ids)):
        track_id = track_ids[i]
        single_track_df = track_df[track_df['track_id'] == track_id]
        avg_velocity_ensemble[i] = avg_velocity_from_track(single_track_df)
        lengths[i] = len(single_track_df)

    return avg_velocity_ensemble , lengths

def prepare_trackmate_to_ultrackdf(trackmate_df):
    trackmate_df = trackmate_df[['TRACK_ID' ,'POSITION_X', 'POSITION_Y', 'POSITION_Z', 'FRAME']]
    trackmate_df.rename(columns={
        'TRACK_ID': 'track_id',
        'POSITION_X': 'x',
        'POSITION_Y': 'y',
        'POSITION_Z': 'z',
        'FRAME': 't'
    }, inplace=True)
    trackmate_df = trackmate_df.iloc[3:].reset_index(drop=True)
    trackmate_df = trackmate_df.astype({'x': 'float', 'y': 'float', 'z': 'float', 't': 'float'})
    return trackmate_df

def track_length_vs_velocity(track_df):
    v , l = avg_velocity_all_tracks(track_df)
    plt.scatter(l , v * 60)
    plt.xlabel('l')
    plt.ylabel('v')
    plt.show()
    return

def dfs_to_velocity_df(track_df_list , df_labels):
    data = pd.DataFrame(columns=df_labels)
    v_list = []
    
    for track_df_path in track_df_list:
        df = pd.read_csv(track_df_path)
        v, l = avg_velocity_all_tracks(df)
        v_list.append(v * 60)

    max_v_length = max(len(v) for v in v_list)
        

    for i in range(len(v_list)):
        x_label = df_labels[i]
        padded_v = np.pad(v_list[i], (0, max_v_length - len(v_list[i])), 'constant', constant_values=np.nan)
        data[x_label] = padded_v
    return data

def plot_superplot_velocity(track_df_list , df_labels , output_path):
    plt.figure()
    data = dfs_to_velocity_df(track_df_list , df_labels)

    sns.swarmplot(data=data, orient='v', size=2, alpha=1, zorder=1)
    sns.boxplot(data=data, orient='v', width=0.5, showfliers=False, notch=True, showmeans=True,
                meanline=True,
                meanprops={'color': 'k', 'ls': '-', 'lw': 2},
                whiskerprops={'visible': False},
                zorder=10,
                showbox=False,
                showcaps=False,
                medianprops={'visible': False}
                )
    #sns.pointplot(data=data, orient='v', join=False, ci='sd', color='black', markers='_', scale=0.5, errwidth=1)

    plt.ylabel('Velocity (µm/min)', fontsize=12, fontweight='bold')
    plt.xlabel('Conditions', fontsize=12, fontweight='bold')
    plt.xticks(fontsize=10, fontweight='bold')
    plt.yticks(fontsize=10, fontweight='bold')
    plt.gca().spines['top'].set_linewidth(1.5)
    plt.gca().spines['top'].set_color('black')
    plt.gca().spines['right'].set_linewidth(1.5)
    plt.gca().spines['right'].set_color('black')
    plt.gca().spines['left'].set_linewidth(1.5)
    plt.gca().spines['left'].set_color('black')
    plt.gca().spines['bottom'].set_linewidth(1.5)
    plt.gca().spines['bottom'].set_color('black')
    
    plt.ylabel('Velocity (µm/min)')
    plt.ylim(bottom=0)
    plt.ylim([0 , 15])
    plt.xlabel('region', labelpad=5)
    plt.show()
    plt.savefig(output_path + f"superplot_velocity.png", dpi=300, bbox_inches='tight')




    return


def plot_velocity_hist(track_df_list , df_labels , output_path):
    data = dfs_to_velocity_df(track_df_list , df_labels)
    plt.figure()
    for column in data.columns:
        sns.histplot(data[column], kde=False, label=column, bins=50, alpha=1, stat="density", edgecolor=None)

    plt.xlabel('Velocity (µm/min)', fontsize=12, fontweight='bold')
    plt.ylabel('Frequency', fontsize=12, fontweight='bold')
    plt.legend(title='Conditions', fontsize=10, title_fontsize=12)
    plt.xticks(fontsize=10, fontweight='bold')
    plt.yticks(fontsize=10, fontweight='bold')
    plt.gca().spines['top'].set_linewidth(1.5)
    plt.gca().spines['top'].set_color('black')
    plt.gca().spines['right'].set_linewidth(1.5)
    plt.gca().spines['right'].set_color('black')
    plt.gca().spines['left'].set_linewidth(1.5)
    plt.gca().spines['left'].set_color('black')
    plt.gca().spines['bottom'].set_linewidth(1.5)
    plt.gca().spines['bottom'].set_color('black')
    plt.xlim([0, 15])
    plt.show()
    plt.savefig(output_path + "velocity_histogram.png", dpi=300, bbox_inches='tight')


if __name__ == "__main__":
    track_df_folder = r'/home/edwheeler/Documents/tissue_analysis_project/tracking_benchmarking/outputs/tracks/tracks_for_plotting_crop1'
    output_path = r'/mnt/Work/Group Fritzsche/Ed/'
    # Load all .csv files in the folder
    csv_files = [os.path.join(track_df_folder, file) for file in os.listdir(track_df_folder) if file.endswith('.csv')]
    labels = ['']
    print(csv_files)
    plot_velocity_hist(csv_files , labels , output_path)


   