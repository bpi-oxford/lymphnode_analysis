import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from Velocity_from_ultrack import avg_velocity_from_track
import numpy as np

def plot_tracks_from_zero_3d(track_df):
    """
    Plots tracks from a pandas DataFrame in 3D, aligning all tracks to start at (0, 0, 0).
    
    Parameters:
        track_df (pd.DataFrame): DataFrame containing track data with columns 'track_id', 'x', 'y', and 'z'.
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    track_ids = track_df['track_id'].unique()
    
    for i in range(len(track_ids)):
        track_id = track_ids[i]
        track_data = track_df[track_df['track_id'] == track_id]

        # Shift each track to start at (0, 0, 0)
        x_shifted = track_data['x'] - track_data['x'].iloc[0]
        y_shifted = track_data['y'] - track_data['y'].iloc[0]
        z_shifted = track_data['z'] - track_data['z'].iloc[0]

        #find track veolcity
        avg_vel = np.round(avg_velocity_from_track(track_data) * 60,1)
        print(f"Track ID: {track_id}, Average Velocity: {avg_vel:.2f} um/min")
        # Plot the shifted track
        ax.plot(x_shifted, y_shifted, z_shifted, label=f'v = {avg_vel} um/min ' + f'id = {track_id}', alpha=0.7)
    
    ax.set_xlabel('X (shifted)')
    ax.set_ylabel('Y (shifted)')
    ax.set_zlabel('Z (shifted)')
    ax.set_title('Tracks Starting at Zero (3D)')
    ax.legend()
    plt.show()

def plot_tracks_from_zero_2d(track_df):
    """
    Plots tracks from a pandas DataFrame in 2D, aligning all tracks to start at (0, 0).
    
    Parameters:
        track_df (pd.DataFrame): DataFrame containing track data with columns 'track_id', 'x', and 'y'.
    """
    plt.figure(figsize=(10, 8))
    track_ids = track_df['track_id'].unique()
    
    for i in range(len(track_ids)):
        track_id = track_ids[i]
        track_data = track_df[track_df['track_id'] == track_id]

        # Shift each track to start at (0, 0)
        x_shifted = track_data['x'] - track_data['x'].iloc[0]
        y_shifted = track_data['y'] - track_data['y'].iloc[0]

        #find track velocity
        avg_vel = np.round(avg_velocity_from_track(track_data) * 60, 1)
        print(f"Track ID: {track_id}, Average Velocity: {avg_vel:.2f} um/min")
        # Plot the shifted track with bolder lines
        plt.plot(x_shifted, y_shifted, alpha=0.7, linewidth=3)
    
    plt.axis('off')
    plt.show()


# Example usage
if __name__ == "__main__":
    # Example DataFrame
    trackdf_path = r"/Users/edwheeler/Downloads/tracks_crop1_15.csv"
    track_df = pd.read_csv(trackdf_path)
    track_df = track_df[['track_id', 'x', 'y', 'z' , 't']]
    track_df = track_df[track_df['track_id'] <10]
    track_df = track_df[track_df['track_id'] > 0]
    
    plot_tracks_from_zero_2d(track_df)