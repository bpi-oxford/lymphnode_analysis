import numpy as np
import pandas as pd
import os
import glob
import matplotlib.pyplot as plt

def num_tracks(tracks_df):
    """
    Count the number of unique tracks in the DataFrame.
    """
    return len(tracks_df['track_id'].unique())

if __name__ == "__main__":
    input_dir = r'/home/edwheeler/Documents/tissue_analysis_project/tracking_benchmarking/outputs/'

    file_name_stem = 'outputstracks_crop2_maxN'
    # Load all files in the directory with the specified stem
    file_pattern = os.path.join(input_dir, f"{file_name_stem}*")
    files = glob.glob(file_pattern)

    # Print loaded files for verification
    print(f"Loaded files: {files}")
    # Initialize an empty list to store the data
    data = []

    # Iterate through files and collect data
    for file in files:
        df = pd.read_csv(file)
        # Extract the max_neighbors and max_distance from the filename
        file_name = os.path.basename(file)
        parts = file_name.split('_')
        max_neighbors = int(parts[2][4:])  # Extracting the number after 'maxN'
        max_distance = int(parts[3][4:-4])  # Extracting the number after 'maxD'
        num_tracks_count = num_tracks(df)
        # Append the data as a tuple
        data.append((max_neighbors, max_distance, num_tracks_count))

    # Create a DataFrame from the collected data
    results_df = pd.DataFrame(data, columns=['max neighbours', 'max distance', 'number of tracks'])

    # Print the DataFrame for verification
    print(results_df)
    # Sort the DataFrame by 'max distance'
    results_df = results_df.sort_values(by='max distance')
    # Plot max_neighbours against number of tracks for all unique max_distance values
    unique_max_distances = results_df['max distance'].unique()

    for max_distance_to_plot in unique_max_distances:
        filtered_df = results_df[results_df['max distance'] == max_distance_to_plot]

        plt.figure(figsize=(8, 6))
        plt.scatter(filtered_df['max neighbours'], filtered_df['number of tracks'], marker='o')
        plt.title(f'Number of Tracks vs Max Neighbours (Max Distance = {max_distance_to_plot})')
        plt.xlabel('Max Neighbours')
        plt.ylabel('Number of Tracks')
        plt.grid(True)
        plt.show()

    # Sort the DataFrame by 'max distance'
    results_df = results_df.sort_values(by='max neighbours')

    # Plot max_neighbours against number of tracks for all unique max_distance values
    unique_max_neighbours = results_df['max neighbours'].unique()

    for max_neighbours_to_plot in unique_max_neighbours:
        filtered_df = results_df[results_df['max neighbours'] == max_neighbours_to_plot]

        plt.figure(figsize=(8, 6))
        plt.scatter(filtered_df['max distance'], filtered_df['number of tracks'], marker='o')
        plt.title(f'Number of Tracks vs Max Distance (Max neighbours = {max_neighbours_to_plot})')
        plt.xlabel('Max distance')
        plt.ylabel('Number of Tracks')
        plt.grid(True)
        plt.show()

        # Save the plot as an image file
        output_plot_path = os.path.join(input_dir, f'plot_maxD_{max_neighbours_to_plot}.png')
        plt.savefig(output_plot_path)
        print(f"Plot saved to {output_plot_path}")


