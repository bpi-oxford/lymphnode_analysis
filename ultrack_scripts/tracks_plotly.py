import plotly
import plotly.graph_objects as go
import pandas as pd
import tifffile as tiff
import numpy as np
import matplotlib.pyplot as plt

def disp_distances(track_data):
    """
    Function to calculate and display distances between consecutive points in a track.
    """
    for i in range(len(track_data) - 1):
        x = track_data['x'].iloc[i + 1]
        y = track_data['y'].iloc[i + 1]
        z = track_data['z'].iloc[i + 1]
        x0 = track_data['x'].iloc[i]
        y0 = track_data['y'].iloc[i]
        z0 = track_data['z'].iloc[i]
        dist = np.sqrt((x - x0) ** 2 + (y - y0) ** 2 + (z - z0) ** 2)
        print(f"Distance between point {i+1} and point {i}: {dist}")
    return track_data

def length_finder(track_df):
    unique_tracks = track_df['track_id'].unique()
    length_arr = np.zeros(len(unique_tracks))
    for i in range(len(unique_tracks)):
        track_id = unique_tracks[i]
        track_df_i = track_df[track_df['track_id'] == track_id]
        length_arr[i] = len(track_df_i)
    return length_arr

def tracks_opening_frame(tracks_df):
    unique_tracks = tracks_df['track_id'].unique()
    opening_frame_track_ids = []
    for i in range(len(unique_tracks)):
        track_id = unique_tracks[i]
        track_df_i = tracks_df[tracks_df['track_id'] == track_id]
        if track_df_i['t'].iloc[0] == 0:
            opening_frame_track_ids.append(track_id)
    return opening_frame_track_ids

def masks_opening_frame(labelled_mask, opening_frame_track_ids):
    for track_id in np.unique(labelled_mask):
        if track_id not in opening_frame_track_ids:
            labelled_mask[labelled_mask == track_id] = 0
    return labelled_mask

def track_crop(labelled_mask , tr):
    return


if __name__ == "__main__":
    # Define data for the 3D plot
    tracks_df = pd.read_csv(r'/home/edwheeler/Documents/tissue_analysis_project/tracking_benchmarking/outputs/tracks_crop1_15.csv')
    manual_track_data = pd.read_csv(r'/home/edwheeler/Documents/cropped_region_1/manual_tracks_crop1.csv')
    print(manual_track_data)
    #labelled_mask = tiff.imread(r"/home/edwheeler/Documents/code/ultrack_scripts/outputs/crop_2_relabelled_masks.tif")
    raw_data = tiff.imread(r"//home/edwheeler/Documents/cropped_region_2_motile/b2-2a_2c_pos6-01_deskew_cgt_cropped_for_segmentation_motile_region.tif")
    track_id = 78
    track_data = tracks_df[tracks_df['track_id'] == track_id]
    manual_track_id = 35
    manual_track_data = manual_track_data[manual_track_data['track id'] == manual_track_id]
    x = track_data['x']
    y = track_data['y']
    z = track_data['z']
    x1 = manual_track_data['x']
    y1 = manual_track_data['y']
    z1 = manual_track_data['z']
    print(len(x))

    print(manual_track_data)

    #disp_distances(track_data)

    '''
    opening_frame_track_ids = tracks_opening_frame(tracks_df)
    opening_frame_tracks_df = tracks_df[tracks_df['track_id'].isin(opening_frame_track_ids)]
    opening_frame_tracks_df.to_csv('opening_frame_tracks.csv', index=False)

    labelled_mask = masks_opening_frame(labelled_mask, opening_frame_track_ids)
    tiff.imwrite('opening_frame_labelled_mask.tif', labelled_mask)
    '''

    # Create a 3D scatter plot with lines connecting the points, colored by length
    fig = go.Figure()

    # Add the first track (track_data)
    fig.add_trace(go.Scatter3d(
        x=x, y=y, z=z, mode='lines+markers',
        marker=dict(size=5), #, color=track_data['t'], colorscale=[[0, 'red'], [1, 'blue']], opacity=0.8, colorbar=dict(title="Time (frames)")),
        #line=dict(color=track_data['t'], colorscale=[[0, 'red'], [1, 'blue']], width=2),
        name='Track Data'
    ))

    # Add the second track (manual_track_data)
    fig.add_trace(go.Scatter3d(
        x=x1, y=y1, z=z1, mode='lines+markers',
        marker=dict(size=5, color='green', opacity=0.8),
        line=dict(color='green', width=2),
        name='Manual Track Data'
    ))


    # Set plot layout
    fig.update_layout(scene=dict(
        xaxis_title='X Axis',
        yaxis_title='Y Axis',
        zaxis_title='Z Axis'
    ), title='3D Scatter Plot Colored by Length')

    # Show the plot
    fig.show()

    # Save the plot as an image file
    fig.write_image("3d_scatter_plot.png")



    napari_open = False
    if napari_open == True:
        import napari
        from napari.utils import nbscreenshot
        single_labelled_mask = (labelled_mask == track_id).astype(int)
        viewer = napari.Viewer()
        viewer.add_image(single_labelled_mask, name='labelled_mask')
        viewer.add_image(raw_data, name='raw_data')
        viewer.add_tracks(track_data[["track_id", "t","z", "y", "x"]], graph=None)
        napari.run()