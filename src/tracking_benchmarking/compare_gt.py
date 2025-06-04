'''
Compare manual tracks to a labeled image
'''
import pandas as pd
import tifffile as tiff
import numpy as np

def compare_tracks(manual_track_df, labeled_image):
    """

    Parameters:
    - manual_df: DataFrame containing a manual track.
    - labeled_image_path: Path to the labeled image.

    Returns:
    - Labels: array of the label of the cell that is closest to
    the position of the manual track
    """

    # Add a new column to store the label values
    manual_track_df['label'] = None
    labels = np.zeros((len(manual_track_df), 1), dtype=int)

    # Iterate through each row in the manual track DataFrame
    for i in range(len(manual_track_df)):
        time = manual_track_df.iloc[i]['t']
        z = manual_track_df.iloc[i]['z']
        y = manual_track_df.iloc[i]['y']
        x = manual_track_df.iloc[i]['x']
        time , z , x , y = int(time), int(z), int(x), int(y)

        #if out of bounds - set to nearest boundary
        if z > labeled_image.shape[1]:
            z = labeled_image.shape[1] - 1

        
        labeled_image_frame = labeled_image[time-1,...]
        label = labeled_image_frame[z, y, x]

        if label == 0:
            # Find the closest non-zero value in the array
            non_zero_indices = np.argwhere(labeled_image_frame > 0)
            distances = np.sqrt((non_zero_indices[:, 0] - z)**2 + 
                                 (non_zero_indices[:, 1] - y)**2 + 
                                 (non_zero_indices[:, 2] - x)**2)
            closest_index = np.argmin(distances)
            label = labeled_image_frame[tuple(non_zero_indices[closest_index])]
        labels[i] = label
        #print(labels)
    return labels

def mistakes_per_track(label_arr):
    """
    Calculate the number of mistakes per track.

    Parameters:
    - label_arr: Array containing the labels for each track.

    Returns:
    - Mistakes - number of times the label changes per track.
    """
    mistake_counter = 0
    mistake_id = np.nan
    for i in range(len(label_arr)):
        # Check if the label is different from the previous one
        if i > 0 and label_arr[i] != label_arr[i - 1]:
            # Increment the mistake count
            print(f"Label changed at index {i}: {label_arr[i]} (previous: {label_arr[i - 1]})")
            # Placeholder for mistake calculation
            if mistake_counter == 0:
                mistake_id = i
            mistake_counter += 1

    print(f"Total mistakes: {mistake_counter}")
        

    return mistake_counter , mistake_id
 
def remove_identical_tracks(manual_df):
    """
    Remove identical tracks from the DataFrame.  
    Just a simple function to remove tracks that I accidentally duplicated

    Parameters:
    - manual_df: DataFrame containing manual tracks.

    Returns:
    - DataFrame with identical tracks removed.
    """
    track_ids = manual_df['track id'].unique()
    for track_id in track_ids:
        track = manual_df[manual_df['track id'] == track_id]
        # Check if the track is identical to any other track
        identical_tracks = manual_df[manual_df['track id'] != track_id].groupby('track id').filter(
            lambda group: len(group) == len(track) and 
            np.all(np.sqrt((group['t'].values - track['t'].values)**2 + 
                   (group['z'].values - track['z'].values)**2 + 
                   (group['y'].values - track['y'].values)**2 + 
                   (group['x'].values - track['x'].values)**2) == 0)
        )
        if not identical_tracks.empty:
            identical_track_ids = identical_tracks['track id'].unique()
            print(f"Track id {track_id} is identical to track ids: {identical_track_ids}")

        # If there are identical tracks, remove them
        if len(identical_tracks) > 1:
            print(f"Removing identical tracks for track id {track_id}")

            manual_df = manual_df[~manual_df.index.isin(identical_tracks.index)]
        
    return manual_df

def traccuarcy(manual_df, labeled_image):

    """
    Calculate the accuracy of the manual tracks compared to the labeled image.

    Parameters:
    - manual_df: DataFrame containing manual tracks.
    - labeled_image: Labeled image array.

    Returns:
    - Mistakes df: 
    columns are track ids, 
    initial label :initial label of the track,
    mistakes: the number of 'mistakes' (the label changing),
    mistake time id : time id of the first mistake
    total length: the length of the track
    
    """
    # Placeholder for accuracy calculation
    mistake_no_list = []
    track_ids_list = []
    mistake_id_list = []
    initial_label_list = []
    track_len_list = []
    track_ids = manual_df['track id'].unique()
    print(track_ids)
    for track_id in track_ids:
        track = manual_df[manual_df['track id'] == track_id]
        track_labels = compare_tracks(track, labeled_image)
        mistake_no , mistake_id = mistakes_per_track(track_labels)
        mistake_no_list.append(mistake_no)
        mistake_id_list.append(mistake_id)
        track_ids_list.append(track_id)
        track_len_list.append(len(track))
        initial_label_list.append(track_labels[0][0])
    
    mistakes_df = pd.DataFrame({'track id': track_ids_list, 'initial label' : initial_label_list, 'mistakes': mistake_no_list , 'mistake time id': mistake_id_list, 'total length': track_len_list})

        # Calculate accuracy for each track

    return mistakes_df


if __name__ == "__main__":
    manual_tracks_path = '/home/edwheeler/Documents/cropped_region_1/manual_tracks_crop1.csv'
    labeled_image_path = '/home/edwheeler/Documents/tissue_analysis_project/tracking_benchmarking/outputs/relabelled_crop1_ultrack15noblur_concatenated.tif'
    manual_df = pd.read_csv(manual_tracks_path)  # Load CSV with specific column names
    print(manual_df)
    #manual_df.columns = ['track id', 't', 'z' , 'y' , 'x']  # Replace with actual column names
    
    # Load the labeled image
    labeled_image = tiff.imread(labeled_image_path)
    labeled_image = labeled_image.reshape( (65 , 120 , 256 , 256))

    # Compare the manual track to the labeled image
    print(manual_df)
    manual_df = remove_identical_tracks(manual_df)

    track = manual_df[manual_df['track id'] == 22]
    compare_tracks(track, labeled_image)
    mistakes_df = traccuarcy(manual_df, labeled_image)
    print(mistakes_df)

    non_nan_mistakes_count = mistakes_df['mistake time id'].notna().sum()
    print(f"Number of non-NaN values in mistakes_df['mistakes']: {non_nan_mistakes_count}")

    print(np.mean(mistakes_df['mistakes']))
    print(np.sum(mistakes_df['mistakes']) / np.sum(mistakes_df['total length']))

    
