from multiprocessing import Pool, cpu_count
import numpy as np
import pandas as pd
from collections import defaultdict
from scipy.spatial import cKDTree
import time
from scipy.ndimage import binary_dilation

def process_timepoint(args):
    t1 = time.time()
    print('processing timepoint', args[0])
    t, seg_masks_t, traces_df, max_size = args

    # Get unique segmentation labels at time t
    unique_labels, counts = np.unique(seg_masks_t, return_counts=True)
    label_sizes = dict(zip(unique_labels, counts))

    # Remove large segments
    large_labels = {lbl for lbl, size in label_sizes.items() if size > max_size}
    filtered_mask = np.where(np.isin(seg_masks_t, large_labels), 0, seg_masks_t)

    # Recalculate unique labels after filtering
    unique_labels = np.unique(filtered_mask)
    unique_labels = unique_labels[unique_labels > 0]  # Exclude background (0)

    # Filter traces_df for the current timepoint
    frame_traces = traces_df[traces_df["t"] == t]

    # Create a mapping of segmentation labels to track IDs
    mapping = {}

    for label in unique_labels:
        # Find pixels belonging to this label
        z_coords, y_coords, x_coords = np.where(filtered_mask == label)

        # Find the closest track ID based on spatial coordinates
        mask_centroid = np.array([np.mean(z_coords), np.mean(y_coords), np.mean(x_coords)])

        # Compute distances to all available track positions in this frame
        trace_coords = frame_traces[["z", "y", "x"]].to_numpy()
        if len(trace_coords) > 0:
            distances = np.linalg.norm(trace_coords - mask_centroid, axis=1)
            closest_idx = np.argmin(distances)
            mapping[label] = frame_traces.iloc[closest_idx]["track_id"]
        else:
            mapping[label] = 0  # Default to background if no match

    # Apply mapping to relabel the mask
    mapping[0] = 0  # Ensure background is always mapped to 0
    relabeled_mask = np.vectorize(mapping.get, otypes=[np.int32])(filtered_mask)
    t2=time.time()
    print('done relabeling timepoint', t , 'time taken', t2-t1)
    return np.where(filtered_mask > 0, relabeled_mask, 0)

def are_regions_in_contact(mask, label1, label2):
        """
        Check if two regions in a segmentation mask are in contact.

        Parameters:
        - mask: np.ndarray -> 3D segmentation mask
        - label1: int -> Label of the first region
        - label2: int -> Label of the second region

        Returns:
        - bool -> True if the regions are in contact, False otherwise
        """
        # Create binary masks for the two regions
        region1 = (mask == label1)
        region2 = (mask == label2)

        # Dilate region1 to check for overlap with region2

        dilated_region1 = binary_dilation(region1)
        contact = np.any(dilated_region1 & region2)
        #print('contact', contact)

        return contact

def process_timepoint_ed(args):
    t1 = time.time()
    print('processing timepoint, new method', args[0])
    t, seg_masks_t, traces_df, max_size = args

    # Get unique segmentation labels at time t
    unique_labels, counts = np.unique(seg_masks_t, return_counts=True)
    label_sizes = dict(zip(unique_labels, counts))

    # Remove large segments
    large_labels = {lbl for lbl, size in label_sizes.items() if size > max_size}
    filtered_mask = np.where(np.isin(seg_masks_t, large_labels), 0, seg_masks_t)

    # Recalculate unique labels after filtering
    unique_labels = np.unique(filtered_mask)
    unique_labels = unique_labels[unique_labels > 0]  # Exclude background (0)

    # Filter traces_df for the current timepoint
    frame_traces = traces_df[traces_df["t"] == t]

    # Create a mapping of segmentation labels to track IDs
    mapping = {}

    # Precompute trace coordinates for efficiency
    trace_coords = frame_traces[["z", "y", "x"]].to_numpy()
    tree = cKDTree(trace_coords)

    # Get all non-zero pixel coordinates and their corresponding labels
    z_coords, y_coords, x_coords = np.where(filtered_mask > 0)
    pixel_labels = filtered_mask[z_coords, y_coords, x_coords]

    # Create a DataFrame for pixel coordinates and labels
    pixel_data = pd.DataFrame({
        "z": z_coords,
        "y": y_coords,
        "x": x_coords,
        "label": pixel_labels
    })

    # Compute centroids using groupby
    centroids_df = pixel_data.groupby("label")[["z", "y", "x"]].mean()
    #print(centroids_df)

    # Convert the centroids DataFrame to a dictionary
    centroids = centroids_df.to_dict(orient="index")

    # Match centroids to track IDs using KD-tree
    for label, mask_centroid in centroids.items():
        if trace_coords.size > 0:
            # Query the nearest neighbor using the KD-tree
            _, closest_idx = tree.query(np.array(list(mask_centroid.values())))  # Convert mask_centroid to a numpy array
            track_id = frame_traces.iloc[closest_idx]["track_id"]

            # Check if the track_id has already been assigned to another centroid
            if track_id in mapping.values():
                #print('track_id already assigned, checking distances')
                # Find the label currently assigned to this track_id
                assigned_label = next(lbl for lbl, tid in mapping.items() if tid == track_id)
                
                # Check if the two regions are in contact
                contact = are_regions_in_contact(filtered_mask, label, assigned_label)
                if contact:
                    #print('Regions are in contact, merging them')
                    # Merge the two regions by assigning the same track_id
                    mapping[label] = track_id
                else:
                    # Compute distances for both centroids
                    assigned_centroid = centroids[assigned_label]
                    assigned_distance = np.linalg.norm(np.array(list(assigned_centroid.values())) - trace_coords[closest_idx])
                    current_distance = np.linalg.norm(np.array(list(mask_centroid.values())) - trace_coords[closest_idx])
                    
                    # Reassign based on which centroid is closer
                    if current_distance < assigned_distance:
                        mapping[assigned_label] = 0  # Unassign the previous label
                        mapping[label] = track_id
                    else:
                        mapping[label] = 0  # Default to background if the current one is farther
            else:
                mapping[label] = track_id
        else:
            mapping[label] = 0  # Default to background if no match
    # Apply mapping to relabel the mask
    mapping[0] = 0  # Ensure background is always mapped to 0
    relabeled_mask = np.vectorize(mapping.get, otypes=[np.int32])(filtered_mask)
    t2=time.time()
    print('done relabeling timepoint', t , 'time taken', t2-t1)
    return np.where(filtered_mask > 0, relabeled_mask, 0)


def relabel_segmentation_masks_MPI(seg_masks, traces_df, max_size):
    """
    Relabels a 4D segmentation mask array with track IDs by matching labels from segmentation.

    Parameters:
    - seg_masks: np.ndarray (T, Z, Y, X) -> 4D segmentation masks
    - traces_df: pd.DataFrame with columns ['track_id', 't', 'z', 'y', 'x']

    Returns:
    - new_masks: np.ndarray (T, Z, Y, X) -> Relabeled masks
    """

    # Convert column names to strings (if necessary)
    traces_df.columns = traces_df.columns.astype(str).str.strip()

    # Ensure required columns exist
    required_cols = {"t", "z", "y", "x", "track_id"}
    if not required_cols.issubset(traces_df.columns):
        raise KeyError(f"Missing columns in traces_df. Expected: {required_cols}, Found: {set(traces_df.columns)}")

    # Get array shape
    T, Z, Y, X = seg_masks.shape

    # Prepare arguments for multiprocessing
    args = [(t, seg_masks[t], traces_df, max_size) for t in range(T)]

    # Use multiprocessing to process each timepoint
    with Pool(processes=cpu_count()-8) as pool:
        print('Starting relabeling with multiprocessing...')
        results = pool.map(process_timepoint_ed, args)

    # Combine results into a single array
    new_masks = np.stack(results, axis=0)

    return new_masks

if __name__ == "__main__":
    # Example usage
    import tifffile as tiff


    print('running relabeling script')
    seg_masks_path = r"/home/edwheeler/Documents/cropped_region_2_motile/b2-2a_2c_pos6-01_deskew_cgt_cropped_for_segmentation_motile_region_seg_masks_minsize_10_filtered_stack.tif"
    seg_masks = tiff.imread(seg_masks_path)
    tracks_df_path = r"/home/edwheeler/Documents/tissue_analysis_project/tracking_benchmarking/outputs/tracks_crop2_motile_15.csv"
    tracks_df = pd.read_csv(tracks_df_path)
    args = (0, seg_masks[0], tracks_df, 10000)
    process_timepoint_ed(args)