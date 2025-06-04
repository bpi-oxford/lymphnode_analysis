def relabel_segmentation_masks(seg_masks, traces_df, max_size):
    """
    Relabels a 4D segmentation mask array with track IDs by matching labels from segmentation.

    Parameters:
    - seg_masks: np.ndarray (T, Z, Y, X) -> 4D segmentation masks
    - traces_df: pd.DataFrame with columns ['track_id', 't', 'z', 'y', 'x']

    Returns:
    - new_masks: np.ndarray (T, Z, Y, X) -> Relabeled masks


    example call:
    new_detection = relabel_segmentation_masks(detection,df_mate,200000)
    """

    import numpy as np

    # Convert column names to strings (if necessary)
    traces_df.columns = traces_df.columns.astype(str).str.strip()

    # Ensure required columns exist
    required_cols = {"t", "z", "y", "x", "track_id"}
    if not required_cols.issubset(traces_df.columns):
        raise KeyError(f"Missing columns in traces_df. Expected: {required_cols}, Found: {set(traces_df.columns)}")

    # Get array shape
    T, Z, Y, X = seg_masks.shape

    # Initialize a new array for relabeled masks
    new_masks = np.zeros_like(seg_masks, dtype=np.int32)

    # Iterate over timepoints
    for t in range(T):
        print(t/T)
        # Get unique segmentation labels at time t
        unique_labels, counts = np.unique(seg_masks[t], return_counts=True)
        label_sizes = dict(zip(unique_labels, counts))

        # Remove large segments
        large_labels = {lbl for lbl, size in label_sizes.items() if size > max_size}
        print(large_labels)
        filtered_mask = np.where(np.isin(seg_masks[t], large_labels), 0, seg_masks[t])

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
        new_masks[t] = np.where(filtered_mask > 0, relabeled_mask, 0)

    return new_masks