import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt

def calculate_volumes(labelled_stack, voxel_size):
    """
    Calculate the volumes of labeled regions in a 3D z-stack.

    Parameters:
        labelled_stack (numpy.ndarray): A 3D array where each labeled region has a unique integer value.

    Returns:
        dict: A dictionary where keys are label IDs and values are the volumes (in voxels).
    """
    # Ensure the input is a numpy array
    labelled_stack = np.asarray(labelled_stack)

    # Find unique labels (excluding background, assumed to be 0)
    unique_labels = np.unique(labelled_stack)
    unique_labels = unique_labels[unique_labels != 0]

    # Calculate volumes
    volumes = {}
    for label_id in unique_labels:
        #print(label_id/len(unique_labels))
        volumes[label_id] = np.sum(labelled_stack == label_id) * voxel_size

    return volumes

def filter_volumes(labelled_stack , volumes , vol_limit_list):
    """
    Filter volumes based on a list of volume limits.

    Parameters:
        labelled_stack (numpy.ndarray): A 3D array where each labeled region has a unique integer value.
        volumes (dict): A dictionary where keys are label IDs and values are the volumes (in voxels).
        vol_limit_list (list): A list of volume limits to filter the labels.

    Returns:
        filtered_stack
    """
    # Create a copy of the labelled stack to modify
    filtered_stack = np.copy(labelled_stack)

    # Loop through each label and check its volume
    for label_id, volume in volumes.items():
        #print(label_id)
        if volume < vol_limit_list[0] or volume > vol_limit_list[1]:
            # Set the label to background (0) if outside the volume limits
            filtered_stack[filtered_stack == label_id] = 0

    return filtered_stack

# Example usage
if __name__ == "__main__":
    # Example 3D labeled z-stack
    z_stack_path = r"/home/edwheeler/Documents/cropped_region_3_mixed/b2-2a_2c_pos6-01_deskew_cgt_crop3_8bitt0_seg_masks.tif" 
    z_stack = tiff.imread(z_stack_path)
    # Calculate volumes
    volumes = calculate_volumes(z_stack , voxel_size=0.145**3)

    # Print results
    for label_id, volume in volumes.items():
        print(f"Label {label_id}: Volume = {volume} um^3")

    # Plot histogram of the labels
    labels = list(volumes.keys())
    volumes_list = list(volumes.values())
    plt.hist(volumes_list, bins=50)
    plt.axvline(x=10, color='black', linestyle='dashed', linewidth=1)
    plt.xlabel(r'$\mathbf{Volume\ (\mu m^{3})}$', fontsize=12, fontweight='bold')
    plt.ylabel(r'$\mathbf{Counts}$', fontsize=12, fontweight='bold')
    plt.show()

    # Filter volumes based on limits
    vol_limit_list = [10, 100000]
    filtered_stack = filter_volumes(z_stack, volumes, vol_limit_list)

    # Calculate volumes
    volumes = calculate_volumes(filtered_stack , voxel_size=0.145**3)

    # Print results
    for label_id, volume in volumes.items():
        print(f"Label {label_id}: Volume = {volume} um^3")

    # Plot histogram of the labels
    labels = list(volumes.keys())
    volumes_list = list(volumes.values())
    plt.hist(volumes_list, bins = 50)
    plt.show()
    #filtered_volumes = calculate_volumes(filtered_stack , voxel_size=0.145**3)
    #filtered_volumes_list = list(filtered_volumes.values())
    output_file_path = z_stack_path[0:-4] + '_minsize_' + str(vol_limit_list[0]) + '_filtered.tif'
    #tiff.imwrite(output_file_path, filtered_stack)