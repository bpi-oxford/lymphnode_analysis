'''
code to take the 3D segmentation and find the interfaces

initially start off by just extracting the 3D segmentation of one cell - just some ideas no solid code yet
not ready for multiple cells?

Then find the interface between two cells
'''

import numpy as np
import pandas as pd
from skimage.measure import marching_cubes
from scipy.ndimage import distance_transform_edt

def find_surface(labelled_image):
    # Assuming labelled_image is a 3D numpy array where the cell of interest is labeled with a specific integer
    verts, faces, normals, values = marching_cubes(labelled_image, level=0.5)
    
    # Create a binary image of the same shape as labelled_image
    surface_image = np.zeros_like(labelled_image, dtype=np.uint8)
    
    # Convert the vertices to integer indices
    verts = np.round(verts).astype(int)
    
    # Mark the surface points in the binary image
    surface_image[verts[:, 0], verts[:, 1], verts[:, 2]] = 1

    return surface_image

def surface_video(labelled_video):
    # Assuming labelled_video is a 4D numpy array (time, z, y, x)
    video_shape = labelled_video.shape
    surface_video = np.zeros_like(labelled_video, dtype=np.uint8)
    for t in range(video_shape[0]):
        # Extract the labelled image at time t
        labelled_image = labelled_video[t, :, :, :]
        
        # Find the surface of the labelled image
        surface_image = find_surface(labelled_image)
        
        # Store the surface image in the video
        surface_video[t, :, :, :] = surface_image
    return surface_video

def find_interface(labelled_two_cell_image ,  distance_threshold):
    '''
    -Input A labelled image of just two cells each with distinct labels
    '''
    # Create binary masks for each cell
    cell1_mask = labelled_two_cell_image == 1
    cell2_mask = labelled_two_cell_image == 2

    # Compute the distance transform for each cell
    distance_to_cell1 = distance_transform_edt(~cell1_mask)
    distance_to_cell2 = distance_transform_edt(~cell2_mask)

    # Find the interface region where both distances are below the threshold
    interface_region = (distance_to_cell1 <= distance_threshold) & (distance_to_cell2 <= distance_threshold)

    return interface_region.astype(np.uint8)


