import numpy as np
from scipy.ndimage import convolve
import os
from glob import glob
from tifffile import imread
from tifffile import imwrite
from natsort import natsorted
import matplotlib.pyplot as plt
'''
takes a 2D bundary segmentation and extracts the edges
'''

def boundary_to_edges(segmentation):
    gx = convolve(segmentation + 1 , np.array([-1. , 0. , 1.]).reshape(1,3))
    gy = convolve(segmentation + 1 , np.array([-1. , 0. , 1.]).reshape(3,1))
    edges_binary = ((gx**2 + gy**2) > 0)
    return edges_binary

directory = r"/home/edwheeler/Documents/cropped_region_2_motile/figures/for_ilastik"
tif_files = natsorted(glob(os.path.join(directory, "*Segmentation.tif")), key=lambda x: os.path.basename(x))

if tif_files:
    first_image = imread(tif_files[0])
    shape = first_image.shape

output_tiff = np.zeros( (len(tif_files) , shape[0] , shape[1] ))

for i in range(len(tif_files)):
    file = tif_files[i]
    print(file)
    segmentation = imread(file)
    plt.imshow(segmentation)
    plt.show()
    # Find the label of the region at the image center
    center_x, center_y = segmentation.shape[0] // 2, segmentation.shape[1] // 2
    center_label = segmentation[center_x, center_y]
    
    # Set all other regions in the segmentation to background
    segmentation[segmentation != center_label] = 0
    plt.imshow(segmentation)
    plt.show()
    edges = boundary_to_edges(segmentation)
    output_tiff[i , ...] = edges

output_tiff = (output_tiff * 255).astype(np.uint8)

imwrite('./single_cell_boundary_multicut.tif' , output_tiff)

