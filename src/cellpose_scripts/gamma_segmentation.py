'''
Try and use the gamma transform to provide multiple segmentation outputs for ultrack to use
'''
import numpy as np
import tifffile as tiff
import pandas as pd
import matplotlib.pyplot as plt
from cellpose_scripts.cellpose_batch import process_video_with_multiprocessing

'''
A simple function to apply gamma transformation to an image. The gamma transformation is called 
in the main_class.py script - cellpose_segmentation function. 
'''

def gamma_transform(image, gamma):
    """
    Apply gamma transformation to the image.

    INPUT:
    ----------------------------------------------------
    image: np.ndarray. The input image to be transformed.
    gamma: float. The gamma value for the transformation.

    OUTPUT:
    ----------------------------------------------------
    transformed_image: np.ndarray. The gamma-transformed image.

    """
    # Normalize the image to [0, 1]
    image_normalized = (image - np.min(image)) / (np.max(image) - np.min(image))
    
    # Apply gamma transformation
    transformed_image = np.power(image_normalized, gamma)
    
    # Rescale back to original range
    transformed_image = transformed_image * (np.max(image) - np.min(image)) + np.min(image)
    
    return transformed_image

