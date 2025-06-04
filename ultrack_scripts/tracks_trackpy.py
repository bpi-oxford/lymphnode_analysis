import trackpy as tp
import pandas as pd
import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # Load the data
    tracks_df = pd.read_csv('tracks.csv')
    trackpy_df = tracks_df[['id', 'x', 'y', 'z']]