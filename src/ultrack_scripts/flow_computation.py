import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ultrack import MainConfig, Tracker,add_flow, segment, link, solve, to_tracks_layer, tracks_to_zarr
from ultrack.utils.edge import labels_to_contours
from ultrack.tracks.stats import tracks_df_movement
from ultrack.imgproc import inverted_edt 
from ultrack.imgproc.flow import timelapse_flow
import tifffile as tiff

'''
scripts to find and save the flow field
'''

def compute_flow(raw_data , output_path=None):
    if output_path is not None:
        flow = timelapse_flow(raw_data , output_path)
    else:
        flow = timelapse_flow(raw_data)
    return flow

if __name__ == "__main__":
    raw_data_path = r"/home/edwheeler/Documents/node2_crop1/b2-6a_overview_pos1-01_deskew_cgt_lymphnode2_crop1_t1.tif"
    raw_data = tiff.imread(raw_data_path)
    print(raw_data.shape)
    raw_data = raw_data[:,0,...] 
    print(raw_data.shape)
    output_path = r'/home/edwheeler/Documents/tissue_analysis_project/tracking_benchmarking/outputs/flow_field_node2_crop1.zarr'
    flow = compute_flow(raw_data , output_path) # , output_path)
    print(flow)
    