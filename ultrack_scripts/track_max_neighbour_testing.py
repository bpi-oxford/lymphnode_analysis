'''
script to run the ultrack tracking_step with various parameters
'''
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import tifffile as tiff
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ultrack
from ultrack import MainConfig, Tracker,add_flow, segment, link, solve, to_tracks_layer, tracks_to_zarr, to_ctc
from ultrack.utils.edge import labels_to_contours
from ultrack.tracks.stats import tracks_df_movement
from ultrack.imgproc import inverted_edt
from helper_function_relabel_segmentation_masks_MPI import relabel_segmentation_masks_MPI
from ultrack.tracks import close_tracks_gaps
import zarr
import glob
import dask.array as da

def linking_param_search(output_dir, parameter_dict ):
    n_workers         = parameter_dict['n_workers']
    max_distance      = parameter_dict['max_distance']
    max_neighbors     = parameter_dict['max_neighbours']
    appear_weight     = parameter_dict['appear_weight']
    disappear_weight  = parameter_dict['disappear_weight']
    division_weight   = parameter_dict['division_weight']
    image_border_size = parameter_dict['image_border_size']
    min_frontier      = parameter_dict['min_frontier']


    #set up configuration
    config = MainConfig()
    config.segmentation_config.n_workers     = n_workers
    config.linking_config.max_distance       = max_distance
    config.linking_config.n_workers          = n_workers
    config.linking_config.max_neighbors      = max_neighbors
    config.data_config.n_workers             = n_workers
    config.segmentation_config.min_frontier  = min_frontier
    config.tracking_config.division_weight   = division_weight
    config.tracking_config.appear_weight     = appear_weight
    config.tracking_config.disappear_weight  = disappear_weight
    config.tracking_config.image_border_size = image_border_size

    #LINK
    link(config , overwrite=True)
    solve(config)
    track_df = to_tracks_layer(config)[0]
    print(track_df)

    output_track_path = output_dir + 'tracks_crop2_maxN' + str(max_neighbors) + '_maxD' + str(max_distance) + '.csv'
    track_df.to_csv(output_track_path)
    return

if __name__ == "__main__":
    label_path = r"/home/edwheeler/Documents/cropped_region_2_motile/b2-2a_2c_pos6-01_deskew_cgt_cropped_for_segmentation_motile_region_seg_masks_minsize_10_filtered_stack.tif"
    labels = tiff.imread(label_path)
    #raw_data_path = r"/home/edwheeler/Documents/cropped_region_2_motile/b2-2a_2c_pos6-01_deskew_cgt_cropped_for_segmentation_motile_region.tif"
    #raw_data = tiff.imread(raw_data_path) 

    output_dir = r'/home/edwheeler/Documents/tissue_analysis_project/tracking_benchmarking/outputs/'
    #first define the configuration:
    n_workers = 4
    # Create a config
    config = MainConfig()
    config.segmentation_config.n_workers = n_workers
    config.segmentation_config.min_area = 10000
    config.segmentation_config.max_area = 1e10
    config.data_config.n_workers = n_workers
    config.linking_config.max_distance = 50
    config.linking_config.n_workers = n_workers
    config.linking_config.max_neighbors = 15
    config.data_config.n_workers = n_workers
    # this removes irrelevant segments from the image
    # see the configuration section for more details
    config.segmentation_config.min_frontier = 0.00
    config.tracking_config.division_weight = -10
    config.tracking_config.appear_weight = -0.05
    config.tracking_config.disappear_weight = -0.05
    config.tracking_config.image_border_size = (20, 40, 40)
    config.segmentation_config.max_noise = 0.0
    #config.data_config.database_path = r'sqlite:////home/edwheeler/Documents/data_crop1.db'


    find_contours = False
    find_edt = False
    do_segmentation = True
    use_flow = True
    Track = True
    relabel = True
 
    '''   
    # Load all files from the directory containing the keyword 'combined'
    directory_path = r'/home/edwheeler/Documents/cropped_region_2_motile/gamma_trans'
    keyword = 'gamma'
    file_paths = glob.glob(os.path.join(directory_path, f'*{keyword}*'))

    # Read the files and store them in a list
    loaded_files = [tiff.imread(file_path) for file_path in file_paths]
    loaded_files_list = list(loaded_files)
    print(f"Loaded {len(loaded_files_list)} files containing the keyword '{keyword}'.")
    '''
    

    if find_contours:
         foreground, edges = labels_to_contours(loaded_files_list)
         output_path = r'/home/edwheeler/Documents/tissue_analysis_project/tracking_benchmarking/outputs/'
         tiff.imwrite(output_path + 'foreground_crop2_motile_gamma.tif' , foreground)
         tiff.imwrite(output_path + 'edges_crop2_motile._gamma.tif'      , edges)
    else:
         foreground = tiff.imread(r'/home/edwheeler/Documents/tissue_analysis_project/tracking_benchmarking/outputs/foreground_crop1_gamma.tif')
         edges      = tiff.imread(r'/home/edwheeler/Documents/tissue_analysis_project/tracking_benchmarking/outputs/edges_crop1_gamma.tif')
         foreground = foreground[: , 0:120 , : , :]
         edges      = edges[: , 0:120 , : , :]

    if find_edt:
        edt_edges = np.zeros_like(foreground)
        edt_edges = edt_edges.astype(np.float32)
        for t in range(foreground.shape[0]):
            print(t)
            edt_edges_frame = inverted_edt(foreground[t,...])
            edt_edges[t,...] = edt_edges_frame

        output_path = output_dir + 'edges_crop2_edt.tif'
        tiff.imwrite(output_path, edt_edges.astype(np.float16))
    else:
        edt_edges = tiff.imread(r'/home/edwheeler/Documents/tissue_analysis_project/tracking_benchmarking/outputs/edges_crop2_edt.tif')

    if do_segmentation:
        print(foreground.shape)
        print(foreground.dtype)
        print(edges.shape)
        print(edges.dtype)
        segment(foreground, edges , config, overwrite=True)

    if use_flow:
        flow_path = r'/home/edwheeler/Documents/tissue_analysis_project/tracking_benchmarking/outputs/flow_field_crop1.zarr'
        flow = zarr.open(flow_path, mode='r')
        print(flow.shape)#
        add_flow(config, flow)


    if Track:
        print('tracking')
        link(config, overwrite=True)
        print('solving')
        solve(config)
        track_df = to_tracks_layer(config)[0]
        print(track_df)
        output_track_path = output_dir + 'tracks_crop1_' + str(config.linking_config.max_neighbors) + '.csv'
        track_df.to_csv(output_track_path)
    else:
        track_df = pd.read_csv(r'/home/edwheeler/Documents/tissue_analysis_project/tracking_benchmarking/outputs/tracks_crop2_motile_15.csv')
        print(track_df)


    if relabel:

        output_file_path = output_dir + 'relabelled_crop1_ultrack' + str(config.linking_config.max_neighbors)+ '.zarr'
        segments = tracks_to_zarr(
            config,
            track_df,
            store_or_path=output_file_path,
            overwrite=True,
        )
        print(f"Segments shape: {segments.shape}")
        print(f"Segments dtype: {segments.dtype}")
        #print(f"Segments data (sample): {segments[:10]}")  # Print a small sample of the data
        print(np.max(segments))
        

        segments = zarr.open(output_file_path, mode='r')

        closed_path = output_file_path.replace('.zarr', '_closed.zarr')
        track_df = close_tracks_gaps(track_df, max_gap=1, max_radius=35, segments=segments, overwrite=True , segments_store_or_path=closed_path)[0]
        print(config.data_config.database_path)
        print(track_df)
        track_df.to_csv(output_track_path)

        segments = zarr.open(closed_path, mode='r')
        
        


        print(np.max(segments))
        # Plot the first frame of the segments
        first_frame = segments[0]
        plt.figure(figsize=(10, 10))
        plt.imshow(first_frame[40,...], cmap='gray')
        plt.title("First Frame of Segments")
        plt.axis('off')
        plt.show()

        concatenated_segments = np.concatenate(segments, axis=0)  # Concatenate all frames along the first axis
        tiff.imwrite(output_file_path.replace('.zarr', 'noblur_concatenated.tif'), concatenated_segments)
        print(f"Zarr file concatenated and converted to 32-bit integer TIFF. Shape: {concatenated_segments.shape}")

    

        #masks = relabel_segmentation_masks_MPI(labels, track_df, 200000)
        #tiff.imwrite( output_file_path, masks)