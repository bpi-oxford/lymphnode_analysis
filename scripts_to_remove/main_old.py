#Idea is to carry out segmentation -> tracking -> relabelling all from one file

#step 2: filter 

import tifffile as tiff
import numpy as np
import os
from cellpose_scripts.cellpose_volume_analyser import calculate_volumes , filter_volumes
from ultrack import MainConfig, Tracker
import ultrack
from ultrack_scripts.helper_function_relabel_segmentation_masks_MPI import relabel_segmentation_masks_MPI
import pandas as pd
from natsort import natsorted
from cellpose_volume_filter_batch import filter_batch_MPI
from cellpose_batch import run_cellpose_timepoint

#load all images in directory that end with seg_masks.tif

if __name__ == "__main__":
    raw_vid_path = r'/home/edwheeler/Documents/cropped_region_1/raw_video/b2-2a_2c_pos6-01_deskew_cgt-cropped_for_segmentation.tif'
    custom_model_path = r"/home/edwheeler/Documents/training_data/train/models/CP_20250430_181517"
    
    segment = False
    if segment == True:
            vid = tiff.imread(raw_vid_path)
            print(vid.shape)
            num_t_steps = vid.shape[0]
            for i in range((num_t_steps)):
                print(i)
                frame = vid[i,...]
                run_cellpose_timepoint(frame , raw_vid_path, custom_model_path,int(i))


    directory = r"/home/edwheeler/Documents/cyto_model_test"
    image_paths = natsorted([os.path.join(directory, f) for f in os.listdir(directory) if f.endswith("seg_masks.tif")])
    image_path = image_paths[0]
    filtering = True
    if filtering == True:
        vol_limit_list = [10 , 1e5]
        filtered_stack = filter_batch_MPI(image_paths=image_paths , vol_limit_list=vol_limit_list)
        output_file_path = image_path[0:-17] + r"_filtered_masks_stack.tif"
        tiff.imwrite(output_file_path, filtered_stack)
    else:
        image_path = image_paths[0]
        output_file_path = image_path[0:-17] + r"_filtered_masks_stack.tif"

    print(image_path)
    filtered_stack =  tiff.imread(r"/home/edwheeler/Documents/cropped_region_3_mixed/b2-2a_2c_pos6-01_deskew_cgt_crop3_8bit__filtered_masks_stack.tif")
    print(filtered_stack.shape)
    #filtered_stack.reshape( (65,182,512,512))
    #print(filtered_stack.shape)

    #step 3: track?
    tracking = False
    if tracking == True:
        labels = filtered_stack.astype(np.uint16)
        print(labels.shape)
        print(type(labels[0,0,0,0]))

        #take a test region
        #labels = labels[0:5 , ...]
        print(labels.shape)

        n_workers = 4

        # Create a config
        config = MainConfig()
        config.data_config.n_workers = n_workers
        config.segmentation_config.n_workers = n_workers
        config.linking_config.max_distance = 25
        config.linking_config.max_neighbors = 10
        config.linking_config.n_workers = n_workers
        # this removes irrelevant segments from the image
        # see the configuration section for more details
        config.segmentation_config.min_frontier = 0.5
        config.tracking_config.division_weight = -1
        config.tracking_config.appear_weight = -0.001
        config.tracking_config.disappear_weight = -0.001
        config.tracking_config.image_border_size = (1, 10, 10)
        print(config)

        # Run the tracking
        tracker = Tracker(config=config)
        tracker.track(labels=labels, overwrite =True)
        # Visualize the results
        tracks, graph = tracker.to_tracks_layer()
        print(tracks)
        output_tracks_path = image_path[0:-17] + '_tracks.csv'
        tracks.to_csv( output_tracks_path)
    else:
        tracks = pd.read_csv(r'/home/edwheeler/Documents/cropped_region_1/seg_frames/b2-2a_2c_pos6-01_deskew_cgt-cropped_for_segmentationt40_seg_ma_tracks.csv')

    #step 4: relabel the segmentation
    masks = relabel_segmentation_masks_MPI(filtered_stack, tracks, 200000)
    output_file_path = output_file_path[0:-4] + 'relabelled2.tif'
    tiff.imwrite( output_file_path, masks)