import cellpose
import numpy as np
from cellpose import models
import tifffile as tiff
from multiprocessing import Pool

'''
functions to run cellpose on a 3D video timepoint by timepoint. With multiprocessing support.

'''

def run_cellpose_timepoint_with_MPI(args):
    vid_frame, custom_model_path, cellpose_config_dict, timepoint = args
    print('running cellpose on timepoint', timepoint)
    print(vid_frame.shape)
    
    # Default parameters for cellpose if cellpose_config_dict is not provided
    default_config = {
        'flow3D_smooth': 2,
        'batch_size': 8,
        'do_3D': True,
        'diameter': 30,
        'min_size': 25,
        'z_axis': 0
    }
    
    # Merge provided config with default config, using defaults for missing fields
    config = {**default_config, **(cellpose_config_dict or {})}
    
    model = models.CellposeModel(gpu=True, pretrained_model=custom_model_path)
    masks, outlines, flows = model.eval(
        vid_frame,
        flow3D_smooth=config['flow3D_smooth'],
        batch_size=config['batch_size'],
        do_3D=config['do_3D'],
        diameter=config['diameter'],
        min_size=config['min_size'],
        z_axis=config['z_axis']
    )
    return masks

def process_video_with_multiprocessing(vid, custom_model_path, cellpose_config_dict=None, nprocesses=4):
    '''
    process a 3D video with cellpose using multiprocessing.

    Inputs:
    --------------------------------------------------
    vid: 4D numpy array of the video to segment (shape: [t, z, y, x])
    custom_model_path: path to the custom cellpose model to use for segmentation
    cellpose_config_dict: dictionary containing cellpose configuration parameters (optional)
    nprocesses: number of processes to use for multiprocessing (default: 4)
    Outputs:
    --------------------------------------------------
    results_array: 4D numpy array of the segmentation masks for the video (shape: [t, z, y, x])
    '''
    num_t_steps = vid.shape[0]

    args_list = [(vid[i, ...], custom_model_path, cellpose_config_dict, i) for i in range(num_t_steps)]
    with Pool(nprocesses) as pool:
        results = pool.map(run_cellpose_timepoint_with_MPI, args_list)
        results_array = np.vstack(results)
        return results_array



if __name__ == "__main__":

    #input video
    vid_path = r"/home/edwheeler/Documents/denoising/predict_data_crop1_3d/b2-2a_2c_pos6-01_deskew_cgt-cropped_for_segmentationt0_n2v.tif"
    vid = tiff.imread(vid_path)
    print(vid.shape)
    custom_model_path = r"/home/edwheeler/Documents/training_data/train/models/CP_20250430_181517"

    num_t_steps = vid.shape[0]
    process_video_with_multiprocessing(vid, custom_model_path, nprocesses=4)

