import os
import glob
import tifffile as tiff
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import zarr
from ultrack import MainConfig, link, solve, to_tracks_layer, tracks_to_zarr, add_flow, segment
from ultrack.tracks import close_tracks_gaps
from ultrack.utils.edge import labels_to_contours
from ultrack.imgproc import inverted_edt


class TissueAnalysis:
    def __init__(self, output_dir , raw_shape):
        self.config = self._initialize_config()
        self.track_df = None
        self.output_dir = output_dir
        self.raw_shape = raw_shape

    def _initialize_config(self):
        config = MainConfig()
        config.segmentation_config.n_workers = 32
        config.data_config.n_workers = 32
        config.data_config.in_memory_db_id = 1
        config.segmentation_config.min_area = 2500
        config.segmentation_config.max_area = 1e10
        config.linking_config.max_distance = 25
        config.linking_config.max_neighbors = 15
        config.segmentation_config.min_frontier = 0.15
        config.tracking_config.division_weight = -10
        config.tracking_config.appear_weight = -0.05
        config.tracking_config.disappear_weight = -0.05
        config.tracking_config.image_border_size = (20, 40, 40)
        config.segmentation_config.max_noise = 0.0
        return config

    def process_labels_to_contours(self, sigma=0 , segmentation_directory_path=None):
        '''
        Description:
        This function processes the labels to extract foreground and edges. Labels must be found in a specific directory.
        If multiple segmentation files are found, they will be combined by ultrack.

        Inputs:
        - find_contours: bool, if True, it will find contours from all the tif files in the segmentation directory. This is 
        useful for when combining multiple gamma filters for example (as ultrack claims to be improved by this - see paper)
        - segmentation_directory_path: str, path to the directory containing segmentation files.

        Outputs:
        - foreground: np.ndarray, the foreground mask.
        - edges: np.ndarray, the edges (values from 0 to 1).
        '''

        if not segmentation_directory_path:
            raise ValueError("Path to segmentation directory must be provided when finding contours.")
        # Load files and find contours
        file_paths = glob.glob(os.path.join(segmentation_directory_path, '*.tif'))
        print(file_paths)
        loaded_files = [tiff.imread(file_path) for file_path in file_paths]
        for file in loaded_files:
            print("Loaded file:")
            print(file.shape)
            print(file.dtype)
        
        #For some reason I dont understand, large files seems to lose their shape when loaded/saved with tifffile.
        # This is a workaround to ensure the files have the correct shape.
        for i, file in enumerate(loaded_files):
            print(file.shape)
            if file.shape[0] != self.raw_shape[0] or file.shape[1:] != self.raw_shape[1:]:
                print('wrong shape! Reshaping the file to match raw_shape (ignoring channels).')
                loaded_files[i] = np.resize(file, (self.raw_shape[0], *self.raw_shape[1:]))
                loaded_files[i]=loaded_files[i].astype(np.uint16)
                print('raw image shape=' +  str(file.shape) + 'segmentation shape = ' + str(loaded_files[i].shape))
        
               
        foreground, edges = labels_to_contours(loaded_files, sigma = sigma, overwrite = True)
        print(edges)
        # Save foreground and edges
        foreground_path = os.path.join(segmentation_directory_path, 'ultrack/foreground.tif')
        print(foreground_path)
        edges_path = os.path.join(segmentation_directory_path, 'ultrack/edges.tif')
        print(edges_path)
        tiff.imwrite(foreground_path, foreground.astype(np.uint16))
        tiff.imwrite(edges_path, edges)
        print(f"Foreground saved to {foreground_path}")
        print(f"Edges saved to {edges_path}")
        return foreground, edges

    def cellpose_segmentation(self, raw_vid_path, custom_model_path, segmentation_directory, gamma_values=None , cellpose_config=None, nprocesses=4):
        '''
        this carries out the cellpose segmentation on the raw_video on frame at a time.
        If gamma_values is provided, it will apply the gamma transformation to the video before segmentation,
        for each value in the list - producing multiple segmentations.

        Inputs:
        --------------------------------------------------
        raw_vid_path: str, path to the raw video file.
        custom_model_path: str, path to the custom cellpose model to use for segmentation.
        segmentation_directory: str, path to the directory where segmentation masks will be saved.
        gamma_values: list, optional, list of gamma values to apply to the video before segmentation. Default is None.
        cellpose_config: dict, optional, dictionary containing cellpose configuration parameters. Default is None. If None provded the
        the default parameters are:
        {
            'flow3D_smooth': 2,
            'batch_size': 8,
            'do_3D': True,
            'diameter': 30,
            'min_size': 25,
            'z_axis': 0
        }

        Outputs:
        --------------------------------------------------
        Saves the segmentation masks to the specified directory with the following naming convention:
        - For segmented video: <video_name>_masks.tif
        - For gamma transformed video: <video_name>_masks_gamma<gamma_value>.tif
        '''

        if gamma_values is not None:
            vid = tiff.imread(raw_vid_path)
            for gamma in gamma_values:
                print(f"Applying gamma transformation with value: {gamma}")
                vid = gamma_transform(vid, gamma)
                video_name = os.path.basename(raw_vid_path).split('.')[0]
                print(f"Processing video: {video_name}, shape: {vid.shape}")
                seg = process_video_with_multiprocessing(vid, custom_model_path ,cellpose_config, nprocesses=nprocesses)
                tiff.imwrite(segmentation_directory + video_name[0:-4] +  'masks_gamma' + str(gamma) +'.tif', seg )
        else:
            print("No gamma values provided. Proceeding with raw video.")
            vid = tiff.imread(raw_vid_path)
            print(f"Processing video: {video_name}, shape: {vid.shape}")
            seg = process_video_with_multiprocessing(vid, custom_model_path ,cellpose_config, nprocesses=nprocesses)
            tiff.imwrite(segmentation_directory + video_name[0:-4] +  'masks.tif'    , seg )
            
    def perform_ultrack_segmentation(self, foreground, edges, overwrite=False):
        '''
        This function performs the extra segmentation using ultrack on the provided foreground and edges. This takes a while!
        '''
        print(f"Foreground shape: {foreground.shape}, dtype: {foreground.dtype}")
        print(f"Edges shape: {edges.shape}, dtype: {edges.dtype}")
        if edges.ndim == 5:
            edges = edges[:,0,...]
            print(f"Reduced edges shape to: {edges.shape}")
        if foreground.ndim == 5:
            foreground = foreground[:,0,...]
            print(f"Reduced foreground shape to: {foreground.shape}")
        segment(foreground, edges, self.config, overwrite=overwrite)

    def add_flow_field(self, raw_vid_path = None, flow_path=None , calculate_flow = False):
        '''
        Adding the flow field to the ultrack config. this may help the tracking process (not clear)
        '''
        if calculate_flow:
            raw_data = tiff.imread(raw_vid_path)
            compute_flow(raw_data=raw_data)
        else:
            flow = zarr.open(flow_path, mode='r')
            print(f"Flow shape: {flow.shape}")
            add_flow(self.config, flow)

    def track_and_solve(self):
        print("Tracking...")
        link(self.config, overwrite=True)
        print("Solving...")
        solve(self.config)
        self.track_df = to_tracks_layer(self.config)[0]
        print(self.track_df)
        self.track_df.to_csv('tracks_temp.csv')

    def save_tracks(self, filename):
        output_path = os.path.join(self.output_dir, filename)
        self.track_df.to_csv(output_path)
        print(f"Tracks saved to {output_path}")

    def relabel_segments(self):
        '''
        Here we relabel the segments using the Ultrack functionand save them to a zarr file.
        However I dont really understand this and therefore I turn it back into a tiff file.

        Inputs:
        --------------------------------------------------
        None
        The track dataframe is an attribute of the class, so it is used directly.
        The data is held in temporary files by ultrack behind the scenes, so we don't need to pass it
        '''
        output_file_path = os.path.join(self.output_dir, 'relabelled_segments.zarr')
        segments = tracks_to_zarr(self.config, self.track_df, store_or_path=output_file_path, overwrite=True)
        print(f"Segments shape: {segments.shape}, dtype: {segments.dtype}")
        segments = zarr.open(output_file_path, mode='r')
        concatenated_segments = np.concatenate(segments, axis=0)  # Concatenate all frames along the first axis
        tiff.imwrite(output_file_path.replace('.zarr', '_concatenated.tif'), concatenated_segments)
        print(f"Zarr file concatenated and converted to 32-bit integer TIFF. Shape: {concatenated_segments.shape}")
        return segments


if __name__ == "__main__":

    '''
    Short scripts to run the TissueAnalysis class for a specific dataset.
    This is a test script for the TissueAnalysis class, which performs segmentation and tracking on a tissue dataset.

    Inputs:
    --------------------------------------------------
    - raw_image_path: str, path to the raw image file.
    - label_path: str, path to the label file (mask video - not used in this script as replaced by segmentation directory).
    - output_dir: str, path to the output directory where results will be saved.
    - segmentation_directory: str, path to the directory where segmentation masks will be saved. THis is also where the labels are loaded from when finding contours in process_labels_to_contours.
    - foreground_path: str, path to the foreground file (if previously calculated).
    - edges_path: str, path to the edges file (if previously calculated).
    - custom_model_path: str, path to the custom cellpose model to use for segmentation.
    - gamma_values: list, optional, list of gamma values to apply to the video before segmentation. Default is [0.75].

    Outputs:
    --------------------------------------------------
    See the class for the outputs of each method.

    '''
    output_dir             = r"/users/kir-fritzsche/aif490/devel/tissue_analysis/lymphnode_analysis/data2track/b2-2a_2c_pos6-01_deskew_cgt/crop3/ultrack_outputs"
    #segmentation_directory = r'/users/kir-fritzsche/aif490/devel/tissue_analysis/lymphnode_analysis/data2track/b2-2a_2c_pos6-01_deskew_cgt/crop2/segmentation/videos'

    foreground_path = r'/users/kir-fritzsche/aif490/devel/tissue_analysis/lymphnode_analysis/data2track/b2-2a_2c_pos6-01_deskew_cgt/crop3/segmentation/edges_foregrounds/combined/zarr/foreground_blur_s0.0.zarr'
    edges_path      = r'/users/kir-fritzsche/aif490/devel/tissue_analysis/lymphnode_analysis/data2track/b2-2a_2c_pos6-01_deskew_cgt/crop3/segmentation/edges_foregrounds/combined/zarr/edges_blur_s1.0.zarr'
    
    foreground = zarr.open(foreground_path, mode='r')['labels_foreground']
    edges      = zarr.open(edges_path, mode='r')['labels_edges']
    
    raw_image_path = r'/users/kir-fritzsche/aif490/devel/tissue_analysis/lymphnode_analysis/data2track/b2-2a_2c_pos6-01_deskew_cgt/crop3/b2-2a_2c_pos6-01_crop_C1_t0-65_z72-328_y598-1622_x568-1592.tiff'

    #flow_path = r'/home/edwheeler/Documents/tissue_analysis_project/tracking_benchmarking/outputs/flow_field_node2_crop1.zarr'
    raw_image = tiff.imread(raw_image_path)
    raw_shape = raw_image.shape
    print(raw_shape)

    analysis = TissueAnalysis(output_dir=output_dir, raw_shape=raw_shape)
    analysis.output_dir=output_dir

    #analysis.cellpose_segmentation(raw_image_path , custom_model_path, segmentation_directory=segmentation_directory, cellpose_config=cellpose_config, gamma_values=gamma, nprocesses=4)
    
    #foreground, edges = analysis.process_labels_to_contours( segmentation_directory_path=segmentation_directory)
    analysis.perform_ultrack_segmentation(foreground, edges, overwrite=False)
    
    #analysis.add_flow_field(raw_vid_path=raw_image_path , calculate_flow=False, flow_path = flow_path)
    
    analysis.track_and_solve()
    analysis.save_tracks(filename= "tracks_crop3_restored_ws_merge.csv")
    
    segments = analysis.relabel_segments()





