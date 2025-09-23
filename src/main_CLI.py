import argparse
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
from cellpose_scripts.cellpose_batch import process_video_with_multiprocessing
from cellpose_scripts.gamma_segmentation import gamma_transform
from ultrack_scripts.flow_computation import compute_flow


class TissueAnalysis:
    def __init__(self, output_dir , raw_shape):
        self.config = self._initialize_config()
        self.track_df = None
        self.output_dir = output_dir
        self.raw_shape = raw_shape

    def _initialize_config(self):
        config = MainConfig()
        config.segmentation_config.n_workers = 6
        config.segmentation_config.min_area = 1000
        config.segmentation_config.max_area = 1e10
        config.linking_config.max_distance = 25
        config.linking_config.max_neighbors = 15
        config.segmentation_config.min_frontier = 0.05
        config.tracking_config.division_weight = -10
        config.tracking_config.appear_weight = -0.05
        config.tracking_config.disappear_weight = -0.05
        config.tracking_config.image_border_size = (20, 40, 40)
        config.segmentation_config.max_noise = 0.05
        return config

    def process_labels_to_contours(self, segmentation_directory_path=None):
        '''
        Description:
        This function processes the labels to extract foreground and edges. Labels must be found in a specific directory.
        If multiple segmentation files are found, they will be combined by ultrack.

        Inputs:
        - find_contours: bool, if True, it will find contours from all the tif files in the segmentation directory. This is 
        useful for when combining multiple gamma filters for example (as ultrack claims to be improved by this - see paper)
        - segmentation_directory_path: str, path to the directory containing segmentation files.
        - foreground_path: str, path to the foreground file. (If previously calculated)
        - edges_path: str, path to the edges file. (If previously calculated)

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
        
        #For some reason I dont understand, large files seems to lose their shape when loaded/saved with tifffile.
        # This is a workaround to ensure the files have the correct shape.
        for i, file in enumerate(loaded_files):
            print(file.shape)
            if file.shape[0] != self.raw_shape[0] or file.shape[1:] != self.raw_shape[1:]:
                print('wrong shape! Reshaping the file to match raw_shape (ignoring channels).')
                loaded_files[i] = np.resize(file, (self.raw_shape[0], *self.raw_shape[1:]))
                loaded_files[i]=loaded_files[i].astype(np.uint8)
                print('raw image shape=' +  str(file.shape) + 'segmentation shape = ' + str(loaded_files[i].shape))
        
               
        foreground, edges = labels_to_contours(loaded_files)
        print(edges)
        # Save foreground and edges
        foreground_path = os.path.join(segmentation_directory_path, 'foreground.tif')
        edges_path = os.path.join(segmentation_directory_path, 'edges.tif')
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
            
    def perform_ultrack_segmentation(self, foreground, edges):
        '''
        This function performs the extra segmentation using ultrack on the provided foreground and edges. This takes a while!
        '''
        print(f"Foreground shape: {foreground.shape}, dtype: {foreground.dtype}")
        print(f"Edges shape: {edges.shape}, dtype: {edges.dtype}")
        segment(foreground, edges, self.config, overwrite=True)

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


def parse_arguments():
    parser = argparse.ArgumentParser(description='TissueAnalysis CLI for segmentation and tracking')
    
    parser.add_argument('--raw_image_path', required=True, help='Path to the raw image file')
    parser.add_argument('--output_dir', required=True, help='Path to the output directory')
    parser.add_argument('--segmentation_directory', required=True, help='Path to the segmentation directory')
    parser.add_argument('--custom_model_path', required=True, help='Path to the custom cellpose model')
    parser.add_argument('--flow_path', help='Path to the flow field file')
    parser.add_argument('--gamma_values', nargs='+', type=float, default=[0.75], help='Gamma values for transformation')
    parser.add_argument('--tracks_filename', default='tracks.csv', help='Output filename for tracks')
    
    # Cellpose configuration arguments
    parser.add_argument('--flow3d_smooth', type=int, default=2, help='Flow 3D smooth parameter')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for cellpose')
    parser.add_argument('--do_3d', action='store_true', default=True, help='Enable 3D processing')
    parser.add_argument('--diameter', type=int, default=30, help='Cell diameter for cellpose')
    parser.add_argument('--min_size', type=int, default=500, help='Minimum cell size')
    parser.add_argument('--z_axis', type=int, default=0, help='Z axis for 3D processing')
    parser.add_argument('--nprocesses', type=int, default=4, help='Number of processes for multiprocessing')
    
    # Pipeline control flags
    parser.add_argument('--run_cellpose', action='store_true', help='Run cellpose segmentation')
    parser.add_argument('--calculate_flow', action='store_true', help='Calculate flow field')
    
    return parser.parse_args()


def main():
    args = parse_arguments()
    
    # Load raw image to get shape
    raw_image = tiff.imread(args.raw_image_path)
    raw_shape = raw_image.shape
    print(f"Raw image shape: {raw_shape}")
    
    # Initialize analysis
    analysis = TissueAnalysis(output_dir=args.output_dir, raw_shape=raw_shape)
    
    # Configure cellpose parameters
    cellpose_config = {
        'flow3D_smooth': args.flow3d_smooth,
        'batch_size': args.batch_size,
        'do_3D': args.do_3d,
        'diameter': args.diameter,
        'min_size': args.min_size,
        'z_axis': args.z_axis
    }
    
    # Run cellpose segmentation if requested
    if args.run_cellpose:
        print("Running cellpose segmentation...")
        analysis.cellpose_segmentation(
            args.raw_image_path, 
            args.custom_model_path, 
            segmentation_directory=args.segmentation_directory, 
            cellpose_config=cellpose_config, 
            gamma_values=args.gamma_values, 
            nprocesses=args.nprocesses
        )
    
    # Process labels to contours
    print("Processing labels to contours...")
    foreground, edges = analysis.process_labels_to_contours(segmentation_directory_path=args.segmentation_directory)
    
    # Perform ultrack segmentation
    print("Performing ultrack segmentation...")
    analysis.perform_ultrack_segmentation(foreground, edges)
    
    # Add flow field if provided
    if args.flow_path or args.calculate_flow:
        print("Adding flow field...")
        analysis.add_flow_field(
            raw_vid_path=args.raw_image_path if args.calculate_flow else None,
            calculate_flow=args.calculate_flow,
            flow_path=args.flow_path
        )
    
    # Track and solve
    print("Tracking and solving...")
    analysis.track_and_solve()
    analysis.save_tracks(filename=args.tracks_filename)
    
    # Relabel segments
    print("Relabeling segments...")
    segments = analysis.relabel_segments()
    
    print("Analysis complete!")


if __name__ == "__main__":
    main()