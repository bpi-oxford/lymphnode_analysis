import glob
import tifffile as tiff
import os
import argparse
from ultrack.utils.edge import labels_to_contours
import numpy as np

def process_labels_to_contours(segmentation_path, sigma=0):
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
        seg_image = tiff.imread(segmentation_path)

        """
        #For some reason I dont understand, large files seems to lose their shape when loaded/saved with tifffile.
        # This is a workaround to ensure the files have the correct shape.
        if seg_image.shape[0] != raw_image_shape[0] or seg_image.shape[1:] != raw_image_shape[1:]:
                print('wrong shape! Reshaping the file to match raw_shape (ignoring channels).')
                seg_image = np.resize(seg_image, (raw_image_shape[0], *raw_image_shape[1:]))
                seg_image=seg_image.astype(np.uint16)
                print('raw image shape=' +  str(raw_image_shape) + 'segmentation shape = ' + str(seg_image.shape))
        """
               
        foreground, edges = labels_to_contours(seg_image, sigma = sigma, overwrite = True)

        return foreground, edges

def dir_labels_to_contours(input_dir, output_dir, raw_image_path, sigma=0 , file_index=None):


        video_files = sorted(glob.glob(os.path.join(input_dir, '*.tif')))
        if not video_files:
                raise FileNotFoundError(f"No TIF files found in directory: {input_dir}")
        else:
                print(f"Found {len(video_files)} TIF files in directory: {input_dir}")
        
        raw_image = tiff.imread(raw_image_path)
        raw_image_shape = raw_image.shape

        from natsort import natsorted
        video_files = natsorted(video_files)

        if file_index is not None:
                if file_index < 0 or file_index >= len(video_files):
                        raise IndexError(f"file_index {file_index} is out of bounds for the number of files {len(video_files)} in the directory.")
                video_files = [video_files[file_index]]
                print(f"Processing only file at index {file_index}: {video_files[0]}")
        
        for video_path in video_files:
                foreground , edges =process_labels_to_contours(video_path, raw_image_shape, sigma=sigma)
                # Save foreground and edges
                input_name = os.path.splitext(os.path.basename(video_path))[0]
                os.makedirs(output_dir, exist_ok=True)

                foreground_path = os.path.join(output_dir, f'{input_name}_foreground.tif')
                edges_path      = os.path.join(output_dir, f'{input_name}_edges.tif')

                print(foreground_path)
                print(edges_path)

                tiff.imwrite(foreground_path, foreground.astype(np.uint16))
                tiff.imwrite(edges_path, edges)
                print(f"Foreground saved to {foreground_path}")
                print(f"Edges saved to {edges_path}")


        return 


def parent_dir_labels_to_contours(input_dir, output_dir, file_index=None, sigma=0):
        '''
        Description:
        This function processes all TIF files in the input directory to extract foreground and edges using ultrack.
        The results are saved in the output directory.

        Inputs:
        - input_dir: str, path to the parent directory containing segmentation TIF files.
        - output_dir: str, path to save the output foreground and edges TIF files.
        - raw_image_path: str, path to the raw image TIF file.
        - sigma: float, sigma parameter for edge smoothing (default: 0).

        Outputs:
        None (saves output files to output_dir).
        '''
        # Use os.walk to find all .tif files in directory and subdirectories
        seg_files = []
        
        if file_index is None:
                raise ValueError("Please provide a file_index to filter segmentation files.")
        
        else: 
                for root, dirs, files in os.walk(input_dir):
                        for file in files:
                                if file.endswith('.tif') or file.endswith('.tiff'):
                                        if "_segmentation" in file:
                                                full_path = os.path.join(root, file)
                                                seg_files.append(full_path)
        
        seg_files = sorted(seg_files)

        if len(seg_files) == 0:
                raise ValueError(f"No segmentation files found for file_index {file_index} in {input_dir} and subdirectories.")
        
        if file_index is not None:
                if file_index < 0 or file_index >= len(seg_files):
                        raise ValueError(f"file_index {file_index} is out of range. Found {len(seg_files)} .tif files.")

                seg_files = [seg_files[file_index]]
                print(f"Processing only file at index {file_index}: {seg_files[0]}")
        
        for seg_path in seg_files:
                foreground , edges =process_labels_to_contours(seg_path, sigma=sigma)

                # Save foreground and edges
                input_name = os.path.splitext(os.path.basename(seg_path))[0]
                os.makedirs(output_dir, exist_ok=True)

                foreground_path = os.path.join(output_dir, f'{input_name}_foreground.tif')
                edges_path      = os.path.join(output_dir, f'{input_name}_edges.tif')

                print(foreground_path)
                print(edges_path)

                tiff.imwrite(foreground_path, foreground.astype(np.uint16))
                tiff.imwrite(edges_path, edges.astype(np.float32))
                print(f"Foreground saved to {foreground_path}")
                print(f"Edges saved to {edges_path}")


        return 


def single_file_labels_to_contours(seg_path, output_dir,  sigma=0 ):
        seg_image = tiff.imread(seg_path)
        foreground, edges = labels_to_contours(seg_image, sigma = sigma, overwrite = True)
        # Save foreground and edges
        input_name = os.path.splitext(os.path.basename(seg_path))[0]
        os.makedirs(output_dir, exist_ok=True)

        foreground_path = os.path.join(output_dir, f'{input_name}_foreground.tif')
        edges_path      = os.path.join(output_dir, f'{input_name}_edges.tif')

        print(foreground_path)
        print(edges_path)
        tiff.imwrite(foreground_path, foreground.astype(np.uint16))
        tiff.imwrite(edges_path, edges.astype(np.float32))
        print(f"Foreground saved to {foreground_path}")
        print(f"Edges saved to {edges_path}")
        return

if __name__ == "__main__":
        parser = argparse.ArgumentParser(
                description='Process labels to extract foreground and edges using ultrack.'
        )
        parser.add_argument(
                '--seg_path',
                type=str,
                help='Path to the segmentation labels TIF file'
        )
        parser.add_argument(
                '--input_dir',
                type=str,
                help='Path to the segmentation labels TIF directory'
        )
        parser.add_argument(
                '--output_dir',
                type=str,
                help='Path to save the output foreground and edges TIF files'
        )
        parser.add_argument(
                '--sigma',
                type=float,
                default=0,
                help='Sigma parameter for edge smoothing (default: 0)'
        )
        parser.add_argument(
                '--file_index',
                type=int,
                default=None,
                help='Index of the file to process from the input directory (default: None, processes all files)'
        )       
        args = parser.parse_args()
        
        
        single_file_labels_to_contours(
                seg_path  = args.seg_path,
                output_dir = args.output_dir,
                sigma=args.sigma,
        )
        
        print("Processing complete!")

