"""
searches in a directory for all foreground fiels and all edges files

foreground files are combined with a logical OR operation
edges files are combined by averaging their values

"""

def combine_edges(input_dir):
    import glob
    from pathlib import Path
    import numpy as np
    import tifffile as tiff
    from scipy.ndimage import gaussian_filter

    input_dir = Path(input_dir)

    # Find all edges files
    edges_files = sorted(glob.glob(str(input_dir / "*_edges.tif")))

    if len(edges_files) == 0:
        raise ValueError("No edges files found in the specified directory.")
    print(f"Found {len(edges_files)} edges files.")


    print(edges_files)
    
    # Initialize combined arrays

    combined_edges = None

    # Combine edges files by averaging
    edge_sum = None
    for edge_file in edges_files:
        edge_image = tiff.imread(edge_file).astype(np.float32)
        if edge_sum is None:
            edge_sum = edge_image
        else:
            edge_sum += edge_image

    combined_edges = edge_sum / len(edges_files)

    #blur the edges in (Z,Y,X)
    shape = combined_edges.shape
    if len(shape) == 4:
        # (T, Z, Y, X)
        combined_edges = gaussian_filter(combined_edges, sigma=(0, 1, 1, 1))
    elif len(shape) == 5:
        #(T, C, Z, Y, X)
        combined_edges = gaussian_filter(combined_edges, sigma=(0, 0, 1, 1, 1))

    tiff.imwrite(str(input_dir / "combined_edges_blurred.tif"), combined_edges.astype(np.float32))

    return combined_edges.astype(np.float32)


def combine_edges_timepoints_byparent(input_dir, output_dir,timepoint=None, sigma=0):
    import os 
    from pathlib import Path
    import numpy as np
    import tifffile as tiff
    from scipy.ndimage import gaussian_filter
    
    # Use os.walk to find all .tif files in directory and subdirectories

    edges_files = []
    if timepoint is None:
        raise ValueError("Please provide a timepoint to filter edges files.")
    
    else: 
        for root, dirs, files in os.walk(input_dir):
            for file in files:
                if file.endswith('.tif') or file.endswith('.tiff'):
                    # Filter by timepoint
                    if f"timepoint_{timepoint:04d}" in file or f"_t{timepoint:04d}" in file:
                        if "_edges" in file:
                            full_path = os.path.join(root, file)
                            edges_files.append(full_path)
    
    edges_files = sorted(edges_files)
    
    if len(edges_files) == 0:
        raise ValueError(f"No edges files found for timepoint {timepoint} in {input_dir} and subdirectories.")
    
    print(f"Found {len(edges_files)} edges files for timepoint {timepoint}.")
    print(edges_files)
    
    # Combine edges files by averaging
    edge_sum = None
    for edge_file in edges_files:
        edge_image = tiff.imread(edge_file).astype(np.float32)
        if edge_sum is None:
            edge_sum = edge_image
        else:
            edge_sum += edge_image
    
    combined_edges = edge_sum / len(edges_files)
    
    # Blur the edges in (Z,Y,X)
    shape = combined_edges.shape
    if len(shape) == 3:
        # (Z, Y, X)
        combined_edges = gaussian_filter(combined_edges, sigma=(sigma, sigma, sigma))
    elif len(shape) == 4:
        # (T, Z, Y, X)
        combined_edges = gaussian_filter(combined_edges, sigma=(0, sigma, sigma, sigma))
    elif len(shape) == 5:
        # (T, C, Z, Y, X)
        combined_edges = gaussian_filter(combined_edges, sigma=(0, 0, sigma, sigma, sigma))
    
    # Save to output_dir
    output_path = Path(output_dir) / f"combined_edges_blur_s{sigma}_t{timepoint:04d}.tif"
    tiff.imwrite(str(output_path), combined_edges.astype(np.float32))
    print(f"Saved combined edges to {output_path}")
    
    return combined_edges.astype(np.float32)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Combine edges files in a directory."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Path to the directory containing edges files.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=False,
        help="Path to the output directory for combined edges file.",
    )
    parser.add_argument(
        "--timepoint",
        type=int,
        required=False,
        help="Timepoint to filter edges files.",
        default=None
    )    
    parser.add_argument(
        "--sigma",
        type=float,
        required=False,
        help="Sigma for Gaussian blur.",
        default=0
    )

    args = parser.parse_args()
    combine_edges_timepoints_byparent(args.input_dir, args.output_dir, timepoint=args.timepoint, sigma=args.sigma)