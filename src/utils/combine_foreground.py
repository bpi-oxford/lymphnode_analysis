"""
searches in a directory for all foreground fiels and all edges files

foreground files are combined with a logical OR operation
edges files are combined by averaging their values

"""



def combine_foreground(input_dir):
    import glob
    from pathlib import Path
    import numpy as np
    import tifffile as tiff

    input_dir = Path(input_dir)

    # Find all foreground files
    foreground_files = sorted(glob.glob(str(input_dir / "*_foreground.tif")))

    if len(foreground_files) == 0:
        raise ValueError("No foreground files found in the specified directory.")
    print(f"Found {len(foreground_files)} foreground files.")

    print(foreground_files)
    
    # Initialize combined arrays
    combined_foreground = None

    # Combine foreground files with logical OR
    for fg_file in foreground_files:
        fg_image = tiff.imread(fg_file).astype(bool)
        if combined_foreground is None:
            combined_foreground = fg_image
        else:
            combined_foreground = np.logical_or(combined_foreground, fg_image)



    tiff.imwrite(str(input_dir / "combined_foreground.tif"), combined_foreground.astype(np.uint8))

    return combined_foreground.astype(np.uint8)


def combine_foreground_timepoints_byparent(input_dir, output_dir,timepoint=None, sigma=0):
    import os 
    from pathlib import Path
    import numpy as np
    import tifffile as tiff
    from scipy.ndimage import gaussian_filter
    
    # Use os.walk to find all .tif files in directory and subdirectories

    foreground_files = []
    if timepoint is None:
        raise ValueError("Please provide a timepoint to filter foreground files.")
    
    else: 
        for root, dirs, files in os.walk(input_dir):
            for file in files:
                if file.endswith('.tif') or file.endswith('.tiff'):
                    # Filter by timepoint
                    if f"timepoint_{timepoint:04d}" in file or f"_t{timepoint:04d}" in file:
                        if "_foreground" in file:
                            full_path = os.path.join(root, file)
                            foreground_files.append(full_path)
    
    foreground_files = sorted(foreground_files)
    
    if len(foreground_files) == 0:
        raise ValueError(f"No foreground files found for timepoint {timepoint} in {input_dir} and subdirectories.")
    
    print(f"Found {len(foreground_files)} foreground files for timepoint {timepoint}.")
    print(foreground_files)
    
    # Combine foreground files with logical OR
    combined_foreground = None
    for fg_file in foreground_files:
        fg_image = tiff.imread(fg_file).astype(bool)
        if combined_foreground is None:
            combined_foreground = fg_image
        else:
            combined_foreground = np.logical_or(combined_foreground, fg_image)
    
    # Blur the foreground in (Z,Y,X)
    shape = combined_foreground.shape
    if len(shape) == 3:
        # (Z, Y, X)
        combined_foreground = gaussian_filter(combined_foreground, sigma=(sigma, sigma, sigma))
    elif len(shape) == 4:
        # (T, Z, Y, X)
        combined_foreground = gaussian_filter(combined_foreground, sigma=(0, sigma, sigma, sigma))
    elif len(shape) == 5:
        # (T, C, Z, Y, X)
        combined_foreground = gaussian_filter(combined_foreground, sigma=(0, 0, sigma, sigma, sigma))
    
    # Save to output_dir
    output_path = Path(output_dir) / f"combined_foreground_blur_s{sigma}_t{timepoint:04d}.tif"
    tiff.imwrite(str(output_path), combined_foreground.astype(np.float32))
    print(f"Saved combined foreground to {output_path}")
    
    return combined_foreground.astype(np.uint16)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Combine foreground files in a directory."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Path to the directory containing foreground files.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=False,
        help="Path to the output directory for combined foreground file.",
    )
    parser.add_argument(
        "--timepoint",
        type=int,
        required=False,
        help="Timepoint to filter foreground files.",
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
    combine_foreground_timepoints_byparent(args.input_dir, args.output_dir, timepoint=args.timepoint, sigma=args.sigma)