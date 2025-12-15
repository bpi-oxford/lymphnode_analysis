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

    # Find all foreground and edges files
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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Combine foreground  files in a directory."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Path to the directory containing foreground  files.",
    )

    args = parser.parse_args()
    combine_foreground(args.input_dir)