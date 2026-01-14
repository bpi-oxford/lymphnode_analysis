import tifffile
import zarr
import numpy as np
from pathlib import Path
from natsort import natsorted

def tif_dir_to_zarr(tif_dir_path: str, zarr_path: str, filter='edges', dtype=np.uint32, chunk_size = 256):
    tif_dir = Path(tif_dir_path)          # folder with TIFFs
    out_store = zarr.DirectoryStore(zarr_path)
    root = zarr.group(store=out_store, overwrite=True)
    print(f"Reading TIFF files from {tif_dir} with filter '{filter}'")

    files = natsorted(tif_dir.glob("*"))

    print(f"Found {files} in directory.")
    
    files = [f for f in files if filter in f.name]
    sample = tifffile.imread(files[0])
    T = len(files)

    if sample.ndim ==3:
        Z , Y , X = sample.shape
        shape = (T, Z, Y, X)
        chunks = (1, Z , chunk_size, chunk_size)

    elif sample.ndim == 4 and sample.shape[0]==1:
        print("Removing singleton channel dimension")
        Z , Y , X = sample.shape[1:]
        shape = (T, Z, Y, X)
        chunks = (1, Z , chunk_size, chunk_size)
    else:
        print(sample.shape)
        raise ValueError("Unexpected TIFF shape")


    print(f"Creating Zarr dataset at {zarr_path} with shape {shape}")
    name = "labels_" + str(filter)
    ds = root.create_dataset(
        name,
        shape=shape,
        dtype=dtype,
        chunks=chunks,
        compressor=None,   # <-- no compression
        overwrite=True,
    )

    for t, f in enumerate(files):
        data = tifffile.imread(f).astype(dtype)
        if data.ndim ==4 and data.shape[0]==1:
            data = data[0]  # remove singleton channel dimension
        ds[t] = data
    print("Zarr dataset creation complete.")
    return



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert a directory of TIFF files to a Zarr store."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Path to the directory containing TIFF files.",
    )
    parser.add_argument(
        "--zarr_path",
        type=str,
        required=True,
        help="Path to the output Zarr store.",
    )
    parser.add_argument(
        "--filter",
        type=str,
        required=True,
        help="Filter string to select specific TIFF files.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="np.uint32",
        help="Data type for the Zarr dataset (e.g., np.uint8, np.uint16, np.uint32).",
    )
    args = parser.parse_args()
    tif_dir_to_zarr(args.input_dir, args.zarr_path, args.filter, np.dtype(args.dtype))