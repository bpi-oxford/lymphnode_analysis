import tifffile
import zarr
import numpy as np
from pathlib import Path
from natsort import natsorted

def tif_dir_to_zarr(tif_dir_path: str, zarr_path: str):
    tif_dir = Path(tif_dir_path)          # folder with TIFFs
    out_store = zarr.DirectoryStore(zarr_path)
    root = zarr.group(store=out_store, overwrite=True)

    files = natsorted(tif_dir.glob("*.tif"))
    sample = tifffile.imread(files[0])
    T = len(files)

    if sample.ndim ==3:
        Z , Y , X = sample.shape
        shape = (T, Z, Y, X)
        chunks = (1, Z , 512, 512)

    elif sample.ndim == 4 and sample.shape[0]==1:
        print("Removing singleton channel dimension")
        Z , Y , X = sample.shape[1:]
        shape = (T, Z, Y, X)
        chunks = (1, Z , 512, 512)
    else:
        print(sample.shape)
        raise ValueError("Unexpected TIFF shape")

    ds = root.create_dataset(
        "labels",
        shape=shape,
        dtype=np.uint32,
        chunks=chunks,
        compressor=None,   # <-- no compression
        overwrite=True,
    )

    for t, f in enumerate(files):
        data = tifffile.imread(f).astype(np.uint32)
        ds[t] = data
    
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

    args = parser.parse_args()
    tif_dir_to_zarr(args.tif_dir, args.zarr_path)