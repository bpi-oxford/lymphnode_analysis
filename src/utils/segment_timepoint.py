import argparse
import numpy as np
import os
import zarr
from pathlib import Path
from ultrack import segment
from ultrack.config import MainConfig

def main():
    parser = argparse.ArgumentParser(
        description="Segment a single timepoint using ultrack with all parameters configurable via CLI"
    )
    
    # Required arguments
    parser.add_argument('--timepoint', type=int, required=True,
                        help='Timepoint index to process')
    parser.add_argument('--foreground_path', type=str, required=True,
                        help='Path to foreground zarr store')
    parser.add_argument('--contours_path', type=str, required=True,
                        help='Path to contours/edges zarr store')
    parser.add_argument('--database_path', type=str, required=True,
                        help='SQLite database path (e.g., sqlite:///path/to/database.db)')
    
    # Segmentation configuration parameters
    parser.add_argument('--min_area', type=int, default=50,
                        help='Minimum area for segments')
    parser.add_argument('--max_area', type=int, default=10000,
                        help='Maximum area for segments')
    parser.add_argument('--min_frontier', type=float, default=0.0,
                        help='Minimum frontier score')
    parser.add_argument('--n_workers', type=int, default=1,
                        help='Number of workers for parallel processing')
    
    # Optional parameters
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite existing database')
    parser.add_argument('--foreground_dataset', type=str, default=None,
                        help='Dataset name in foreground zarr (if not root)')
    parser.add_argument('--contours_dataset', type=str, default=None,
                        help='Dataset name in contours zarr (if not root)')
    
    args = parser.parse_args()
    
    t = args.timepoint
    print(f"Processing timepoint {t}")
    print(f"Foreground: {args.foreground_path}")
    print(f"Contours: {args.contours_path}")
    print(f"Database: {args.database_path}")
    
    # Load data from zarr stores
    foreground_store = zarr.open(args.foreground_path, 'r')
    contours_store = zarr.open(args.contours_path, 'r')
    
    # Access dataset (either root or specified dataset)
    if args.foreground_dataset:
        foreground = foreground_store[args.foreground_dataset]
    else:
        foreground = foreground_store
        
    if args.contours_dataset:
        contours = contours_store[args.contours_dataset]
    else:
        contours = contours_store
    
    # Use FULL arrays - batch_index will handle timepoint selection
    print(f"Full foreground shape: {foreground.shape}, dtype: {foreground.dtype}")
    print(f"Full contours shape: {contours.shape}, dtype: {contours.dtype}")
    
    # Configure ultrack from CLI arguments
    # Parse database path (format: sqlite:///path/to/database.db)
    # Extract the file path from the URI
    if args.database_path.startswith('sqlite:///'):
        db_file_path = args.database_path.replace('sqlite:///', '')
    else:
        db_file_path = args.database_path
    
    # The database path already includes the timepoint directory (from Snakefile)
    # Just use the directory of the database file
    working_dir = os.path.abspath(os.path.dirname(db_file_path))
    os.makedirs(working_dir, exist_ok=True)
    
    print(f"DEBUG: db_file_path = {db_file_path}")
    print(f"DEBUG: working_dir = {working_dir}")
    
    # Initialize config with database settings using dict
    from ultrack.config import DataConfig, SegmentationConfig
    from ultrack.core.database import Base
    from sqlalchemy import create_engine
    
    # Create config with defaults first
    config = MainConfig()
    
    # Set the working directory to where the database will be created
    config.data_config.working_dir = Path(working_dir)
    
    # Set segmentation parameters
    config.segmentation_config.min_area = args.min_area
    config.segmentation_config.max_area = args.max_area
    config.segmentation_config.min_frontier = args.min_frontier
    config.segmentation_config.n_workers = args.n_workers
    
    print(f"Configuration:")
    print(f"  working_dir: {config.data_config.working_dir}")
    print(f"  database_path: {config.data_config.database_path}")
    print(f"  min_area: {config.segmentation_config.min_area}")
    print(f"  max_area: {config.segmentation_config.max_area}")
    print(f"  min_frontier: {config.segmentation_config.min_frontier}")
    print(f"  n_workers: {config.segmentation_config.n_workers}")
    
    # Create database schema before segmentation
    print(f"Initializing database schema...")
    engine = create_engine(str(config.data_config.database_path))
    Base.metadata.create_all(engine)
    engine.dispose()
    print(f"Database schema created")
    
    # Process this single timepoint using batch_index
    # ultrack will automatically assign t={t} for nodes in this batch
    print(f"Running segmentation with batch_index={t}")
    segment(
        foreground,
        contours,
        config,
        batch_index=t,
        overwrite=args.overwrite
    )
    
    print(f"✓ Timepoint {t} segmentation complete")
    
    # Verify the database has correct t value
    import sqlite3
    db_path = os.path.join(working_dir, 'data.db')
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT t FROM nodes")
    t_values = cursor.fetchall()
    cursor.execute("SELECT COUNT(*) FROM nodes")
    node_count = cursor.fetchone()[0]
    cursor.execute("SELECT MIN(id), MAX(id) FROM nodes")
    id_range = cursor.fetchone()
    conn.close()
    
    print(f"Database t values: {[tv[0] for tv in t_values]}")
    print(f"Total nodes: {node_count:,}")
    print(f"Node ID range: {id_range[0]:,} - {id_range[1]:,}")

if __name__ == '__main__':
    main()
