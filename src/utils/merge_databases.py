import sqlalchemy as sqla
from sqlalchemy.orm import Session
from pathlib import Path
from tqdm import tqdm
from ultrack.core.database import Base, NodeDB, OverlapDB
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import tempfile
import os

def _merge_batch(args):
    """Merge a batch of timepoints into an intermediate database (parallel worker)"""
    database_dir, timepoint_batch, intermediate_db_path = args
    
    engine_out = sqla.create_engine(f"sqlite:///{intermediate_db_path}")
    Base.metadata.create_all(engine_out)
    
    batch_nodes = 0
    batch_overlaps = 0
    
    for t in timepoint_batch:
        db_file_subdir = Path(database_dir) / f"timepoint_{t:04d}" / "data.db"
        db_file_flat = Path(database_dir) / f"timepoint_{t:04d}.db"
        
        if db_file_subdir.exists():
            db_file = db_file_subdir
        elif db_file_flat.exists():
            db_file = db_file_flat
        else:
            continue
        
        engine_in = sqla.create_engine(f"sqlite:///{db_file}")
        
        with Session(engine_in) as session_in, Session(engine_out) as session_out:
            nodes = session_in.query(NodeDB).all()
            overlaps = session_in.query(OverlapDB).all()
            
            batch_nodes += len(nodes)
            batch_overlaps += len(overlaps)
            
            for node in nodes:
                session_out.merge(node)
            for overlap in overlaps:
                session_out.merge(overlap)
            
            session_out.commit()
        
        engine_in.dispose()
    
    engine_out.dispose()
    
    return intermediate_db_path, batch_nodes, batch_overlaps

def merge_timepoint_databases(database_dir, output_db, n_timepoints=80, n_workers=None, batch_size=10):
    """
    Merge all timepoint databases into one final database using parallel processing.
    
    Args:
        database_dir: Directory containing timepoint_XXXX subdirectories
        output_db: Path to output merged database
        n_timepoints: Number of timepoints to merge
        n_workers: Number of parallel workers (default: CPU count)
        batch_size: Number of timepoints to process per batch (default: 10)
    """
    
    if n_workers is None:
        n_workers = max(1, mp.cpu_count() - 1)
    
    print(f"Creating output database: {output_db}")
    print(f"Using {n_workers} parallel workers with batch size {batch_size}")
    
    # Create batches of timepoints
    timepoint_batches = [list(range(i, min(i + batch_size, n_timepoints))) 
                        for i in range(0, n_timepoints, batch_size)]
    
    # Create temporary directory for intermediate databases
    temp_dir = tempfile.mkdtemp(prefix="ultrack_merge_")
    intermediate_dbs = []
    
    try:
        # Phase 1: Parallel merging into intermediate databases
        print(f"\n[Phase 1/2] Parallel merging into {len(timepoint_batches)} intermediate databases...")
        
        batch_args = []
        for i, batch in enumerate(timepoint_batches):
            intermediate_db = os.path.join(temp_dir, f"intermediate_{i:03d}.db")
            intermediate_dbs.append(intermediate_db)
            batch_args.append((database_dir, batch, intermediate_db))
        
        total_nodes = 0
        total_overlaps = 0
        
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(_merge_batch, args): args for args in batch_args}
            
            with tqdm(total=len(futures), desc="Processing batches") as pbar:
                for future in as_completed(futures):
                    db_path, nodes, overlaps = future.result()
                    total_nodes += nodes
                    total_overlaps += overlaps
                    pbar.update(1)
        
        # Phase 2: Sequential merge of intermediate databases into final output
        print(f"\n[Phase 2/2] Merging {len(intermediate_dbs)} intermediate databases into final output...")
        
        engine_out = sqla.create_engine(f"sqlite:///{output_db}")
        Base.metadata.create_all(engine_out)
        
        for intermediate_db in tqdm(intermediate_dbs, desc="Merging intermediates"):
            if not os.path.exists(intermediate_db):
                continue
                
            engine_in = sqla.create_engine(f"sqlite:///{intermediate_db}")
            
            with Session(engine_in) as session_in, Session(engine_out) as session_out:
                nodes = session_in.query(NodeDB).all()
                overlaps = session_in.query(OverlapDB).all()
                
                for node in nodes:
                    session_out.merge(node)
                for overlap in overlaps:
                    session_out.merge(overlap)
                
                session_out.commit()
            
            engine_in.dispose()
        
        engine_out.dispose()
        
        print(f"\n✓ Merge complete!")
        print(f"  Total nodes: {total_nodes:,}")
        print(f"  Total overlaps: {total_overlaps:,}")
        print(f"  Output: {output_db}")
        
    finally:
        # Cleanup temporary files
        print("\nCleaning up temporary files...")
        for db in intermediate_dbs:
            if os.path.exists(db):
                os.remove(db)
        try:
            os.rmdir(temp_dir)
        except:
            pass

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Merge individual timepoint databases into one final database (parallelized)"
    )
    parser.add_argument(
        '--database_dir',
        type=str,
        required=True,
        help='Directory containing timepoint_XXXX subdirectories with data.db files'
    )
    parser.add_argument(
        '--output_db',
        type=str,
        required=True,
        help='Path to output merged database file'
    )
    parser.add_argument(
        '--n_timepoints',
        type=int,
        required=True,
        help='Number of timepoints to merge'
    )
    parser.add_argument(
        '--n_workers',
        type=int,
        default=None,
        help='Number of parallel workers (default: CPU count - 1)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=10,
        help='Number of timepoints per batch (default: 10)'
    )
    
    args = parser.parse_args()
    
    merge_timepoint_databases(
        database_dir=args.database_dir,
        output_db=args.output_db,
        n_timepoints=args.n_timepoints,
        n_workers=args.n_workers,
        batch_size=args.batch_size
    )
