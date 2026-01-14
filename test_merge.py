import sys
sys.path.insert(0, 'src/utils')
from merge_databases import merge_timepoint_databases

# Test with just 3 timepoints
merge_timepoint_databases(
    database_dir='SLURM/databases_node2',
    output_db='SLURM/databases_node2/test_merged.db',
    n_timepoints=3
)
