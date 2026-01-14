#!/bin/bash
# Script to submit the segment_timepoint Snakemake workflow to SLURM
# Usage: ./run_segment_timepoint.sh
# Or with dry-run: ./run_segment_timepoint.sh --dry-run

# Create temp directory in project space (avoid full /tmp partition)
export TMPDIR="$PWD/tmp"
mkdir -p "$TMPDIR"

# Load Snakemake module
module load snakemake/8.4.2-foss-2023a

# Run Snakemake with SLURM executor
snakemake \
    --snakefile segment_timepoint_Snakefile \
    --executor slurm \
    --jobs 100 \
    --default-resources slurm_account=kir.prj slurm_partition=short mem_mb=128000 runtime=60 \
    --latency-wait 100 \
    --retries 0 \
    "$@"
