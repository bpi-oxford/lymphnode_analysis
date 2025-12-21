#!/bin/bash
# Simple script to submit the Snakemake workflow to SLURM
# Usage: ./segmentations2edges.sh

# Load Snakemake module
module load snakemake/8.4.2-foss-2023a

# Run Snakemake with SLURM executor
snakemake \
    --snakefile segmentations2edges_Snakefile \
    --executor slurm \
    --jobs 150 \
    --default-resources slurm_account=kir.prj slurm_partition=short mem_mb=128000 runtime=30 \
    --latency-wait 100 \
    --retries 0 \
    "$@"
