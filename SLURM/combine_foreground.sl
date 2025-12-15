#!/bin/bash

#SBATCH --job-name      combine_foreground_edges
#SBATCH --cpus-per-task 1
##SBATCH --partition=gpu_short                #Select partition. You can run sinfo command to list all partitions
##SBATCH --gpus-per-node=1                    #Number of GPUs. Always starts with 1 ( more GPU, more wait time)               
#SBATCH --mem           128G
#SBATCH --time          00:00:00         #days-minutes-seconds
#SBATCH --output        slogs/combine_foreground.%j.out
#SBATCH --error         slogs/combine_foreground.%j.err
#SBATCH --exclude       compg009,compg010,compg011,compg013


module purge
source /well/kir/config/modules.sh
module load Python/3.10.8-GCCcore-12.2.0
source ~/devel/venv/Python-3.10.8-GCCcore-12.2.0/ultrack_env/bin/activate


python3 /users/kir-fritzsche/aif490/devel/tissue_analysis/lymphnode_analysis/src/utils/combine_foreground.py \
    --input_dir /users/kir-fritzsche/aif490/devel/tissue_analysis/lymphnode_analysis/data2track/b2-2a_2c_pos6-01_deskew_cgt/crop2/segmentation/videos/ultrack

