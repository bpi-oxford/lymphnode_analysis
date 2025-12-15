#!/bin/bash

#SBATCH --job-name      labels_to_contours_directory_arr
#SBATCH --cpus-per-task 1
##SBATCH --partition=gpu_short                #Select partition. You can run sinfo command to list all partitions
##SBATCH --gpus-per-node=1                    #Number of GPUs. Always starts with 1 ( more GPU, more wait time)               
#SBATCH --mem           256G
#SBATCH --time          00:30:00         #days-minutes-seconds
#SBATCH --output        slogs/labels_to_contours_directory_arr.%j.out
#SBATCH --error         slogs/labels_to_contours_directory_arr.%j.err
#SBATCH --exclude       compg009,compg010,compg011,compg013


module purge
source /well/kir/config/modules.sh
module load Python/3.10.8-GCCcore-12.2.0
source ~/devel/venv/Python-3.10.8-GCCcore-12.2.0/ultrack_env/bin/activate


File=${SLURM_ARRAY_TASK_ID}

python3 /users/kir-fritzsche/aif490/devel/tissue_analysis/lymphnode_analysis/src/utils/batch_labels_to_contours.py \
    --input_dir /users/kir-fritzsche/aif490/devel/tissue_analysis/lymphnode_analysis/data2track/b2-2a_2c_pos6-01_deskew_cgt/crop2/videos_for_ultrack \
    --output_dir /users/kir-fritzsche/aif490/devel/tissue_analysis/lymphnode_analysis/data2track/b2-2a_2c_pos6-01_deskew_cgt/crop2/videos_for_ultrack/foreground_edges \
    --sigma 1 \
    --raw_image_path /users/kir-fritzsche/aif490/devel/tissue_analysis/lymphnode_analysis/data2track/b2-2a_2c_pos6-01_deskew_cgt/crop2/b2-2a_2c_pos6-01_crop_C1_t0-65_z50-359_y750-1262_x1000-1512.tiff \
    --file_index ${File}
