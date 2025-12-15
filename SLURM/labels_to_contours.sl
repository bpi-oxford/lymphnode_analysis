#!/bin/bash

#SBATCH --job-name      labels_to_contours
#SBATCH --array         0-7              # Adjust based on number of TIF files
#SBATCH --cpus-per-task 1
##SBATCH --partition=gpu_short                #Select partition. You can run sinfo command to list all partitions
##SBATCH --gpus-per-node=1                    #Number of GPUs. Always starts with 1 ( more GPU, more wait time)               
#SBATCH --mem           48G
#SBATCH --time          00:30:00         #days-minutes-seconds
#SBATCH --output        slogs/labels_to_contours_%A_%a.out
#SBATCH --error         slogs/labels_to_contours_%A_%a.err
#SBATCH --exclude       compg009,compg010,compg011,compg013

module purge
source /well/kir/config/modules.sh
module load Python/3.10.8-GCCcore-12.2.0
source ~/devel/venv/Python-3.10.8-GCCcore-12.2.0/ultrack_env/bin/activate

# ============= Configuration Flags =============
# Set SKIP_LABELS_TO_CONTOURS=1 to skip the labels_to_contours processing
# and only run the post-processing script
SKIP_LABELS_TO_CONTOURS=0

# Input directory containing TIF files
INPUT_DIR="/users/kir-fritzsche/aif490/devel/tissue_analysis/lymphnode_analysis/data2track/b2-2a_2c_pos6-01_deskew_cgt/crop2/segmentation/videos"
raw_image_path="/users/kir-fritzsche/aif490/devel/tissue_analysis/lymphnode_analysis/data2track/b2-2a_2c_pos6-01_deskew_cgt/crop2/b2-2a_2c_pos6-01_crop_C1_t0-65_z50-359_y750-1262_x1000-1512.tiff"
segmentation_path="/users/kir-fritzsche/aif490/devel/tissue_analysis/lymphnode_analysis/data2track/b2-2a_2c_pos6-01_deskew_cgt/crop2/segmentation/videos"   



# Get list of TIF files
TIF_FILES=(${INPUT_DIR}/*.tif)

echo "Found ${#TIF_FILES[@]} TIF files:"
for i in "${!TIF_FILES[@]}"; do
    echo "  [$i] ${TIF_FILES[$i]}"
done
echo ""

# Get the file for this array task
TIF_FILE=${TIF_FILES[$SLURM_ARRAY_TASK_ID]}

echo ${TIF_FILE}

# Check if file exists
if [ ! -f "${TIF_FILE}" ]; then
    echo "Error: File ${TIF_FILE} not found"
    exit 1
fi

echo "Processing: ${TIF_FILE}"
echo "Task ID: ${SLURM_ARRAY_TASK_ID}"


# Run Python script with the TIF file
python /users/kir-fritzsche/aif490/devel/tissue_analysis/lymphnode_analysis/src/utils/batch_labels_to_contours.py "${raw_image_path}" "${TIF_FILE}"

echo "Completed processing: ${TIF_FILE}"
