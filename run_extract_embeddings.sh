#!/bin/bash
# Extract Percival visual-tower embeddings from a manifest of CT volumes
# (.pt or .nii / .nii.gz).
#
# Usage:
#   sbatch run_extract_embeddings.sh \
#       <weights.pth> <config.yaml> <input.xlsx> <output_dir>
#
# Add your cluster-specific SBATCH directives below (partition, gpus,
# cpus-per-gpu, mem-per-gpu, time, account, etc.) before submitting.
#
#SBATCH --job-name=percival_extract_embeddings
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.err

set -euo pipefail
mkdir -p slurm_logs

conda activate percival

WEIGHTS=${1:?weights.pth required}
CONFIG=${2:?config.yaml required}
INPUT_XLSX=${3:?input manifest xlsx required}
OUTPUT_DIR=${4:?output dir required}
BATCH_SIZE=${5:-8}

echo "[INFO] weights:     ${WEIGHTS}"
echo "[INFO] config:      ${CONFIG}"
echo "[INFO] input xlsx:  ${INPUT_XLSX}"
echo "[INFO] output dir:  ${OUTPUT_DIR}"
echo "[INFO] batch size:  ${BATCH_SIZE}"

python extract_embeddings.py \
    --weights     "${WEIGHTS}" \
    --config      "${CONFIG}" \
    --input-xlsx  "${INPUT_XLSX}" \
    --output-dir  "${OUTPUT_DIR}" \
    --batch-size  "${BATCH_SIZE}"
