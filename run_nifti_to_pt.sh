#!/bin/bash
# Batch NIfTI -> PyTorch tensor conversion. CPU-only array job.
#
# Workflow:
#   1. Prepare chunks from a manifest (rank 0 only, one-time):
#        python nifti_to_pt/nifti_to_pt.py --prepare --n-chunks 60 \
#            --manifest /path/to/scans_manifest.xlsx \
#            --chunk-dir /path/to/chunks/
#   2. Submit this array job (auto-distributes chunks across tasks):
#        sbatch run_nifti_to_pt.sh
#   3. Summarize results and write final parquet (rank 0 only):
#        python nifti_to_pt/nifti_to_pt.py --summarize \
#            --results-dir /path/to/results/ \
#            --chunk-dir /path/to/chunks/ \
#            --manifest /path/to/scans_manifest.xlsx
#
# Add your cluster-specific SBATCH directives below (partition, mem,
# cpus-per-task, time, --array=1-N matching your --n-chunks N, etc.)
# before submitting.
#
#SBATCH --job-name=nifti_to_pt
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.err

set -euo pipefail
mkdir -p slurm_logs

conda activate percival

# Paths — edit for your environment
CHUNK_DIR=${CHUNK_DIR:-/path/to/chunks}
SRC_ROOT=${SRC_ROOT:-/path/to/source/nifti/root}
DST_ROOT=${DST_ROOT:-/path/to/dest/pt/root}
RESULTS_DIR=${RESULTS_DIR:-/path/to/results}

CHUNK_PATH="${CHUNK_DIR}/chunk_${SLURM_ARRAY_TASK_ID}.csv"

if [ ! -f "${CHUNK_PATH}" ]; then
  echo "ERROR: missing chunk file ${CHUNK_PATH}" >&2
  echo "       Did you run --prepare first?" >&2
  exit 1
fi

echo "[INFO] processing chunk ${SLURM_ARRAY_TASK_ID}: ${CHUNK_PATH}"

python nifti_to_pt/nifti_to_pt.py \
    --process \
    --chunk-path  "${CHUNK_PATH}" \
    --chunk-id    "${SLURM_ARRAY_TASK_ID}" \
    --src-root    "${SRC_ROOT}" \
    --dst-root    "${DST_ROOT}" \
    --results-dir "${RESULTS_DIR}"
