#!/bin/bash
#SBATCH --job-name=percival_train
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.err



module load cuda/12.9
module load cudnn/8.9.7.29-12

conda activate percival


# Safer CUDA allocator (correct casing) + mild NCCL logs
export PYTORCH_ALLOC_CONF=expandable_segments:True
export NCCL_DEBUG=WARN
export TORCH_DISTRIBUTED_DEBUG=DETAIL
# Optional: set a shared cache root to avoid re-downloading HF / torch
# artifacts on every node. Uncomment + point at your own writable path:
# CACHE_ROOT=/path/to/shared/cache
# export HF_HOME="$CACHE_ROOT/hf"
# export TORCH_HOME="$CACHE_ROOT/torch"
# export TRITON_CACHE_DIR="$CACHE_ROOT/triton"

GPU_COUNT=4
NUM_PROCS=4
NUM_MACHINES=1
CPU_PER_GPU=8




echo "---- SLURM ENV ----"
echo "SLURM_NODELIST=${SLURM_NODELIST}"
echo "SLURM_NNODES=${SLURM_NNODES:-unset}"
echo "SLURM_GPUS=${SLURM_GPUS:-unset}"
echo "SLURM_CPUS_PER_GPU=${SLURM_CPUS_PER_GPU:-unset}"
echo "SLURM_CPUS_ON_NODE=${SLURM_CPUS_ON_NODE:-unset}"
echo "--------------------"

echo "---- DERIVED CONFIG FOR ACCELERATE ----"
echo "GPU_COUNT=${GPU_COUNT}"
echo "NUM_PROCS=${NUM_PROCS}"
echo "NUM_MACHINES=${NUM_MACHINES}"
echo "CPU_PER_GPU=${CPU_PER_GPU}"
echo "---------------------------------------"

cd "$(dirname "$0")"

CONFIG=${1:-configs/augreg_base_v0.yaml}
echo "[INFO] training with config: ${CONFIG}"

echo "Launching with Accelerate..."
echo "---- GPU PRE-FLIGHT ----"
nvidia-smi --query-gpu=index,memory.total,memory.used,memory.free --format=csv,noheader,nounits
nvidia-smi --query-compute-apps=gpu_bus_id,pid,process_name,used_memory --format=csv
echo "------------------------"
accelerate launch \
  --num_processes "${NUM_PROCS}" \
  --num_machines "${NUM_MACHINES}" \
  --mixed_precision fp16 \
  --num_cpu_threads_per_process "${CPU_PER_GPU}" \
  train.py --config "${CONFIG}"
