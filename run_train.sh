echo "NODELIST=${SLURM_NODELIST}"
echo "CPUS_ON_NODE=${SLURM_CPUS_ON_NODE}"

export PYTORCH_ALLOC_CONF=expandable_segments:True
export NCCL_DEBUG=WARN
export TORCH_DISTRIBUTED_DEBUG=DETAIL



NUM_PROCS=2
NUM_MACHINES=1
CPU_PER_GPU=12


echo "Launching with Accelerate..."
accelerate launch \
  --num_processes "${NUM_PROCS}" \
  --num_machines "${NUM_MACHINES}" \
  --mixed_precision fp16 \
  --num_cpu_threads_per_process "${CPU_PER_GPU}" \
  train.py

