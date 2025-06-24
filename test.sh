#!/bin/bash
#SBATCH --job-name=inference_q6
#SBATCH --output=slurm_logs/inference_q7_%j.out
#SBATCH --error=slurm_logs/inference_q7_%j.err

#SBATCH --partition=ai
#SBATCH --nodes=1
#SBATCH --gpus-per-node=a100:1
#SBATCH --gpus=a100:1
#SBATCH --cpus-per-gpu=32
#SBATCH --mem-per-gpu=120G
#SBATCH --time=48:00:00



echo "NODELIST="${SLURM_NODELIST}
echo "NODELIST="${SLURM_CPUS_ON_NODE}


module load cuda/11.8
module load cudnn/8.2.1
source ~/miniconda3/etc/profile.d/conda.sh
conda activate percival

conda env list



srun python /cbica/home/beechec/public_code/percival/inference.py