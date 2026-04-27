#!/bin/bash
#SBATCH -p gpu
#SBATCH -J phase3_opt13b
#SBATCH --gres=gpushare:4
#SBATCH --time=04:00:00
#SBATCH -o experiments/phase2/phase3_%j.out
#SBATCH --export=ALL,CUDA_MPS_PIPE_DIRECTORY=/var/run/nvidia-mps,CUDA_MPS_LOG_DIRECTORY=/var/log/nvidia-mps,CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=100

set -euo pipefail
echo "START $(date) JobID=$SLURM_JOB_ID"
echo "PWD=$(pwd)"
echo "GPU share: ${SLURM_GRES:-<unset>}"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

source /data/conda/bin/activate
conda activate /home/tyliu/quantization/.conda
export HF_HOME=/home/tyliu/.cache/huggingface

cd /home/tyliu/gptq-spectral-reg
export PYTHONPATH=$PWD:/home/tyliu/quantization:

echo "Running Phase 3: OPT-1.3B Spectral regularization"
python scripts/run_phase2.py
echo "EXIT_CODE=$?"

echo "DONE $(date)"
