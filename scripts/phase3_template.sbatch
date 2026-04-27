#!/bin/bash
#SBATCH -p gpu
#SBATCH -J TRAIN_NAME
#SBATCH --gres=gpushare:1
#SBATCH --time=24:00:00
#SBATCH -o %x-%j.out
#SBATCH --export=ALL,CUDA_MPS_PIPE_DIRECTORY=/var/run/nvidia-mps,CUDA_MPS_LOG_DIRECTORY=/var/log/nvidia-mps,CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25

## Uncomment to restrict CPU/memory:
## #SBATCH --cpus-per-task=8
## #SBATCH --mem=32G

## For exclusive GPU access, replace --gres line above with:
## #SBATCH --gres=gpushare:4
## and set CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=100

set -euo pipefail
echo "START $(date) JobID=$SLURM_JOB_ID user=$(whoami) host=$(hostname)"
echo "PWD=$(pwd)"
echo "GPU share: gpushare:${SLURM_GRES:-<unset>}"
echo "MPS thread%: ${CUDA_MPS_ACTIVE_THREAD_PERCENTAGE:-<unset>}"

# --- 用户自行加载环境（示例）---
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate bioemu

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

# --- 运行命令（用户修改）---
# python train.py --config configs/a.yaml

echo "DONE $(date)"
