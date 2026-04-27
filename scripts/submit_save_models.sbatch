#!/bin/bash
#SBATCH -J save_models_nfs
#SBATCH -o /mnt/host-share/gptq-models/save_models.log
#SBATCH -e /mnt/host-share/gptq-models/save_models.err
#SBATCH -p gpu
#SBATCH -n 1
#SBATCH -N 1
#SBATCH --gres=gpushare:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 24:00:00

echo "START $(date) JobID=$SLURM_JOB_ID"
echo "PWD=$(pwd)"
echo "HOST=$(hostname)"

# Conda
source /data/conda/bin/activate /home/tyliu/quantization/.conda
which python
python --version

# Check GPU
nvidia-smi

python scripts/save_quantized_models.py

echo "END $(date)"
