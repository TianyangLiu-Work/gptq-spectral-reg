#!/bin/bash
#SBATCH -p gpu
#SBATCH -J phase4_7b
#SBATCH --gres=gpushare:1
#SBATCH --time=08:00:00
#SBATCH -o experiments/phase4/run.log
#SBATCH -e experiments/phase4/run.err
# No MPS export — let Slurm handle GPU isolation

echo "START $(date) JobID=$SLURM_JOB_ID"
echo "PWD=$(pwd)"

export CUDA_VISIBLE_DEVICES=0

cd /home/tyliu/gptq-spectral-reg
export PYTHONPATH=$PWD:/home/tyliu/quantization:

# Run directly — Python should inherit SLURM_JOB_ID env var
/home/tyliu/quantization/.conda/bin/python -u scripts/run_phase4.py

RC=$?
echo "EXIT_CODE=$RC"
echo "DONE $(date)"
