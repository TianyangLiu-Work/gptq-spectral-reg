#!/bin/bash
#SBATCH -p gpu
#SBATCH -J phase5_7b_sweep
#SBATCH --gres=gpushare:1
#SBATCH --time=24:00:00
#SBATCH -o experiments/phase5/run.log
#SBATCH -e experiments/phase5/run.err

echo "START $(date) JobID=$SLURM_JOB_ID"
echo "PWD=$(pwd)"

export CUDA_VISIBLE_DEVICES=0

cd /home/tyliu/gptq-spectral-reg
export PYTHONPATH=$PWD:/home/tyliu/quantization:

/home/tyliu/quantization/.conda/bin/python -u scripts/run_phase5.py

RC=$?
echo "EXIT_CODE=$RC"
echo "DONE $(date)"
