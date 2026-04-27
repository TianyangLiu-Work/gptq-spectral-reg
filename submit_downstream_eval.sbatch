#!/bin/bash
#SBATCH -p gpu
#SBATCH -J eval_downstream
#SBATCH --gres=gpushare:1
#SBATCH --time=12:00:00
#SBATCH -o experiments/phase4/downstream.log
#SBATCH -e experiments/phase4/downstream.err

echo "START $(date) JobID=$SLURM_JOB_ID"
echo "PWD=$(pwd)"

export PYTHONPATH=/home/tyliu/quantization2:/home/tyliu/quantization:
export CUDA_VISIBLE_DEVICES=0

/home/tyliu/quantization/.conda/bin/python -u eval_downstream.py

RC=$?
echo "EXIT_CODE=$RC"
echo "DONE $(date)"
