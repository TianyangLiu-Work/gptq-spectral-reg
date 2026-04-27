#!/bin/bash
#SBATCH -p gpu
#SBATCH -J eval_downstream
#SBATCH --gres=gpushare:1
#SBATCH --time=12:00:00
#SBATCH -o experiments/phase4/downstream.log
#SBATCH -e experiments/phase4/downstream.err

echo "START $(date) JobID=$SLURM_JOB_ID"
echo "PWD=$(pwd)"

export CUDA_VISIBLE_DEVICES=0

cd /home/tyliu/gptq-spectral-reg
export PYTHONPATH=$PWD:/home/tyliu/quantization:

/home/tyliu/quantization/.conda/bin/python -u scripts/eval_downstream.py

RC=$?
echo "EXIT_CODE=$RC"
echo "DONE $(date)"
