#!/bin/bash
#SBATCH -p gpu
#SBATCH -J phase4_7b
#SBATCH --gres=gpushare:1
#SBATCH --time=08:00:00
#SBATCH -o experiments/phase4/run.log
#SBATCH -e experiments/phase4/run.err
# No MPS export — let Slurm handle GPU isolation

echo "START Sun Apr 26 04:18:38 PM CDT 2026 JobID="
echo "PWD=/home/tyliu/quantization2"

export PYTHONPATH=/home/tyliu/quantization2:/home/tyliu/quantization:
export CUDA_VISIBLE_DEVICES=0

# Run directly — Python should inherit SLURM_JOB_ID env var
/home/tyliu/quantization/.conda/bin/python -u run_phase4.py

RC=0
echo "EXIT_CODE="
echo "DONE Sun Apr 26 04:18:38 PM CDT 2026"
