#!/bin/bash
#SBATCH -p gpu
#SBATCH -J test_env
#SBATCH --gres=gpushare:1
#SBATCH --time=00:05:00
#SBATCH -o experiments/test_env.out
echo "START"
export CUDA_VISIBLE_DEVICES=0
/home/tyliu/quantization/.conda/bin/python -u /tmp/check_env.py
echo "EXIT=$?"
echo "DONE"
