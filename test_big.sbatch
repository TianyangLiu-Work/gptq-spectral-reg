#!/bin/bash
#SBATCH -p gpu
#SBATCH -J test_big
#SBATCH --gres=gpushare:1
#SBATCH --time=00:10:00
#SBATCH -o experiments/test_big.out
echo "START"
export CUDA_VISIBLE_DEVICES=0
/home/tyliu/quantization/.conda/bin/python -u /tmp/big_gpu_test.py
echo "EXIT=$?"
echo "DONE"
