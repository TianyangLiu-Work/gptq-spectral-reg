#!/bin/bash
#SBATCH -p gpu
#SBATCH -J phase4_7b
#SBATCH --gres=gpushare:1
#SBATCH --time=08:00:00
#SBATCH -o experiments/phase4/run.log
#SBATCH --export=ALL,CUDA_MPS_PIPE_DIRECTORY=/var/run/nvidia-mps,CUDA_MPS_LOG_DIRECTORY=/var/log/nvidia-mps,CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25

echo "START Sun Apr 26 04:07:54 PM CDT 2026 JobID="
echo "PWD=/home/tyliu/quantization2"

export PYTHONPATH=/home/tyliu/quantization2:/home/tyliu/quantization:
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Pass MPS env to srun explicitly
srun --gres=gpushare:1   --export=ALL,CUDA_MPS_PIPE_DIRECTORY=/var/run/nvidia-mps,CUDA_MPS_LOG_DIRECTORY=/var/log/nvidia-mps,CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25   /home/tyliu/quantization/.conda/bin/python -u run_phase4.py

RC=0
echo "EXIT_CODE="
echo "DONE Sun Apr 26 04:07:54 PM CDT 2026"
