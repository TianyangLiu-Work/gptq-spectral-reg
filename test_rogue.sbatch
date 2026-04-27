#!/bin/bash
#SBATCH -p gpu
#SBATCH -J test_rogue
#SBATCH --gres=gpushare:1
#SBATCH --time=00:05:00
#SBATCH -o ~/quantization2/experiments/test_rogue.out
#SBATCH --export=ALL,CUDA_MPS_PIPE_DIRECTORY=/var/run/nvidia-mps,CUDA_MPS_LOG_DIRECTORY=/var/log/nvidia-mps,CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25

echo "START $(date) JobID=$SLURM_JOB_ID"

export CUDA_VISIBLE_DEVICES=0
/home/tyliu/quantization/.conda/bin/python -c "
import torch, time
print(\"CUDA available:\", torch.cuda.is_available())
t = torch.zeros(5000, 5000, device=\"cuda\")
print(\"Allocated ~100MB on GPU\")
for i in range(10):
    print(f\"Alive at {i*10}s...\", flush=True)
    time.sleep(10)
print(\"Test passed!\")
"

echo "EXIT_CODE=$?"
echo "DONE $(date)"
