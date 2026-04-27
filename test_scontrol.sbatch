#!/bin/bash
#SBATCH -p gpu
#SBATCH -J test_sctl
#SBATCH --gres=gpushare:1
#SBATCH --time=00:05:00
#SBATCH -o experiments/test_scontrol.out
#SBATCH --export=ALL,CUDA_MPS_PIPE_DIRECTORY=/var/run/nvidia-mps,CUDA_MPS_LOG_DIRECTORY=/var/log/nvidia-mps,CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25

echo "START Sun Apr 26 04:08:08 PM CDT 2026 JobID="
echo "scontrol listpids :"
scontrol listpids 
echo "==="
echo "All running jobs:"
squeue -h -t R -o "%i"
echo "==="
echo "My processes:"
pstree -p 2257422 2>/dev/null || ps -ef | grep  | head -10
echo "==="
echo "I will now do some GPU work..."
export CUDA_VISIBLE_DEVICES=0
/home/tyliu/quantization/.conda/bin/python -c "
import torch, os, time
print(f\"Python PID: {os.getpid()}\")
t = torch.zeros(100, device=cuda)
print(f\"GPU allocated OK, sleeping 120s...\", flush=True)
time.sleep(120)
print(\"Done!\")
"
echo "EXIT_CODE=0"
echo "DONE Sun Apr 26 04:08:08 PM CDT 2026"
