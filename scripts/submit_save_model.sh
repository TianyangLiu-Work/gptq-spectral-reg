#!/bin/bash
# Submit one Slurm job per quantization config.
# Each job: loads model → builds Hessians → quantizes → saves to /mnt/host-share/gptq-models/
set -e

SCRIPT="scripts/save_quantized_models.py"

submit() {
  local label="$1" method="$2" beta="$3" gs="$4"
  local name="save_g${gs}_${label}"
  local log="/mnt/host-share/gptq-models/logs/${name}.log"
  local err="/mnt/host-share/gptq-models/logs/${name}.err"
  mkdir -p "/mnt/host-share/gptq-models/logs"

  if [ "$label" = "FP16" ]; then
    sbatch <<EOF
#!/bin/bash
#SBATCH -J $name
#SBATCH -o $log
#SBATCH -e $err
#SBATCH -p gpu
#SBATCH -n 1
#SBATCH -N 1
#SBATCH --gres=gpushare:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 4:00:00

echo "START \$(date) JobID=\$SLURM_JOB_ID"
echo "Config: $label"

source /data/conda/bin/activate /home/tyliu/quantization/.conda
cd /home/tyliu/gptq-spectral-reg
python $SCRIPT --label $label

echo "END \$(date)"
EOF
  else
    sbatch <<EOF
#!/bin/bash
#SBATCH -J $name
#SBATCH -o $log
#SBATCH -e $err
#SBATCH -p gpu
#SBATCH -n 1
#SBATCH -N 1
#SBATCH --gres=gpushare:1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 8:00:00

echo "START \$(date) JobID=\$SLURM_JOB_ID"
echo "Config: $label gs=$gs method=$method beta=$beta"

source /data/conda/bin/activate /home/tyliu/quantization/.conda
cd /home/tyliu/gptq-spectral-reg
python $SCRIPT --label $label --method $method --beta $beta --group_size $gs

echo "END \$(date)"
EOF
  fi

  echo "  Submitted: $name"
}

echo "Submitting all save_model jobs..."
echo ""

# FP16 (original weights, no quantization)
submit "FP16" "none" "0.0" "128"

# g128 methods
submit "GPTQ_base" "none" "0.01" "128"
submit "Damp_0.05" "stronger_damping" "0.05" "128"
submit "Frob_0.001" "frobenius" "0.001" "128"
submit "Spectral_0.01" "spectral" "0.01" "128"

# g32 methods
submit "GPTQ_base" "none" "0.01" "32"
submit "Damp_0.05" "stronger_damping" "0.05" "32"
submit "Frob_0.001" "frobenius" "0.001" "32"
submit "Spectral_0.01" "spectral" "0.01" "32"

echo ""
echo "All 9 jobs submitted! Check with: squeue -u tyliu | watch turm"
