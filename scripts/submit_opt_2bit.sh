#!/bin/bash
# Submit 2-bit OPT-6.7B quantization jobs
set -e

MODELS_DIR="/mnt/host-share/gptq-models"
SCRIPT="scripts/save_quantized_models.py"
LOG_DIR="$MODELS_DIR/logs"
mkdir -p "$LOG_DIR"

BITS=2
GROUP=128

# Configs: label method beta
declare -A CONFIGS=(
    ["GPTQ_base"]="none 0.01"
    ["Damp_0.05"]="stronger_damping 0.05"
    ["Frob_0.001"]="frobenius 0.001"
    ["Spectral_0.01"]="spectral 0.01"
)

for label in "${!CONFIGS[@]}"; do
    read -r method beta <<< "${CONFIGS[$label]}"

    job_name="save_opt2b_${label}"
    log_file="$LOG_DIR/${job_name}.log"

    # Skip if already saved
    saved="$MODELS_DIR/opt-6.7b-g${GROUP}-${BITS}bit-${label}"
    if [ -f "$saved/quantization_info.json" ]; then
        echo "Skipping $label (already saved)"
        continue
    fi

    sbatch <<EOF
#!/bin/bash
#SBATCH -J $job_name
#SBATCH -o $LOG_DIR/${job_name}.log
#SBATCH -e $LOG_DIR/${job_name}.err
#SBATCH -p gpu
#SBATCH -n 1
#SBATCH -N 1
#SBATCH --gres=gpushare:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 12:00:00

echo "START \$(date) JobID=\$SLURM_JOB_ID"
echo "Config: $label method=$method beta=$beta bits=$BITS group=$GROUP"

source /data/conda/bin/activate /home/tyliu/quantization/.conda
cd /home/tyliu/gptq-spectral-reg
python $SCRIPT --label "$label" --method "$method" --beta "$beta" --group_size $GROUP --bits $BITS

echo "END \$(date)"
EOF

    echo "Submitted: $job_name"
done
