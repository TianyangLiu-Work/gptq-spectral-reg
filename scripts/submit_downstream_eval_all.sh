#!/bin/bash
# Submit downstream eval jobs for all saved models.
set -e

MODELS_DIR="/mnt/host-share/gptq-models"
SCRIPT="scripts/eval_saved_model.py"
TASKS="lambada_openai,hellaswag,arc_challenge,winogrande"
LIMIT=200

# Only submit for models that exist and don't have results yet
for d in "$MODELS_DIR"/opt-6.7b-*/; do
    name=$(basename "$d")
    result_file="experiments/downstream/${name}.json"

    # Skip if already evaluated
    if [ -f "$result_file" ]; then
        echo "  Skipping $name (already evaluated)"
        continue
    fi

    # Skip if this model doesn't have safetensors yet (still running)
    if [ ! -f "$d/model.safetensors" ]; then
        echo "  Skipping $name (no model.safetensors yet)"
        continue
    fi

    logdir="/mnt/host-share/gptq-models/logs"
    mkdir -p "$logdir"
    job_name="eval_${name}"

    sbatch <<EOF
#!/bin/bash
#SBATCH -J $job_name
#SBATCH -o $logdir/${job_name}.log
#SBATCH -e $logdir/${job_name}.err
#SBATCH -p gpu
#SBATCH -n 1
#SBATCH -N 1
#SBATCH --gres=gpushare:1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 8:00:00

echo "START \$(date) JobID=\$SLURM_JOB_ID"
echo "Eval: $name  Tasks: $TASKS  Limit: $LIMIT"

source /data/conda/bin/activate /home/tyliu/quantization/.conda
cd /home/tyliu/gptq-spectral-reg
python $SCRIPT --model_dir $d --tasks "$TASKS" --limit $LIMIT

echo "END \$(date)"
EOF

    echo "  Submitted: $job_name"
done

echo ""
echo "All eval jobs submitted!"
