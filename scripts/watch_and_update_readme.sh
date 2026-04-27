#!/bin/bash
# Watch Slurm queue + eval results, auto-update README.md when all done.
# Run this on the server.

set -e

PROJECT_DIR="$HOME/gptq-spectral-reg"
README="$PROJECT_DIR/README.md"
EVAL_DIR="$PROJECT_DIR/experiments/downstream"
SCRIPT_DIR="$PROJECT_DIR/scripts"

# How often to check (seconds)
INTERVAL=60

echo "[$(date)] Watch started — monitoring Slurm queue and eval results..."

while true; do
    # Get current queue
    QUEUE=$(squeue -u tyliu 2>/dev/null | grep -v "JOBID" | grep -v "^$" || true)
    
    if [ -z "$QUEUE" ]; then
        echo "[$(date)] Queue EMPTY — checking eval results..."
        
        # Collect all results
        RESULTS_DIR="$EVAL_DIR"
        if compgen -G "$RESULTS_DIR/opt-*.json" > /dev/null 2>&1; then
            echo "[$(date)] Found eval results! Updating README..."
            
            # Run update script
            cd "$PROJECT_DIR"
            python "$SCRIPT_DIR/update_readme_results.py"
            
            echo "[$(date)] README updated! Watch exiting."
            exit 0
        fi
        
        echo "[$(date)] No eval results yet, but queue is empty. Will keep checking..."
        sleep $INTERVAL
        continue
    fi
    
    # Print queue summary
    RUNNING=$(echo "$QUEUE" | grep -c " R " || true)
    PENDING=$(echo "$QUEUE" | grep -c " PD " || true)
    echo "[$(date)] Queue: $RUNNING running, $PENDING pending"
    echo "$QUEUE" | awk '{print "  " $2 " (" $1 "): " $4}'
    
    sleep $INTERVAL
done
