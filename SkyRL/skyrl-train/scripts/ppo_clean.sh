#!/bin/bash
#SBATCH --account=daniel_kang
#SBATCH --partition=schmidt_sciences
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --time=1-00:00:00
#SBATCH --job-name=ppo_clean
#SBATCH --mem=1024g

# Check that the GPU is available
nvidia-smi

export WANDB_API_KEY=${WANDB_API_KEY}

until bash examples/text_to_sql/run_corrected-rl-sql-ppo.sh; do
    echo "Script crashed with exit code $?.  Respawning.." >&2
    sleep 1
done

echo "Script completed successfully."
