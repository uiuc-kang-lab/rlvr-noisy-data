#!/bin/bash
#SBATCH --account=daniel_kang
#SBATCH --partition=schmidt_sciences
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --time=1-00:00:00
#SBATCH --job-name=dapo_noise
#SBATCH --mem=128g

# Check that the GPU is available
nvidia-smi

export WANDB_API_KEY=${WANDB_API_KEY}

bash examples/text_to_sql/run_noisy-rl-sql-dapo.sh
