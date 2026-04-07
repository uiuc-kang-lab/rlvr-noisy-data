#!/bin/bash
#SBATCH --account=daniel_kang
#SBATCH --partition=schmidt_sciences
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=1-00:00:00
#SBATCH --job-name=dapo_clean
#SBATCH --mem=128g

# Check that the GPU is available
nvidia-smi

export WANDB_API_KEY=${WANDB_API_KEY}

bash rl_noise/arc/run_arc.sh \
  --base_dir=/data/daniel_kang_group/rl_noise \
  --run_name=arc_test \
  --model=Qwen/Qwen3-0.6B \
  --debug=True