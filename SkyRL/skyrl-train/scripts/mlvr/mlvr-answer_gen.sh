#!/bin/bash
#SBATCH --account=daniel_kang
#SBATCH --partition=schmidt_sciences
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --time=1-00:00:00
#SBATCH --job-name=mlvr-answers
#SBATCH --mem=128g

# Check that the GPU is available
nvidia-smi

export WANDB_API_KEY=${WANDB_API_KEY}

export OPENAI_API_KEY=${OPENAI_API_KEY}

bash rl_noise/mlvr/run_mlvr_gen_answer.sh --debug --base_dir=/data/daniel_kang_group/rl_noise
