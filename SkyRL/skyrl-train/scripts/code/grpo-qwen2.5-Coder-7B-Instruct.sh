#!/bin/bash
#SBATCH --account=daniel_kang
#SBATCH --partition=schmidt_sciences
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --time=1-00:00:00
#SBATCH --job-name=qwen2.5-code
#SBATCH --cpus-per-task=32

# Check that the GPU is available
nvidia-smi

export WANDB_API_KEY=${WANDB_API_KEY}

bash rl_noise/code/run_lcb_long_length.sh --model=Qwen/Qwen2.5-Coder-7B-Instruct --noise_level=0 --base_dir=/data/daniel_kang_group/rl_noise/ --run_name=grpo_qwen2.5-Coder-7B-16k_noise-0