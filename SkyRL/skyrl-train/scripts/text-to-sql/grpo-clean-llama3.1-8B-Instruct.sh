#!/bin/bash
#SBATCH --account=daniel_kang
#SBATCH --partition=schmidt_sciences
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=32
#SBATCH --time=1-00:00:00
#SBATCH --job-name=llama-clean
#SBATCH --mem=128g

# Check that the GPU is available
nvidia-smi

export WANDB_API_KEY=${WANDB_API_KEY}
export HF_HUB_CACHE=/data/daniel_kang_group/huggingface/hub
export HF_HOME=/data/daniel_kang_group/huggingface
export HF_TOKEN=${HF_TOKEN}

bash rl_noise/text-to-sql/grpo_clean.sh --base_dir=/data/daniel_kang_group/rl_noise/ --model=meta-llama/Llama-3.1-8B-Instruct --run_name=GRPO_Llama-3.1-8B-Instruct_Platinum
