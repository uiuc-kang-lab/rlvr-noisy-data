#!/bin/bash
#SBATCH --account=daniel_kang
#SBATCH --partition=schmidt_sciences
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --time=1-00:00:00
#SBATCH --job-name=mlvr-clean
#SBATCH --cpus-per-task=16

# Check that the GPU is available
nvidia-smi

export WANDB_API_KEY=${WANDB_API_KEY}
export HF_HUB_CACHE=/data/daniel_kang_group/huggingface/hub
export HF_HOME=/data/daniel_kang_group/huggingface
export HF_TOKEN=${HF_TOKEN}
export OPENAI_API_KEY=${OPENAI_API_KEY}

bash rl_noise/mlvr/run_mlvr.sh --model=Qwen/Qwen2.5-7B-Instruct --noise_level=0 --base_dir=/data/daniel_kang_group/rl_noise/ --run_name=grpo_qwen2.5-7B-Instruct_clean