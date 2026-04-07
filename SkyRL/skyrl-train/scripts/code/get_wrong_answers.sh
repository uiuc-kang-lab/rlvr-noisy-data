#!/bin/bash
#SBATCH --account=daniel_kang
#SBATCH --partition=schmidt_sciences
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --time=1-00:00:00
#SBATCH --job-name=code-gen
#SBATCH --cpus-per-task=32

# Check that the GPU is available
nvidia-smi

export WANDB_API_KEY=${WANDB_API_KEY}
export HF_HUB_CACHE=/data/daniel_kang_group/huggingface/hub
export HF_HOME=/data/daniel_kang_group/huggingface
export HF_TOKEN=${HF_TOKEN}
export UV_CACHE_DIR=/data/yuxuan_zhu/.cache/uv
export PATH=/data/daniel_kang_group/.local/bin:$PATH


bash rl_noise/code/get_wrong_answers.sh --model=Qwen/Qwen3-8B --base_dir=/data/daniel_kang_group/rl_noise --run_name=get_answers_3