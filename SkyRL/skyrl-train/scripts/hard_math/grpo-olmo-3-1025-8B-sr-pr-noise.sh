#!/bin/bash
#SBATCH --account=daniel_kang
#SBATCH --partition=schmidt_sciences
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --time=1-00:00:00
#SBATCH --job-name=o-m-pr
#SBATCH --cpus-per-task=32
#SBATCH --mem=512G

# Check that the GPU is available

export WANDB_API_KEY=${WANDB_API_KEY}
export HF_HUB_CACHE=/data/daniel_kang_group/huggingface/hub
export HF_HOME=/data/daniel_kang_group/huggingface
export HF_TOKEN=${HF_TOKEN}
export UV_CACHE_DIR=/data/yuxuan_zhu/.cache/uv
export PATH=/data/daniel_kang_group/.local/bin:$PATH

module load cuda/12.8


bash rl_noise/math_hard/run_deepscaler.sh \
  --model=allenai/Olmo-3-7B-Instruct-SFT \
  --base_dir=/data/daniel_kang_group/rl_noise/ \
  --run_name=grpo_olmo-3-8B-SFT_sr-pr-noise \
  --train_data=deepscaler_train_pr_noise.parquet \
  --disable_gradient_checkpointing
