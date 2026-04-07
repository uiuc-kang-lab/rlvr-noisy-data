#!/bin/bash
#SBATCH --account=daniel_kang
#SBATCH --partition=schmidt_sciences
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --time=1-00:00:00
#SBATCH --job-name=m-pr
#SBATCH --cpus-per-task=32
#SBATCH --mem=512G

# Check that the GPU is available
nvidia-smi

export WANDB_API_KEY=${WANDB_API_KEY}
export HF_HUB_CACHE=/data/daniel_kang_group/huggingface/hub
export HF_HOME=/data/daniel_kang_group/huggingface
export HF_TOKEN=${HF_TOKEN}
export UV_CACHE_DIR=/data/yuxuan_zhu/.cache/uv
export PATH=/data/daniel_kang_group/.local/bin:$PATH
export TMPDIR=/data/daniel_kang_group/tmp

noise=$1

bash rl_noise/math_hard/run_deepscaler.sh \
  --model=Qwen/Qwen2.5-Math-7B \
  --base_dir=/data/daniel_kang_group/rl_noise/ \
  --run_name=deepscaler-grpo_qwen2.5-math-7B_sr-pr-noise-pgfc-$noise \
  --train_data=deepscaler_train_pr-noise-$noise.parquet \
  --format_only=false \
  --pgfc_noise_rate=$noise
