#!/bin/bash
#SBATCH --account=daniel_kang
#SBATCH --partition=schmidt_sciences
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --time=1-00:00:00
#SBATCH --job-name=eval
#SBATCH --cpus-per-task=32
#SBATCH --mem=512G

# Check that the GPU is available

export WANDB_API_KEY=${WANDB_API_KEY}
export HF_HUB_CACHE=/data/daniel_kang_group/huggingface/hub
export HF_HOME=/data/daniel_kang_group/huggingface
export HF_TOKEN=${HF_TOKEN}
export UV_CACHE_DIR=/data/daniel_kang_group/uv
export PATH=/data/daniel_kang_group/.local/bin:$PATH
export TMPDIR=/data/daniel_kang_group/tmp

module load cuda/12.8

checkpoint_name=$1

bash rl_noise/math_hard/run_deepscaler_eval.sh \
--model=/data/daniel_kang_group/rl_noise//exports/deepscaler_grpo_qwen2.5-math-7B_sr-clean/global_step_810/policy \
--base_dir=/data/daniel_kang_group/rl_noise/ \
--run_name=$checkpoint_name \
--train_data=deepscaler_train_pr_noise.parquet \
--format_only=false
