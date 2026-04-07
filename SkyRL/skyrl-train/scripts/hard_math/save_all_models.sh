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

# sql_noise_qwen2.5_coder_7b_instruct
# sql_clean_qwen2.5_coder_7b_instruct

# module load cuda/12.8

# grpo_qwen2.5-math-7B_sr-pr-noise grpo_qwen2.5-math-7B_sr-format grpo_qwen2.5-math-7B_sr-pr-noise-0.5-dapo grpo_qwen2.5-math-7B_sr-pr-noise-0.5-drgrpo grpo_qwen2.5-math-7B_sr-pr-noise-0.5-sapo grpo_qwen2.5-math-7B_sr-pr-noise-0.5-tis grpo_qwen2.5-math-7B_sr-pr-noise-0.5 deepscaler-grpo_qwen2.5-math-7B_sr-pr-noise-pgfc-0.5 grpo_qwen2.5-math-7B_sr-pr-noise deepscaler-grpo_qwen2.5-math-7B_sr-pr-noise-0.2 grpo_qwen2.5-math-7B_sr-pr-noise-0.5-cispo grpo_qwen2.5-math-7B_sr-pr-noise-cispo grpo_qwen2.5-math-7B_sr-pr-noise-drgrpo grpo_qwen2.5-math-7B_sr-pr-noise-tis deepscaler-grpo_qwen2.5-math-7B_sr-pr-noise-0.3 grpo_qwen2.5-math-7B_sr-pr-noise-dapo deepscaler-grpo_qwen2.5-math-7B_sr-pr-noise-0.4 grpo_qwen2.5-math-7B_sr-random-noise 

for checkpoint_name in deepscaler-grpo_qwen2.5-math-7B_sr-pr-noise-0.1 grpo_qwen2.5-math-7B_sr-pr-noise-sapo
do
    bash rl_noise/math_hard/run_deepscaler_save.sh \
    --model=Qwen/Qwen2.5-Math-7B \
    --base_dir=/data/daniel_kang_group/rl_noise/ \
    --run_name=$checkpoint_name \
    --train_data=deepscaler_train_pr_noise.parquet \
    --format_only=false

    echo "Done saving model for checkpoint: $checkpoint_name"
done