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

model_names=(
    "uiuc-kang-lab/Qwen2.5-Math-7B-GRPO-noise-0.2-epoch-3"
    "uiuc-kang-lab/Qwen2.5-Math-7B-GRPO-noise-0.3-epoch-3"
    "uiuc-kang-lab/Qwen2.5-Math-7B-GRPO-noise-0.4-epoch-3"
    "uiuc-kang-lab/Qwen2.5-Math-7B-PGFC-noise-0.5-epoch-3"
    "uiuc-kang-lab/Qwen2.5-Math-7B-GRPO-format-epoch-3"
    "uiuc-kang-lab/Qwen2.5-Math-7B-DAPO-noise-0.5-epoch-3"
    "uiuc-kang-lab/Qwen2.5-Math-7B-DrGRPO-noise-0.5-epoch-3"
    "uiuc-kang-lab/Qwen2.5-Math-7B-SAPO-noise-0.5-epoch-3"
    "uiuc-kang-lab/Qwen2.5-Math-7B-TIS-noise-0.5-epoch-3"
    "uiuc-kang-lab/Qwen2.5-Math-7B-GRPO-noise-0.5-epoch-3"
    "uiuc-kang-lab/Qwen2.5-Math-7B-GRPO-noise-epoch-3"
    "uiuc-kang-lab/Qwen2.5-Math-7B-GRPO-random-epoch-2"
    "uiuc-kang-lab/Qwen2.5-Math-7B-GRPO-noise-0.1-epoch-3"
)

for i in "${!model_names[@]}"; do
    rm -rf /data/daniel_kang_group/huggingface/hub/models--uiuc-kang-lab*
    model_name="${model_names[$i]}"
    model_base_name=$(basename "$model_name")
    bash rl_noise/math_hard/run_deepscaler_eval.sh \
        --model=$model_name \
        --base_dir=/data/daniel_kang_group/rl_noise/ \
        --run_name=$model_base_name \
        --train_data=deepscaler_train_pr_noise.parquet \
        --format_only=false \
        --debug
    echo "Done evaluating model: $model_name"
done