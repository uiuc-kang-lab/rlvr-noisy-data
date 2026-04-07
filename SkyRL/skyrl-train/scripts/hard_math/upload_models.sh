#!/bin/bash
#SBATCH --account=daniel_kang
#SBATCH --partition=schmidt_sciences_cpu
#SBATCH --nodes=1
#SBATCH --time=1-00:00:00
#SBATCH --job-name=eval
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G

export WANDB_API_KEY=${WANDB_API_KEY}
export HF_HUB_CACHE=/data/daniel_kang_group/huggingface/hub
export HF_HOME=/data/daniel_kang_group/huggingface
export HF_TOKEN=${HF_TOKEN}
export UV_CACHE_DIR=/data/daniel_kang_group/uv
export PATH=/data/daniel_kang_group/.local/bin:$PATH
export TMPDIR=/data/daniel_kang_group/tmp

model_names=(
    # "uiuc-kang-lab/Qwen2.5-Math-7B-GRPO-clean-epoch-3"
    # "uiuc-kang-lab/Qwen2.5-Math-7B-GRPO-clean-epoch-4"
    # "uiuc-kang-lab/Qwen2.5-Math-7B-GRPO-noise-0.2-epoch-3"
    # "uiuc-kang-lab/Qwen2.5-Math-7B-GRPO-noise-0.3-epoch-3"
    # "uiuc-kang-lab/Qwen2.5-Math-7B-GRPO-noise-0.4-epoch-3"
    # "uiuc-kang-lab/Qwen2.5-Math-7B-PGFC-noise-0.5-epoch-3"
    # "uiuc-kang-lab/Qwen2.5-Math-7B-GRPO-format-epoch-3"
    # "uiuc-kang-lab/Qwen2.5-Math-7B-DAPO-noise-0.5-epoch-3"
    # "uiuc-kang-lab/Qwen2.5-Math-7B-DrGRPO-noise-0.5-epoch-3"
    # "uiuc-kang-lab/Qwen2.5-Math-7B-SAPO-noise-0.5-epoch-3"
    # "uiuc-kang-lab/Qwen2.5-Math-7B-TIS-noise-0.5-epoch-3"
    # "uiuc-kang-lab/Qwen2.5-Math-7B-GRPO-noise-0.5-epoch-3"
    # "uiuc-kang-lab/Qwen2.5-Math-7B-GRPO-noise-epoch-3"
    # "uiuc-kang-lab/Qwen2.5-Math-7B-GRPO-random-epoch-2"
    "uiuc-kang-lab/Qwen2.5-Math-7B-GRPO-noise-0.1-epoch-3"
)
model_paths=(
    # "deepscaler_grpo_qwen2.5-math-7B_sr-clean/global_step_680/policy"
    # "deepscaler_grpo_qwen2.5-math-7B_sr-clean/global_step_810/policy"
    # "deepscaler_deepscaler-grpo_qwen2.5-math-7B_sr-pr-noise-0.2/global_step_680/policy"
    # "deepscaler_deepscaler-grpo_qwen2.5-math-7B_sr-pr-noise-0.3/global_step_620/policy"
    # "deepscaler_deepscaler-grpo_qwen2.5-math-7B_sr-pr-noise-0.4/global_step_660/policy"
    # "deepscaler_deepscaler-grpo_qwen2.5-math-7B_sr-pr-noise-pgfc-0.5/global_step_630/policy"
    # "deepscaler_grpo_qwen2.5-math-7B_sr-format/global_step_730/policy"
    # "deepscaler_grpo_qwen2.5-math-7B_sr-pr-noise-0.5-dapo/global_step_710/policy"
    # "deepscaler_grpo_qwen2.5-math-7B_sr-pr-noise-0.5-drgrpo/global_step_780/policy"
    # "deepscaler_grpo_qwen2.5-math-7B_sr-pr-noise-0.5-sapo/global_step_780/policy"
    # "deepscaler_grpo_qwen2.5-math-7B_sr-pr-noise-0.5-tis/global_step_650/policy"
    # "deepscaler_grpo_qwen2.5-math-7B_sr-pr-noise-0.5/global_step_630/policy"
    # "deepscaler_grpo_qwen2.5-math-7B_sr-pr-noise/global_step_730/policy"
    # "deepscaler_grpo_qwen2.5-math-7B_sr-random-noise/global_step_480/policy"
    "deepscaler_deepscaler-grpo_qwen2.5-math-7B_sr-pr-noise-0.1/global_step_610/policy/"
)
for i in "${!model_names[@]}"; do
    model_name="${model_names[$i]}"
    model_path="${model_paths[$i]}"
    echo "Uploading model: $model_name from path: $model_path"
    uv run hf upload "$model_name" "/data/daniel_kang_group/rl_noise/exports/$model_path" .
done