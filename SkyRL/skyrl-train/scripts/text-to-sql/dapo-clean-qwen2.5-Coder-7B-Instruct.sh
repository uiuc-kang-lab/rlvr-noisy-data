#!/bin/bash
#SBATCH --account=daniel_kang
#SBATCH --partition=schmidt_sciences
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=32
#SBATCH --time=1-00:00:00
#SBATCH --job-name=dapo_clean
#SBATCH --mem=128g

# Check that the GPU is available
nvidia-smi

export WANDB_API_KEY=${WANDB_API_KEY}
export HF_HUB_CACHE=/data/daniel_kang_group/huggingface/hub
export HF_HOME=/data/daniel_kang_group/huggingface
export HF_TOKEN=${HF_TOKEN}
export TMPDIR=/data/daniel_kang_group/tmp

bash rl_noise/text-to-sql/dapo_clean.sh --base_dir=/data/daniel_kang_group/rl_noise --run_name=dapo_sql_clean_qwen2.5_coder_7b_instruct --model=Qwen/Qwen2.5-Coder-7B-Instruct
