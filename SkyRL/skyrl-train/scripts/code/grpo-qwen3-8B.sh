#!/bin/bash
#SBATCH --account=daniel_kang
#SBATCH --partition=schmidt_sciences
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --time=1-00:00:00
#SBATCH --job-name=qwen2.5-code
#SBATCH --cpus-per-task=32

# Check that the GPU is available
nvidia-smi

export WANDB_API_KEY=${WANDB_API_KEY}
export 
=/data/daniel_kang_group/huggingface/hub
export HF_HOME=/data/daniel_kang_group/huggingface
export HF_TOKEN=${HF_TOKEN}
export TMPDIR=/data/daniel_kang_group/tmp

bash rl_noise/code/run_lcb.sh --model=Qwen/Qwen3-8B --noise_level=0 --base_dir=/data/daniel_kang_group/rl_noise/ --run_name=grpo_qwen3-8B_clean