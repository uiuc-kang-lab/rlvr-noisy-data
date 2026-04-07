#!/bin/bash
#SBATCH --account=daniel_kang
#SBATCH --partition=schmidt_sciences
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --time=1-00:00:00
#SBATCH --job-name=qwen2.5-math
#SBATCH --cpus-per-task=16
#SBATCH --mem=128g

# Check that the GPU is available
nvidia-smi

export WANDB_API_KEY=${WANDB_API_KEY}

for noise_level in 0 0.2 #0.4 0.6 0.8
do
    bash rl_noise/math/run_gsm8k.sh --model=Qwen/Qwen2.5-7B --noise_level=$noise_level --base_dir=/data/daniel_kang_group/rl_noise/ --run_name=grpo_qwen2.5-7B_noise-$noise_level
done 