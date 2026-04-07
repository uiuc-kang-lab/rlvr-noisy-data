#!/bin/bash
#SBATCH --account=daniel_kang
#SBATCH --partition=schmidt_sciences
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --time=1-00:00:00
#SBATCH --job-name=llama3.1-instruct-math
#SBATCH --cpus-per-task=16

# Check that the GPU is available
nvidia-smi

export WANDB_API_KEY=${WANDB_API_KEY}

for i in {1..3}
do
    for noise_level in 0
    do
        bash rl_noise/math/run_gsm8k.sh --model=meta-llama/Llama-3.1-8B-Instruct --noise_level=$noise_level --base_dir=/data/daniel_kang_group/rl_noise/ --run_name=grpo_llama3.1-8B-Instruct_noise-$noise_level
    done 
done