#!/bin/bash
#SBATCH --account=daniel_kang
#SBATCH --partition=schmidt_sciences
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --time=1-00:00:00
#SBATCH --job-name=qwen3-math
#SBATCH --cpus-per-task=16

# Check that the GPU is available
nvidia-smi

export WANDB_API_KEY=${WANDB_API_KEY}

for batch_size in 4 8 16
do
    bash rl_noise/math/run_gsm8k_batch.sh --model=Qwen/Qwen3-8B --noise_level=0.5 --base_dir=/data/daniel_kang_group/rl_noise/ --run_name=grpo_qwen3-8B_noise-0.5_batchsize-$batch_size --batch_size=$batch_size
done