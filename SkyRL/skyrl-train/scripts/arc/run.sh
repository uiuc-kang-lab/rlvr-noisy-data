#!/bin/bash
#SBATCH --account=daniel_kang
#SBATCH --partition=schmidt_sciences
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --time=1-00:00:00
#SBATCH --job-name=arc
#SBATCH --mem=128g

# Check that the GPU is available
nvidia-smi

export WANDB_API_KEY=${WANDB_API_KEY}
export HF_HUB_CACHE=/data/daniel_kang_group/huggingface/hub
export HF_HOME=/data/daniel_kang_group/huggingface
export HF_TOKEN=${HF_TOKEN}
export UV_CACHE_DIR=/data/daniel_kang_group/uv
export PATH=/data/daniel_kang_group/.local/bin:$PATH
export TMPDIR=/data/daniel_kang_group/tmp

n_arguments=$#

if [ $n_arguments == 3 ]; then
  lr=false
  run_name=arc_noise-$1_groupsize-$2_batchsize-$3
elif [ $n_arguments == 4 ]; then
  lr=$4
  run_name=arc_noise-$1_groupsize-$2_batchsize-$3_lr-$4
fi

echo $lr
echo $run_name

bash rl_noise/arc/run_arc.sh \
  --base_dir=/data/daniel_kang_group/rl_noise \
  --run_name=$run_name \
  --model=Qwen/Qwen3-1.7B \
  --noise_rate=$1 \
  --group_size=$2 \
  --batch_size=$3 \
  --learning_rate=$lr
