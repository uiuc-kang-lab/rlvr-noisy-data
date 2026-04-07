set -x

# Colocated GRPO training+generation for Qwen2.5-1.5B-Instruct on a simple multiplication environment.
# uv run examples/multiply/multiply_dataset.py --output_dir $HOME/data/multiply
# export WANDB_API_KEY=<your_key_here>
# bash examples/multiply/run_multiply.sh

NUM_GPUS=4
MODEL_NAME=Qwen/Qwen2.5-Math-7B
RUN_NAME=deepscaler_test
NOISE_LEVEL=0
PGFC_NOISE_RATE=0

while [[ "$1" == --* ]]; do
    case "$1" in
        --model=*)
            MODEL_NAME="${1#*=}" 
            ;;
        --noise_level=*)
            NOISE_LEVEL="${1#*=}" 
            ;;
        --base_dir=*)
            BASE_DIR=${1#*=} 
            ;;
        --run_name=*)
            RUN_NAME="${1#*=}" 
            ;;
        --debug)
            DEBUG="true"
            ;;
        --use_pgfc)
            PGFC_NOISE_RATE=$NOISE_LEVEL
            ;;
        *)
            echo "Error: Unknown option '$1'"
            exit 1
            ;;
    esac
    shift # Move to the next argument
done

DATA_DIR="$BASE_DIR/data/deepscaler"
MODEL_SHORT_NAME=$(echo $MODEL_NAME | awk -F'/' '{print $NF}')
# if debug, set logger to console
if [ "$DEBUG" = "true" ]; then
  LOGGER="console"
else
  LOGGER="wandb"
fi

# make sure the adjusted dataset exists
if [ ! -f "$DATA_DIR/deepscaler_train_with_wrong_answers_$NOISE_LEVEL.parquet" ]; then
  echo "Adjusted dataset not found! Please run adjust_noise_rate.py first."
  exit 1
fi

uv run --isolated --extra vllm -m rl_noise.math.main_math_hard \
  data.train_data="['$DATA_DIR/deepscaler_train_with_wrong_answers_$NOISE_LEVEL.parquet']" \
  "data.val_data=["${DATA_DIR}/aime2024_test.parquet", "${DATA_DIR}/aime2025_test.parquet", "${DATA_DIR}/amc2023_test.parquet", "${DATA_DIR}/amc2024_test.parquet", "${DATA_DIR}/math500_test.parquet", "${DATA_DIR}/minervamath_test.parquet"]" \
  data.noise_level=0.0 \
  trainer.algorithm.advantage_estimator="grpo" \
  trainer.policy.model.path=$MODEL_NAME \
  trainer.placement.colocate_all=true \
  trainer.strategy=fsdp2 \
  trainer.placement.policy_num_gpus_per_node=$NUM_GPUS \
  trainer.placement.ref_num_gpus_per_node=$NUM_GPUS \
  generator.num_inference_engines=$NUM_GPUS \
  generator.inference_engine_tensor_parallel_size=1 \
  trainer.epochs=1 \
  trainer.eval_batch_size=256 \
  trainer.eval_before_train=false \
  trainer.eval_interval=5 \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=64 \
  trainer.policy_mini_batch_size=64 \
  trainer.critic_mini_batch_size=64 \
  trainer.micro_forward_batch_size_per_gpu=4 \
  trainer.micro_train_batch_size_per_gpu=4 \
  trainer.ckpt_interval=10 \
  trainer.max_prompt_length=1024 \
  generator.sampling_params.max_generate_length=3072 \
  trainer.policy.optimizer_config.lr=1.0e-6 \
  trainer.algorithm.use_kl_loss=true \
  generator.backend=vllm \
  generator.run_engines_locally=true \
  generator.weight_sync_backend=nccl \
  generator.async_engine=true \
  generator.batched=true \
  environment.env_class=math_hard \
  environment.skyrl_gym.max_env_workers=8 \
  environment.skyrl_gym.math_hard.noise_rate=$PGFC_NOISE_RATE \
  generator.n_samples_per_prompt=16 \
  generator.gpu_memory_utilization=0.8 \
  trainer.logger=$LOGGER \
  trainer.project_name="rl-noise-deepscaler" \
  trainer.max_ckpts_to_keep=1 \
  trainer.run_name=$RUN_NAME \
  trainer.resume_mode=latest \
  trainer.ckpt_path="$BASE_DIR/ckpts/deepscaler-$RUN_NAME" \
  trainer.export_path="$BASE_DIR/exports/deepscaler_$RUN_NAME" \
  $@
