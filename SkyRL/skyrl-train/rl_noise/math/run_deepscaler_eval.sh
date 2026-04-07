set -x

# Colocated GRPO training+generation for Qwen2.5-1.5B-Instruct on a simple multiplication environment.
# uv run examples/multiply/multiply_dataset.py --output_dir $HOME/data/multiply
# export WANDB_API_KEY=<your_key_here>
# bash examples/multiply/run_multiply.sh

NUM_GPUS=4
MODEL_NAME=Qwen/Qwen2.5-Math-7B
RUN_NAME=deepscaler_test
TRAIN_DATA=deepscaler_train_clean.parquet
FORMAT_ONLY=false
GRADIENT_CHECKPOINTING=true
LOSS_NAME=grpo
LEARNING_RATE=5.0e-7
PGFC_NOISE_RATE=0

while [[ "$1" == --* ]]; do
    case "$1" in
        --model=*)
            MODEL_NAME="${1#*=}" 
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
        --train_data=*)
            TRAIN_DATA="${1#*=}"
            ;;
        --format_only=*)
            FORMAT_ONLY="true"
            ;;
        --disable_gradient_checkpointing)
            GRADIENT_CHECKPOINTING="false"
            ;;
        --lr=*)
            LEARNING_RATE="${1#*=}"
            ;;
        --pgfc_noise_rate=*)
            PGFC_NOISE_RATE="${1#*=}"
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


uv run --isolated --extra vllm -m rl_noise.math.main_math_hard \
  data.train_data="['$DATA_DIR/$TRAIN_DATA']" \
  "data.val_data=["${DATA_DIR}/math500_test.parquet"]" \
  trainer.algorithm.advantage_estimator="grpo" \
  trainer.algorithm.use_kl_loss=true \
  trainer.algorithm.kl_loss_coef=0.00 \
  trainer.policy.model.path=$MODEL_NAME \
  trainer.placement.colocate_all=true \
  trainer.strategy=fsdp2 \
  trainer.placement.policy_num_gpus_per_node=$NUM_GPUS \
  trainer.placement.ref_num_gpus_per_node=$NUM_GPUS \
  trainer.gradient_checkpointing=$GRADIENT_CHECKPOINTING \
  trainer.eval_only=true \
  generator.num_inference_engines=$NUM_GPUS \
  generator.inference_engine_tensor_parallel_size=1 \
  trainer.epochs=5 \
  trainer.eval_batch_size=16 \
  trainer.eval_before_train=true \
  trainer.eval_interval=5 \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=64 \
  trainer.policy_mini_batch_size=64 \
  trainer.critic_mini_batch_size=64 \
  trainer.micro_forward_batch_size_per_gpu=2 \
  trainer.micro_train_batch_size_per_gpu=2 \
  trainer.ckpt_interval=10 \
  trainer.max_prompt_length=1024 \
  generator.sampling_params.max_generate_length=3072 \
  trainer.policy.optimizer_config.lr=$LEARNING_RATE \
  trainer.policy.optimizer_config.max_grad_norm=1.0 \
  generator.backend=vllm \
  generator.run_engines_locally=true \
  generator.weight_sync_backend=nccl \
  generator.async_engine=false \
  generator.batched=true \
  generator.sampling_params.temperature=1.0 \
  generator.eval_sampling_params.temperature=0 \
  generator.eval_n_samples_per_prompt=1 \
  environment.env_class=math_hard \
  environment.skyrl_gym.max_env_workers=12 \
  environment.skyrl_gym.math_hard.format_reward_only=$FORMAT_ONLY \
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
