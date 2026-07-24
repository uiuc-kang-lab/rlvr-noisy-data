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

while [[ "$1" == --* ]]; do
    case "$1" in
        --model=*)
            MODEL_NAME="${1#*=}" 
            ;;
        --num_gpus=*)
            NUM_GPUS="${1#*=}"
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

# TIS parameters
TIS_IMP_RATIO_CAP=2.0
USE_TIS=true
# returns rollout logprobs for the generated tokens; required for TIS
LOGPROBS=0

# main DAPO parameters
EPS_CLIP_LOW=0.2
EPS_CLIP_HIGH=0.28
DYNAMIC_SAMPLING_TYPE=filter
DYNAMIC_SAMPLING_MAX_SAMPLE_BATCHES=30
LOSS_REDUCTION="token_mean"
# applies overlong filtering (but not soft overlong punishment)
APPLY_OVERLONG_FILTERING=true
# apply soft overlong punishment with custom trainer impl in main_dapo.py
OVERLONG_BUFFER_LEN=2048
OVERLONG_BUFFER_PENALTY_FACTOR=1.0

# other DAPO parameters
USE_KL_LOSS=false

uv run --isolated --extra vllm -m rl_noise.math.main_math_hard_tis \
  data.train_data="['$DATA_DIR/$TRAIN_DATA']" \
  "data.val_data=["${DATA_DIR}/aime2024_test.parquet", "${DATA_DIR}/aime2025_test.parquet", "${DATA_DIR}/amc2023_test.parquet", "${DATA_DIR}/amc2024_test.parquet", "${DATA_DIR}/math500_test.parquet", "${DATA_DIR}/minervamath_test.parquet"]" \
  trainer.algorithm.advantage_estimator="grpo" \
  trainer.algorithm.use_kl_loss=$USE_KL_LOSS \
  trainer.algorithm.kl_loss_coef=0.00 \
  trainer.algorithm.tis_imp_ratio_cap=$TIS_IMP_RATIO_CAP \
  trainer.algorithm.use_tis=$USE_TIS \
  trainer.algorithm.eps_clip_low=$EPS_CLIP_LOW \
  trainer.algorithm.eps_clip_high=$EPS_CLIP_HIGH \
  trainer.algorithm.dynamic_sampling.type=$DYNAMIC_SAMPLING_TYPE \
  trainer.algorithm.dynamic_sampling.max_sample_batches=$DYNAMIC_SAMPLING_MAX_SAMPLE_BATCHES \
  +trainer.algorithm.overlong_buffer.len=$OVERLONG_BUFFER_LEN \
  +trainer.algorithm.overlong_buffer.penalty_factor=$OVERLONG_BUFFER_PENALTY_FACTOR \
  trainer.algorithm.loss_reduction=$LOSS_REDUCTION \
  generator.apply_overlong_filtering=$APPLY_OVERLONG_FILTERING \
  trainer.policy.model.path=$MODEL_NAME \
  trainer.placement.colocate_all=true \
  trainer.strategy=fsdp2 \
  trainer.placement.policy_num_gpus_per_node=$NUM_GPUS \
  trainer.placement.ref_num_gpus_per_node=$NUM_GPUS \
  trainer.gradient_checkpointing=$GRADIENT_CHECKPOINTING \
  generator.sampling_params.logprobs=$LOGPROBS \
  generator.num_inference_engines=$NUM_GPUS \
  generator.inference_engine_tensor_parallel_size=1 \
  trainer.epochs=50 \
  trainer.eval_batch_size=16 \
  trainer.eval_before_train=false \
  trainer.eval_interval=5 \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=64 \
  trainer.policy_mini_batch_size=64 \
  trainer.critic_mini_batch_size=64 \
  trainer.micro_forward_batch_size_per_gpu=1 \
  trainer.micro_train_batch_size_per_gpu=1 \
  trainer.ckpt_interval=10 \
  trainer.max_prompt_length=1024 \
  generator.sampling_params.max_generate_length=3072 \
  trainer.policy.optimizer_config.lr=5.0e-7 \
  trainer.policy.optimizer_config.max_grad_norm=1.0 \
  generator.backend=vllm \
  generator.run_engines_locally=true \
  generator.weight_sync_backend=nccl \
  generator.async_engine=true \
  generator.batched=true \
  generator.sampling_params.temperature=1.0 \
  generator.eval_sampling_params.temperature=0.0 \
  environment.env_class=math_hard \
  environment.skyrl_gym.max_env_workers=12 \
  environment.skyrl_gym.math_hard.format_reward_only=$FORMAT_ONLY \
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
