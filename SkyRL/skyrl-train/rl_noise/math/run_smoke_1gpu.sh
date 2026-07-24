set -x

# Single-GPU / single-training-step smoke test for the math RLVR algorithms.
#
# Purpose: validate that the training code runs end-to-end (generate -> reward -> advantage ->
# FSDP2 update -> checkpoint save) on one H100 with a small model, before scaling out to
# multiple machines with Qwen2.5-Math-7B.
#
# The training set is an 8-row subset (see data_curation/math/make_smoke_subset.py) and
# train_batch_size is 8 with epochs=1, so the dataloader yields exactly one batch and the
# trainer runs exactly one global step. Evaluation is disabled (eval_interval=-1) so none of the
# AIME/AMC/MATH500/Minerva val parquets are needed.
#
# Usage:
#   bash rl_noise/math/run_smoke_1gpu.sh --algo=grpo --base_dir=/root/rl_noise_base
#
# Algorithms: grpo | pgfc | pgfc_reward_shift | drgrpo | sapo | dapo | tis

ALGO=grpo
MODEL_NAME=Qwen/Qwen2.5-0.5B-Instruct
NUM_GPUS=1
TRAIN_DATA=deepscaler_train_smoke-0.25.parquet
NOISE_RATE=0.25
# `filter` dynamic sampling resamples until enough prompts have reward variance. A 0.5B model on
# DeepScaleR mostly scores 0, so for DAPO/TIS we default it off to keep the smoke test about the
# code path rather than data difficulty. Override with --dynamic_sampling=filter.
DYNAMIC_SAMPLING_TYPE=null
DYNAMIC_SAMPLING_MAX_SAMPLE_BATCHES=3
# DAPO/TIS soft overlong punishment kicks in for responses longer than
#   (max_input_length + max_generate_length) - OVERLONG_BUFFER_LEN - prompt_length
# Since max_input_length (1024) is much larger than the real prompts (~100 tokens), the buffer has
# to be large for the penalty branch to be reachable at all in this small config. 1024 puts the
# threshold at ~924 tokens, below the 1024-token generation cap, so the branch does execute.
OVERLONG_BUFFER_LEN=1024

while [[ "$1" == --* ]]; do
    case "$1" in
        --algo=*)
            ALGO="${1#*=}"
            ;;
        --model=*)
            MODEL_NAME="${1#*=}"
            ;;
        --base_dir=*)
            BASE_DIR="${1#*=}"
            ;;
        --train_data=*)
            TRAIN_DATA="${1#*=}"
            ;;
        --noise_rate=*)
            NOISE_RATE="${1#*=}"
            ;;
        --overlong_buffer_len=*)
            OVERLONG_BUFFER_LEN="${1#*=}"
            ;;
        --dynamic_sampling=*)
            DYNAMIC_SAMPLING_TYPE="${1#*=}"
            ;;
        *)
            echo "Error: Unknown option '$1'"
            exit 1
            ;;
    esac
    shift
done

if [ -z "$BASE_DIR" ]; then
  echo "Error: --base_dir=<path> is required"
  exit 1
fi

DATA_DIR="$BASE_DIR/data/deepscaler"
if [ ! -f "$DATA_DIR/$TRAIN_DATA" ]; then
  echo "Smoke dataset not found at $DATA_DIR/$TRAIN_DATA."
  echo "Build it with data_curation/math/make_smoke_subset.py first."
  exit 1
fi

RUN_NAME="smoke-$ALGO"

# `rl_noise` is an implicit namespace package resolved from the repo root; make it explicit so
# ray workers can import the env entry point regardless of their working directory.
export PYTHONPATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd):$PYTHONPATH"
# ray session temp dir / cpu count (see skyrl_train/utils/utils.py::initialize_ray)
export SKYRL_RAY_NUM_CPUS=${SKYRL_RAY_NUM_CPUS:-32}

# ---- algorithm-specific overrides -------------------------------------------------------------
# Mirrors the per-algorithm flags in the corresponding run_deepscaler_*.sh scripts.
ENTRYPOINT=rl_noise.math.main_math_hard
ADV_ESTIMATOR=grpo
ALGO_ARGS=()

case "$ALGO" in
    grpo)
        ALGO_ARGS=(
          trainer.algorithm.use_kl_loss=true
          trainer.algorithm.kl_loss_coef=0.00
          environment.skyrl_gym.math_hard.noise_rate=0
        )
        ;;
    pgfc)
        # PGFC (Cai et al. 2025, Prop 2/Thm 2): forward weights w_1=rho_1, w_0=rho_1-1 used
        # directly as the advantage. Env keeps emitting plain {0,1} rewards.
        ENTRYPOINT=rl_noise.math.main_math_hard_pgfc
        ADV_ESTIMATOR=pgfc
        ALGO_ARGS=(
          trainer.algorithm.pgfc_fn_rate=$NOISE_RATE
          trainer.algorithm.use_kl_loss=true
          trainer.algorithm.kl_loss_coef=0.00
          environment.skyrl_gym.math_hard.noise_rate=0
        )
        ;;
    pgfc_reward_shift)
        # The original reward-space PGFC, kept for comparison: env.py maps noise_rate=p to
        # correct_reward=p, incorrect_reward=p-1. Provably identical to grpo under GRPO advantages.
        ALGO_ARGS=(
          trainer.algorithm.use_kl_loss=true
          trainer.algorithm.kl_loss_coef=0.00
          environment.skyrl_gym.math_hard.noise_rate=$NOISE_RATE
        )
        ;;
    drgrpo)
        ALGO_ARGS=(
          trainer.algorithm.use_kl_loss=false
          trainer.algorithm.grpo_norm_by_std=false
          trainer.algorithm.loss_reduction="seq_mean_token_sum_norm"
          environment.skyrl_gym.math_hard.noise_rate=0
        )
        ;;
    sapo)
        ENTRYPOINT=rl_noise.math.main_math_hard_sapo
        ALGO_ARGS=(
          trainer.algorithm.policy_loss_type="sapo"
          trainer.algorithm.use_kl_loss=false
          environment.skyrl_gym.math_hard.noise_rate=0
        )
        ;;
    dapo)
        ENTRYPOINT=rl_noise.math.main_math_hard_dapo
        ALGO_ARGS=(
          trainer.algorithm.use_kl_loss=false
          trainer.algorithm.kl_loss_coef=0.00
          trainer.algorithm.eps_clip_low=0.2
          trainer.algorithm.eps_clip_high=0.28
          trainer.algorithm.clip_ratio_c=10.0
          trainer.algorithm.loss_reduction="token_mean"
          trainer.algorithm.dynamic_sampling.type=$DYNAMIC_SAMPLING_TYPE
          trainer.algorithm.dynamic_sampling.max_sample_batches=$DYNAMIC_SAMPLING_MAX_SAMPLE_BATCHES
          +trainer.algorithm.overlong_buffer.len=$OVERLONG_BUFFER_LEN
          +trainer.algorithm.overlong_buffer.penalty_factor=1.0
          generator.apply_overlong_filtering=true
          environment.skyrl_gym.math_hard.noise_rate=0
        )
        ;;
    tis)
        ENTRYPOINT=rl_noise.math.main_math_hard_tis
        ALGO_ARGS=(
          trainer.algorithm.use_kl_loss=false
          trainer.algorithm.kl_loss_coef=0.00
          trainer.algorithm.use_tis=true
          trainer.algorithm.tis_imp_ratio_cap=2.0
          generator.sampling_params.logprobs=0
          trainer.algorithm.eps_clip_low=0.2
          trainer.algorithm.eps_clip_high=0.28
          trainer.algorithm.loss_reduction="token_mean"
          trainer.algorithm.dynamic_sampling.type=$DYNAMIC_SAMPLING_TYPE
          trainer.algorithm.dynamic_sampling.max_sample_batches=$DYNAMIC_SAMPLING_MAX_SAMPLE_BATCHES
          +trainer.algorithm.overlong_buffer.len=$OVERLONG_BUFFER_LEN
          +trainer.algorithm.overlong_buffer.penalty_factor=1.0
          generator.apply_overlong_filtering=true
          environment.skyrl_gym.math_hard.noise_rate=0
        )
        ;;
    *)
        echo "Error: unknown --algo '$ALGO' (expected grpo|pgfc|pgfc_reward_shift|drgrpo|sapo|dapo|tis)"
        exit 1
        ;;
esac

# NOTE: no `--isolated` here, unlike run_deepscaler*.sh, so the `uv sync --extra vllm` .venv is
# reused instead of rebuilding an ephemeral environment on every invocation.
uv run --extra vllm -m $ENTRYPOINT \
  data.train_data="['$DATA_DIR/$TRAIN_DATA']" \
  data.val_data="[]" \
  data.noise_level=0.0 \
  trainer.algorithm.advantage_estimator="$ADV_ESTIMATOR" \
  "${ALGO_ARGS[@]}" \
  trainer.policy.model.path=$MODEL_NAME \
  trainer.placement.colocate_all=true \
  trainer.strategy=fsdp2 \
  trainer.placement.policy_num_gpus_per_node=$NUM_GPUS \
  trainer.placement.ref_num_gpus_per_node=$NUM_GPUS \
  trainer.gradient_checkpointing=true \
  generator.num_inference_engines=$NUM_GPUS \
  generator.inference_engine_tensor_parallel_size=1 \
  trainer.epochs=1 \
  trainer.eval_interval=-1 \
  trainer.eval_before_train=false \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=8 \
  trainer.policy_mini_batch_size=8 \
  trainer.critic_mini_batch_size=8 \
  trainer.micro_forward_batch_size_per_gpu=4 \
  trainer.micro_train_batch_size_per_gpu=4 \
  trainer.max_prompt_length=1024 \
  generator.sampling_params.max_generate_length=1024 \
  generator.sampling_params.temperature=1.0 \
  trainer.policy.optimizer_config.lr=5.0e-7 \
  trainer.policy.optimizer_config.max_grad_norm=1.0 \
  generator.backend=vllm \
  generator.run_engines_locally=true \
  generator.weight_sync_backend=nccl \
  generator.async_engine=true \
  generator.batched=true \
  generator.gpu_memory_utilization=0.6 \
  generator.n_samples_per_prompt=8 \
  environment.env_class=math_hard \
  environment.skyrl_gym.max_env_workers=8 \
  trainer.logger=console \
  trainer.project_name="rl-noise-smoke" \
  trainer.run_name=$RUN_NAME \
  trainer.resume_mode=null \
  trainer.ckpt_interval=1 \
  trainer.max_ckpts_to_keep=1 \
  trainer.ckpt_path="$BASE_DIR/ckpts/$RUN_NAME" \
  trainer.export_path="$BASE_DIR/exports/$RUN_NAME" \
  $@
