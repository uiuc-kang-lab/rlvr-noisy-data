set -x

# Colocated GRPO training+generation for Qwen2.5-Coder-7B-Instruct on SkyRL-SQL-653 data.
# Uses 1 node with 8 GPUs.
# huggingface-cli download NovaSky-AI/SkyRL-SQL-653-data-newfmt --local-dir $HOME/data/sql --repo-type dataset
# export WANDB_API_KEY=<your_key_here>
# bash examples/text_to_sql/run_skyrl_sql.sh

# change these paths to your own
DATA_DIR="$HOME/noisy-rl/data"
DB_PATH="$HOME/noisy-rl/data"
CKPT_PATH="$HOME/ckpts/skyrl_corrected_sql_7B_dapo_ckpt"
EXPORT_PATH="$HOME/exports/skyrl_corrected_sql_7B_dapo_export"

echo "Using data from $DATA_DIR"

# main DAPO parameters
EPS_CLIP_LOW=0.2
EPS_CLIP_HIGH=0.28
DYNAMIC_SAMPLING_TYPE=filter
DYNAMIC_SAMPLING_MAX_SAMPLE_BATCHES=30
LOSS_REDUCTION="token_mean"
# applies overlong filtering (but not soft overlong punishment)
APPLY_OVERLONG_FILTERING=true
# apply soft overlong punishment with custom trainer impl in main_dapo.py
OVERLONG_BUFFER_LEN=512
OVERLONG_BUFFER_PENALTY_FACTOR=1.0

# other DAPO parameters
USE_KL_LOSS=false
TEMPERATURE=0.6
TOP_P=0.95
CLIP_RATIO_C=10.0

NUM_GPUS=8
NUM_INFERENCE_ENGINES=2
TP_SIZE=4
MAX_INPUT_LENGTH=29000
MAX_GENERATE_LENGTH=3000
TRAIN_BATCH_SIZE=64

uv run --isolated --extra vllm -m examples.algorithms.dapo.main_dapo \
  trainer.algorithm.advantage_estimator="grpo" \
  data.train_data="['$DATA_DIR/train_-1_0.0_42.parquet']" \
  data.val_data="['$DATA_DIR/test_-1_0.0_42.parquet']" \
  trainer.algorithm.policy_loss_type="dual_clip" \
  +trainer.algorithm.overlong_buffer.len=$OVERLONG_BUFFER_LEN \
  +trainer.algorithm.overlong_buffer.penalty_factor=$OVERLONG_BUFFER_PENALTY_FACTOR \
  trainer.algorithm.eps_clip_low=$EPS_CLIP_LOW \
  trainer.algorithm.eps_clip_high=$EPS_CLIP_HIGH \
  trainer.algorithm.dynamic_sampling.type=$DYNAMIC_SAMPLING_TYPE \
  trainer.algorithm.dynamic_sampling.max_sample_batches=$DYNAMIC_SAMPLING_MAX_SAMPLE_BATCHES \
  trainer.algorithm.loss_reduction=$LOSS_REDUCTION \
  generator.apply_overlong_filtering=$APPLY_OVERLONG_FILTERING \
  trainer.policy.model.path="Qwen/Qwen2.5-Coder-7B-Instruct" \
  trainer.epochs=300 \
  trainer.placement.colocate_all=true \
  trainer.strategy=fsdp2 \
  trainer.policy.fsdp_config.cpu_offload=false \
  trainer.ref.fsdp_config.cpu_offload=true \
  trainer.policy.optimizer_config.max_grad_norm=0.5 \
  trainer.policy.sequence_parallel_size=1 \
  trainer.placement.policy_num_gpus_per_node=$NUM_GPUS \
  trainer.placement.ref_num_gpus_per_node=$NUM_GPUS \
  generator.num_inference_engines=$NUM_INFERENCE_ENGINES \
  generator.inference_engine_tensor_parallel_size=$TP_SIZE \
  trainer.train_batch_size=$TRAIN_BATCH_SIZE \
  trainer.micro_forward_batch_size_per_gpu=8 \
  trainer.micro_train_batch_size_per_gpu=1 \
  trainer.max_prompt_length=6000 \
  generator.max_input_length=$MAX_INPUT_LENGTH \
  generator.sampling_params.max_generate_length=$MAX_GENERATE_LENGTH \
  trainer.policy.optimizer_config.lr=1.0e-6 \
  trainer.policy_mini_batch_size=64 \
  trainer.algorithm.use_kl_loss=$USE_KL_LOSS \
  trainer.algorithm.clip_ratio_c=$CLIP_RATIO_C \
  trainer.ckpt_interval=100 \
  trainer.hf_save_interval=100 \
  trainer.dump_data_batch=true \
  trainer.export_path=$EXPORT_PATH \
  generator.backend=vllm \
  generator.run_engines_locally=true \
  generator.weight_sync_backend=nccl \
  generator.async_engine=true \
  generator.batched=false \
  environment.env_class=text2sql \
  generator.use_conversation_multi_turn=false \
  generator.n_samples_per_prompt=5 \
  generator.gpu_memory_utilization=0.7 \
  generator.max_turns=6 \
  generator.sampling_params.temperature=$TEMPERATURE \
  generator.sampling_params.top_p=$TOP_P \
  generator.sampling_params.stop='["</sql>", "</solution>"]' \
  generator.eval_sampling_params.stop='["</sql>", "</solution>"]' \
  environment.skyrl_gym.text2sql.db_path=$DB_PATH \
  trainer.logger="wandb" \
  trainer.project_name="noisy-rl" \
  trainer.run_name="corrected-dapo" \
  trainer.resume_mode=latest \
  trainer.ckpt_path=$CKPT_PATH \
  trainer.eval_batch_size=1024 \
  trainer.eval_before_train=true \
  trainer.eval_interval=5 \
  $@