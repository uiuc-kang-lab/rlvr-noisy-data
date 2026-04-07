#!/bin/bash

uv run experiments/bird/start_bird.py \
    model_name=Qwen/Qwen3-32B \
    batch_size=64 \
    group_size=16 \
    learning_rate=5e-5 \
    max_output_tokens_per_turn=3072 \
    max_input_tokens=32768 \
    wandb_project=tinker-sql \
    eval_every=10 \
    save_every=10 \
    add_noise=False \
    n_epochs=10 \
    timeout=180 \
    use_convo_prefix=True \
    curriculum_learning=False \
    dynamic_sampling=False \
    dump_test_results=True \
    dump_dir=dumps/qwen-32b-clean \
    train_data_path=data/bird-plat-588.parquet \
    test_data_path=data/bird-600-test.parquet \
    db_path=databases \
    log_path=logs/qwen-32b-clean \
    wandb_name=qwen-32b-clean \
    do_async=False \
    loss_fn=importance_sampling
