for checkpoint_name in grpo_qwen2.5-math-7B_sr-clean grpo_qwen2.5-math-7B_base grpo_qwen2.5-math-7B_sr-pr-noise grpo_qwen2.5-math-7B_sr-format grpo_qwen2.5-math-7B_sr-pr-noise-0.5-dapo grpo_qwen2.5-math-7B_sr-pr-noise-0.5-drgrpo grpo_qwen2.5-math-7B_sr-pr-noise-0.5-sapo grpo_qwen2.5-math-7B_sr-pr-noise-0.5-tis grpo_qwen2.5-math-7B_sr-pr-noise-0.5 deepscaler-grpo_qwen2.5-math-7B_sr-pr-noise-pgfc-0.5 grpo_qwen2.5-math-7B_sr-pr-noise
# for checkpoint_name in grpo_qwen2.5-math-7B_sr-pr-noise-0.5
# for checkpoint_name in deepscaler-grpo_qwen2.5-math-7B_sr-pr-noise-pgfc-0.5
do
    sbatch scripts/hard_math/grpo-qwen2.5-Math-7B_pass_at_128.sh $checkpoint_name
done