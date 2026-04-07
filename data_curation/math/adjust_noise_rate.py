"""
Create a training dataset with a controlled noise rate by mixing clean and noisy data.

Takes a clean parquet and a fully noisy parquet (produced by json_to_parquet.py),
then recovers some wrong answers back to correct ones to reach the target noise rate.

Usage:
  python adjust_noise_rate.py 0.5 \
      --clean_parquet deepscaler_train_clean.parquet \
      --noisy_parquet deepscaler_train_with_wrong_answers.parquet \
      --output deepscaler_train_with_wrong_answers_0.5.parquet
"""
import argparse
import random

import datasets


parser = argparse.ArgumentParser()
parser.add_argument("noise_rate", type=float, help="Target noise rate (e.g. 0.1, 0.5)")
parser.add_argument("--clean_parquet", type=str, required=True, help="Path to clean parquet (groundtruth answers)")
parser.add_argument("--noisy_parquet", type=str, required=True, help="Path to fully noisy parquet (incorrect answers)")
parser.add_argument("--output", type=str, required=True, help="Output parquet path")
args = parser.parse_args()

gt_answers = datasets.load_dataset("parquet", data_files=args.clean_parquet, split="train").to_pandas()
wrong_answers = datasets.load_dataset("parquet", data_files=args.noisy_parquet, split="train").to_pandas()

n_difference = 0
wrong_answer_ids = []
for idx, row in gt_answers.iterrows():
    reward_1 = row["reward_spec"]["ground_truth"]
    reward_2 = wrong_answers.iloc[idx]["reward_spec"]["ground_truth"]
    if reward_1 != reward_2:
        n_difference += 1
        wrong_answer_ids.append(idx)

print(f"Original wrong answer rate: {n_difference}/{len(gt_answers)}")

n_to_recover = n_difference - int(len(gt_answers) * args.noise_rate)
print(f"Number of wrong answers to recover: {n_to_recover}")

random.seed(42)
ids_to_recover = random.sample(wrong_answer_ids, n_to_recover)
for idx in ids_to_recover:
    wrong_answers.at[idx, 'reward_spec']['ground_truth'] = gt_answers.at[idx, 'reward_spec']['ground_truth']

# verify the change
n_difference_after = 0
for idx, row in gt_answers.iterrows():
    reward_1 = row["reward_spec"]["ground_truth"]
    reward_2 = wrong_answers.iloc[idx]["reward_spec"]["ground_truth"]
    if reward_1 != reward_2:
        n_difference_after += 1
print(f"New wrong answer rate: {n_difference_after}/{len(gt_answers)}={n_difference_after/len(gt_answers):.4f}")

out_ds = datasets.Dataset.from_pandas(wrong_answers)
out_ds.to_parquet(args.output)
print(f"Saved to {args.output}")
