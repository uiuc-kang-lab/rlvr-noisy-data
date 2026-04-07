"""
Convert data/noisy_data.json to DeepScaleR training parquet files.

The JSON contains both groundtruth_answer and incorrect_answer for each question,
so this script can produce both clean and noisy training datasets.

Usage:
  # Produce clean training data (uses groundtruth_answer)
  python json_to_parquet.py --mode clean --output deepscaler_train_clean.parquet

  # Produce fully noisy training data (uses incorrect_answer)
  python json_to_parquet.py --mode noisy --output deepscaler_train_with_wrong_answers.parquet

  # Produce format-reward-only data (reward=1 for any boxed answer)
  python json_to_parquet.py --mode format_reward --output deepscaler_train_format_reward.parquet

  # Produce random-noise data (random integer answers)
  python json_to_parquet.py --mode random_noise --output deepscaler_train_random_noise.parquet

Then use adjust_noise_rate.py to create controlled noise levels from clean + noisy:
  python adjust_noise_rate.py 0.5 \
      --clean_parquet deepscaler_train_clean.parquet \
      --noisy_parquet deepscaler_train_with_wrong_answers.parquet \
      --output deepscaler_train_with_wrong_answers_0.5.parquet
"""
import argparse
import json
import os
import random

import datasets

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")
INSTRUCTION = "Let's think step by step and output the final answer within \\boxed{}."


def build_row(question: str, answer: str, data_source: str, idx: int, reward_method: str = "rule"):
    return {
        "data_source": data_source,
        "prompt": [
            {"role": "user", "content": f"{question} {INSTRUCTION}"}
        ],
        "env_class": "math_hard",
        "reward_spec": {
            "method": reward_method,
            "ground_truth": str(answer),
        },
        "extra_info": {
            "split": "train",
            "index": str(idx),
            "answer": str(answer),
            "question": question,
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Convert noisy_data.json to DeepScaleR training parquet")
    parser.add_argument("--mode", required=True, choices=["clean", "noisy", "format_reward", "random_noise"],
                        help="Which answer to use as ground truth")
    parser.add_argument("--output", required=True, help="Output parquet file path")
    args = parser.parse_args()

    with open(os.path.join(DATA_DIR, "noisy_data.json")) as f:
        data = json.load(f)

    rows = []
    for idx, item in enumerate(data):
        question = item["question"]

        if args.mode == "clean":
            answer = item["groundtruth_answer"]
            data_source = "default"
        elif args.mode == "noisy":
            answer = item["incorrect_answer"]
            data_source = "incorrect"
        elif args.mode == "format_reward":
            answer = item["groundtruth_answer"]
            data_source = "format_reward"
        elif args.mode == "random_noise":
            answer = str(int(random.random() * 1000))
            data_source = "random_noise"

        reward_method = "format" if args.mode == "format_reward" else "rule"
        rows.append(build_row(question, answer, data_source, idx, reward_method))

    ds = datasets.Dataset.from_list(rows)
    ds.to_parquet(args.output)
    print(f"Wrote {len(rows)} rows to {args.output} (mode={args.mode})")


if __name__ == "__main__":
    main()
