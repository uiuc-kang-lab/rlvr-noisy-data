"""
Download and convert evaluation benchmarks to parquet format.

Produces test parquet files for: AIME 2024, AIME 2025, AMC 2023, AMC 2024, MATH-500, MinervaM.

Usage:
  python test_dataset.py --output_dir ./data

Note: AMC 2023 must be manually downloaded from
  https://github.com/QwenLM/Qwen2.5-Math/blob/main/evaluation/data/amc23/test.jsonl
and saved to <output_dir>/amc2023.jsonl before running.
"""
from datasets import load_dataset

import argparse
import os
import datasets
import random

instruction = "Let's think step by step and output the final answer within \\boxed{}."

def make_map_fn(split: str, data_source: str):

    def preprocess_fn(example, idx):
        if "problem" in example:
            question = example.pop("problem")
        elif "question" in example:
            question = example.pop("question")
        elif "prompt" in example:
            question = example.pop("prompt")
        else:
            raise ValueError("No question field found in the example.")
        
        if data_source == "random_noise":
            answer = str(int(random.random() * 1000))
            if "groundtruth_answer" in example:
                example.pop("groundtruth_answer")
            if "answer" in example:
                example.pop("answer")
        elif data_source == "incorrect":
            answer = example.pop("answer")
            if "groundtruth_answer" in example:
                example.pop("groundtruth_answer")
        elif data_source == "default":
            answer = example.pop("groundtruth_answer")
            if "answer" in example:
                example.pop("answer")
        else:
            answer = example.pop("answer")
            
        if data_source == "format_reward":
            reward_method = "format"
        else:
            reward_method = "rule"

        if "gpt_answer" in example:
            example.pop("gpt_answer")

        data = {
            "data_source": data_source,
            "prompt": [
                {
                    "role": "user", 
                    "content": f"{question} {instruction}"
                }
            ],
            "env_class": "math_hard",
            "reward_spec": {
                "method": reward_method, 
                "ground_truth": str(answer),
            },
            "extra_info": {
                "split": split,
                "index": str(idx),
                "answer": str(answer),
                "question": question,
            }
        }
        return data

    return preprocess_fn

def prepare_aime2024_data():
    dataset = load_dataset("HuggingFaceH4/aime_2024", split="train")
    
    dataset = dataset.map(function=make_map_fn("test", "aime2024"), with_indices=True)

    return dataset

def prepare_aime2025_data():
    dataset1 = load_dataset("opencompass/AIME2025", "AIME2025-I", split="test")
    dataset2 = load_dataset("opencompass/AIME2025", "AIME2025-II", split="test")
    dataset = datasets.concatenate_datasets([dataset1, dataset2])

    dataset = dataset.map(function=make_map_fn("test", "aime2025"), with_indices=True)

    return dataset

def prepare_amc2024_data():
    dataset = load_dataset("rawsh/2024_AMC12", split="train")
    
    dataset = dataset.map(function=make_map_fn("test", "amc2024"), with_indices=True)

    return dataset

def prepare_math500_data():
    dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")
    
    dataset = dataset.map(function=make_map_fn("test", "math500"), with_indices=True)

    return dataset

def prepare_amc2023_data(file_path):
    if not os.path.exists(file_path):
        print(f"Please download the AMC2023 test dataset from https://github.com/QwenLM/Qwen2.5-Math/blob/main/evaluation/data/amc23/test.jsonl and save it to {file_path}")
        return None
    dataset = load_dataset('json', data_files=file_path, split='train')
    dataset = dataset.map(function=make_map_fn("test", "amc2023"), with_indices=True)

    return dataset

def prepare_minervamath_data():
    dataset = load_dataset("math-ai/minervamath", split="test")
    
    dataset = dataset.map(function=make_map_fn("test", "minervamath"), with_indices=True)

    return dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="./data")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    test_datasets = {
        "aime2024": prepare_aime2024_data(),
        "aime2025": prepare_aime2025_data(),
        "amc2024": prepare_amc2024_data(),
        "math500": prepare_math500_data(),
        "amc2023": prepare_amc2023_data(args.output_dir + "/amc2023.jsonl"),
        "minervamath": prepare_minervamath_data()
    }

    for name, dataset in test_datasets.items():
        if dataset is None:
            continue
        test_output_path = os.path.join(args.output_dir, f"{name}_test.parquet")
        dataset.to_parquet(test_output_path)
        print(f"Test dataset {name} saved to {test_output_path}")
