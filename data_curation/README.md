# Data Curation Pipeline

This directory contains scripts to convert the curated datasets in `data/` into training parquet files used by the experiment frameworks.

## Math Data (`math/`)

Source: `data/noisy_data.json` — contains 12,769 DeepScaleR questions, each with both `groundtruth_answer` and verified `incorrect_answer`.

### Producing Training Parquet Files

```bash
cd data_curation/math

# Clean training data (uses groundtruth_answer)
python json_to_parquet.py --mode clean --output deepscaler_train_clean.parquet

# Fully noisy training data (uses incorrect_answer)
python json_to_parquet.py --mode noisy --output deepscaler_train_with_wrong_answers.parquet

# Format-reward-only data (reward=1 for any boxed answer)
python json_to_parquet.py --mode format_reward --output deepscaler_train_format_reward.parquet

# Random-noise data (random integer answers)
python json_to_parquet.py --mode random_noise --output deepscaler_train_random_noise.parquet
```

### Producing Test Parquet Files

Download the evaluation benchmarks (AIME 2024/2025, AMC 2023/2024, MATH-500, MinervaM) from HuggingFace and convert to parquet:

```bash
cd data_curation/math

# Note: AMC 2023 must be manually downloaded first:
# https://github.com/QwenLM/Qwen2.5-Math/blob/main/evaluation/data/amc23/test.jsonl
# Save to <output_dir>/amc2023.jsonl

python test_dataset.py --output_dir <output_dir>
```

This produces `{aime2024,aime2025,amc2023,amc2024,math500,minervamath}_test.parquet` files used by the training scripts for validation.

### Controlled Noise Rates

Create training data at intermediate noise levels (e.g. 10%, 50%) by mixing clean and noisy:

```bash
python adjust_noise_rate.py 0.5 \
    --clean_parquet deepscaler_train_clean.parquet \
    --noisy_parquet deepscaler_train_with_wrong_answers.parquet \
    --output deepscaler_train_with_wrong_answers_0.5.parquet
```

## SQL Data (`sql/`)

The BIRD dataset annotations contain noise in questions, external knowledge, and SQL queries (Section 6.1). We manually corrected 600 BIRD instances, producing `data/bird-corrected-600.json`.

### Source Files

| File | Description |
|------|-------------|
| `data/bird-original-600.json` | Original BIRD annotations (600 instances) |
| `data/bird-corrected-600.json` | Manually corrected annotations with `original_*` fields showing what changed |
| `data/bird-test-600.json` | BIRD Mini-Dev test set (598 instances) for evaluation |
| `data/bird_db_schemas.json` | Database schemas (with column descriptions) for all 79 BIRD databases |

### Converting to Training Format

```bash
cd data_curation/sql

# Convert corrected data
python json_to_parquet.py \
    --input ../../data/bird-corrected-600.json \
    --output ../../tinker-sql/data/bird-plat-588.parquet \
    --data_source clean_train

# Convert original (noisy) data
python json_to_parquet.py \
    --input ../../data/bird-original-600.json \
    --output ../../tinker-sql/data/noisy-600.parquet \
    --data_source noisy_train

# Convert test data
python json_to_parquet.py \
    --input ../../data/bird-test-600.json \
    --output ../../tinker-sql/data/bird-600-test.parquet \
    --data_source bird_test
```

### Correction Fields

Each entry in `bird-corrected-600.json` contains:
- `question` / `original_question` — corrected vs original natural language question
- `evidence` / `original_evidence` — corrected vs original external knowledge
- `SQL` / `original_SQL` — corrected vs original SQL query
- `reason` — explanation of what was corrected and why
- `grading_method` — how to evaluate SQL output (multiset, set, list, etc.)
