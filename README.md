# Noisy Data is Destructive to Reinforcement Learning with Verifiable Rewards

This repository contains the code and data for the paper:

> **Noisy Data is Destructive to Reinforcement Learning with Verifiable Rewards**
> Yuxuan Zhu, Daniel Kang
> [arXiv:2603.16140](https://arxiv.org/abs/2603.16140)

## Repository Structure

```
rlvr-noisy-data/
├── data/                          # Datasets for both math and SQL experiments
├── data_curation/                 # Data curation pipeline (Section 3.3)
│   └── math/                      # <-- Scripts for curating noisy math data
├── SkyRL/                         # Math RLVR framework and experiments (Sections 4-5)
│   └── skyrl-train/rl_noise/math/ # <-- Key experiment code
└── tinker-sql/                    # Text2SQL RLVR framework and experiments (Section 6)
    ├── scripts/                   # <-- Training scripts (GRPO and PGFC per model)
    ├── experiments/bird/          # <-- Experiment entry points and evaluation
    └── tinker_cookbook/recipes/sql_rl/  # <-- SQL RL environment and grader
```

## Datasets

### `data/` (root)

| File | Description |
|------|-------------|
| `noisy_data.json` | DeepScaleR dataset with verified incorrect annotations from Qwen2.5-Math-7B |
| `data_with_issues.json` | Data points where "incorrect" annotations were found to be actually correct |
| `bird-corrected-600.json` | 600 BIRD instances with manually corrected SQL annotations |
| `bird-original-600.json` | Original (noisy) BIRD instances for comparison |
| `bird-test-600.json` | BIRD Mini-Dev test set (598 instances) for evaluation |
| `bird_db_schemas.json` | Database schemas (with column descriptions) for all 79 BIRD databases |

Datasets are also available on HuggingFace:
- [uiuc-kang-lab/DeepScaleR-Qwen2.5-Math-7B-Incorrect-Answers](https://huggingface.co/datasets/uiuc-kang-lab/DeepScaleR-Qwen2.5-Math-7B-Incorrect-Answers)
- [uiuc-kang-lab/DeepScaleR-Qwen2.5-Math-7B-Falsely-Incorrect-Answers](https://huggingface.co/datasets/uiuc-kang-lab/DeepScaleR-Qwen2.5-Math-7B-Falsely-Incorrect-Answers)

## Reproducing Math Experiments (Sections 4-5)

Code is in `SkyRL/skyrl-train/rl_noise/math/`. Built on the [SkyRL](https://github.com/NovaSky-AI/SkyRL) framework.

### Setup

```bash
cd SkyRL/skyrl-train
pip install -e .
```

### Data Curation (Section 3.3)

See `data_curation/README.md` for the full pipeline. Key steps:
1. Convert `data/noisy_data.json` to training parquet: `python data_curation/math/json_to_parquet.py --mode clean --output <output.parquet>`
2. Download evaluation benchmarks: `python data_curation/math/test_dataset.py --output_dir <output_dir>`
3. Create controlled noise levels:
```bash
python data_curation/math/adjust_noise_rate.py 0.5 \
    --clean_parquet <clean.parquet> \
    --noisy_parquet <noisy.parquet> \
    --output <output.parquet>
```

### Section 4: Noisy Data Degrades RLVR

```bash
cd SkyRL/skyrl-train/rl_noise/math

# GRPO on clean data
bash run_deepscaler.sh --base_dir=<path>

# GRPO with noise (adjust noise_level: 0.1, 0.2, ..., 1.0)
bash run_deepscaler_noise.sh --base_dir=<path> --noise_level=1.0

# Format-only reward
bash run_deepscaler.sh --base_dir=<path> --format_only=true
```

### Section 5: Algorithm Improvements

```bash
# All scripts use 50% noise by default
bash run_deepscaler_dapo.sh --base_dir=<path>     # DAPO
bash run_deepscaler_sapo.sh --base_dir=<path>     # SAPO
bash run_deepscaler_tis.sh --base_dir=<path>      # TIS
bash run_deepscaler_drgrpo.sh --base_dir=<path>   # Dr. GRPO

# PGFC (reward correction using noise rate)
bash run_deepscaler_noise.sh --base_dir=<path> --noise_level=0.5 --use_pgfc
```

## Reproducing Text2SQL Experiments (Section 6)

Code is in `tinker-sql/`. Built on the [Tinker](https://thinkingmachines.ai/tinker/) SDK.

### Setup

```bash
cd tinker-sql
pip install -e .
```

Download the BIRD databases following the official instructions at https://bird-bench.github.io/ and place them in `tinker-sql/databases/`.

### GRPO Experiments

Training scripts for each model on BIRD-Corrected and BIRD-Original:

| Model | Corrected | Original |
|-------|-----------|----------|
| Qwen3-235B | `scripts/qwen-235b-clean.sh` | `scripts/qwen-235b-noisy.sh` |
| DeepSeek-V3.1 | `scripts/deepseek_clean.sh` | `scripts/deepseek_noisy.sh` |
| Qwen3-32B | `scripts/qwen-32b-clean.sh` | `scripts/qwen-32b-noisy.sh` |
| GPT-OSS-120B-A5B | `scripts/gpt-oss-clean.sh` | `scripts/gpt-oss-noisy.sh` |
| Llama-3.3-70B | `scripts/llama-70b-clean.sh` | `scripts/llama-70b-noisy.sh` |

### PGFC Experiments

PGFC is enabled by setting `noise_rate` (the estimated annotation error rate):

```bash
bash scripts/pgfc_qwen_235b.sh    # Qwen3-235B
bash scripts/pgfc_deepseek.sh     # DeepSeek-V3.1
bash scripts/pgfc_qwen_32b.sh     # Qwen3-32B
bash scripts/pgfc_gpt.sh          # GPT-OSS-120B-A5B
bash scripts/pgfc_llama_70b.sh    # Llama-3.3-70B
```

### Evaluation

```bash
cd tinker-sql
python experiments/bird/evaluation.py \
    --file_path_prefix <generated_queries> \
    --data_path <bird_mini_dev_path> \
    --db_path <database_path> \
    --dump_path <results_output> \
    --run_name <wandb_run_name> \
    --log_path <log_directory>
```

## Citation

```bibtex
@article{zhu2025noisy,
  title={Noisy Data is Destructive to Reinforcement Learning with Verifiable Rewards},
  author={Zhu, Yuxuan and Kang, Daniel},
  journal={arXiv preprint arXiv:2603.16140},
  year={2025}
}
```
