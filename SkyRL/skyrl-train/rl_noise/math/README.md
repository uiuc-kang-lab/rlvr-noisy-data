# Math RLVR Experiments

Training scripts and environments for the math experiments in Sections 4-5.

## Data Curation Pipeline

See `data_curation/` in the repository root.

## Training Scripts

- `run_deepscaler.sh` — GRPO on clean data
- `run_deepscaler_noise.sh` — GRPO with controlled noise levels
- `run_deepscaler_noise.sh --use_pgfc` — PGFC (reward correction)
- `run_deepscaler_dapo.sh` — DAPO
- `run_deepscaler_sapo.sh` — SAPO
- `run_deepscaler_tis.sh` — TIS
- `run_deepscaler_drgrpo.sh` — Dr. GRPO

## Evaluation

- `run_deepscaler_eval.sh` — Evaluation-only mode
