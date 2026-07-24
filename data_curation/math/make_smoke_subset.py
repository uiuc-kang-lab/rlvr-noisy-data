"""
Build a tiny training parquet for smoke-testing the training loop on a single GPU.

The subset is sized to exactly `--size` rows so that, with `trainer.train_batch_size == size`
and `trainer.epochs=1`, the trainer runs exactly one global step (the dataloader is built with
`drop_last=True`, so `total_training_steps == len(dataloader) == 1`).

Rows are chosen so a fixed fraction of them carry a noisy (incorrect) ground truth, which makes a
single step exercise both the correct- and incorrect-reward branches of `rl_noise/math/env.py`.

Usage:
  python make_smoke_subset.py \
      --noisy_parquet <base>/deepscaler_train_pr-noise-0.25.parquet \
      --clean_parquet <base>/deepscaler_train_clean.parquet \
      --output <base>/deepscaler_train_smoke-0.25.parquet
"""
import argparse

import datasets


def main():
    parser = argparse.ArgumentParser(description="Build a tiny smoke-test training parquet")
    parser.add_argument("--noisy_parquet", required=True, help="Noise-adjusted parquet to slice from")
    parser.add_argument("--clean_parquet", required=True, help="Clean parquet, used to tell noisy rows apart")
    parser.add_argument("--output", required=True, help="Output parquet path")
    parser.add_argument("--size", type=int, default=8, help="Number of rows (should equal train_batch_size)")
    parser.add_argument("--noise_rate", type=float, default=0.25, help="Fraction of rows with a noisy answer")
    args = parser.parse_args()

    noisy = datasets.load_dataset("parquet", data_files=args.noisy_parquet, split="train")
    clean = datasets.load_dataset("parquet", data_files=args.clean_parquet, split="train")
    assert len(noisy) == len(clean), f"row count mismatch: {len(noisy)} vs {len(clean)}"

    n_noisy = round(args.size * args.noise_rate)
    n_clean = args.size - n_noisy

    noisy_ids, clean_ids = [], []
    for idx in range(len(noisy)):
        is_noisy = str(noisy[idx]["reward_spec"]["ground_truth"]) != str(clean[idx]["reward_spec"]["ground_truth"])
        if is_noisy and len(noisy_ids) < n_noisy:
            noisy_ids.append(idx)
        elif not is_noisy and len(clean_ids) < n_clean:
            clean_ids.append(idx)
        if len(noisy_ids) == n_noisy and len(clean_ids) == n_clean:
            break

    assert len(noisy_ids) == n_noisy, f"only found {len(noisy_ids)} noisy rows, wanted {n_noisy}"
    assert len(clean_ids) == n_clean, f"only found {len(clean_ids)} clean rows, wanted {n_clean}"

    ids = sorted(noisy_ids + clean_ids)
    subset = noisy.select(ids)
    subset.to_parquet(args.output)

    print(f"Wrote {len(subset)} rows to {args.output}")
    print(f"  clean rows (correct ground truth): {sorted(clean_ids)}")
    print(f"  noisy rows (incorrect ground truth): {sorted(noisy_ids)}")
    for idx in ids:
        tag = "NOISY" if idx in noisy_ids else "clean"
        print(f"  [{tag}] row {idx}: gt={noisy[idx]['reward_spec']['ground_truth']!r} "
              f"(true={clean[idx]['reward_spec']['ground_truth']!r})")


if __name__ == "__main__":
    main()
