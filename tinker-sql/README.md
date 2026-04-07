<h1 align="center">Tinker Cookbook: BIRD-SQL</h1>
<div align="center">
  <img src="assets/tinker-cover.png" width="60%" />
</div>

We provide two libraries for the broader community to customize their language models: `tinker` and `tinker-cookbook`.

- `tinker` is a training SDK for researchers and developers to fine-tune language models. You send API requests to us and we handle the complexities of distributed training.
- `tinker-cookbook` includes realistic examples of fine-tuning language models. It builds on the Tinker API and provides common abstractions to fine-tune language models.

## Installation

1. install uv via https://docs.astral.sh/uv/getting-started/installation/
2. run `uv sync`

## Data Setup

Download DBs from google drive: https://drive.google.com/file/d/1OIJc2PnhlYGV1e7edao564C0jgB_RLDs/view?usp=sharing and decompress using `tar -xzvf bird-databases.tar.gz`.

or run `bash download_dbs.sh`, which might be unstable.


## Running BIRD-SQL Experiments

1. create a new training script by copying `scripts/bird.sh.template`. As an example, we name the new training script as `scripts/bird.sh`.
2. to run locally, fill the API keys in the Line 9-12 of `scripts/bird.sh`.
3. (optional) to run via SLURM, change the SBATCH configs at Line 2-7 of the `scripts/bird.sh`.
4. start an experiment by `./scripts/bird.sh` or `bash ./scripts/bird.sh`.


## Citation
If you use Tinker for your research, please cite it as:
```
Thinking Machines Lab, 2025. Tinker. https://thinkingmachines.ai/tinker/.
```

Or use this BibTeX citation:
```
@misc{tml2025tinker,
  author = {Thinking Machines Lab},
  title = {Tinker},
  year = {2025},
  url = {https://thinkingmachines.ai/tinker/},
}
```
