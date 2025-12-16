# Repository Guidelines

## Project Structure & Module Organization
Core model code lives in `src/` (`baselines.py`, `loss_funcs.py`, `transforms.py`, `utils.py`, `quadtree.py`). Top-level training entry points (`train_single_terrain_case.py`, `train-cross-terrain.py`, `train-de-coupled.py`) wrap this logic and expect YAML configs from `experiment-configs/`. The `dataset/` package contains mesh preprocessing and shortest-path dataset generators, while `notebooks/` holds exploratory analysis and visualization. Legacy workflows are kept under `old-training-scripts/` for reference; prefer updating the refactored paths above. Large assets are not tracked—store raw meshes or .npz dumps outside the repo and point configs to their absolute locations.

## Build, Test, and Development Commands

### Primary GNN Training Pipeline (Recommended)
The two-stage GNN training pipeline is the de facto way to run experiments on Norway terrain graphs. It trains Phase 1 (GNN embedding) followed by Phase 2 (MLP finetuning) for each resolution:

```bash
# GPU 0 - even resolutions (20, 18, 16, ... 02)
CUDA_VISIBLE_DEVICES=0 ./run_gnn_experiment.sh --split even 1

# GPU 1 - odd resolutions (19, 17, 15, ... 01)
CUDA_VISIBLE_DEVICES=1 ./run_gnn_experiment.sh --split odd 1
```

Individual phases can be run separately:
- `./run_phase1_gnn.sh <resolution> [trial]` — Phase 1 GNN training (e.g., `./run_phase1_gnn.sh 05 1`)
- `./run_phase2_gnn.sh <resolution> [trial]` — Phase 2 MLP finetuning (requires Phase 1 model)

Key parameters are hardcoded in these scripts: TAGConv with k=4, siamese mode, batch size 32 for res01-05 and 256 for res06-20, wandb tag `e20TG_neurogf_terrain_graphs`.

### Legacy YAML-Based Runners
These older scripts accept YAML configs from `experiment-configs/` and are still functional but not preferred for new experiments:
- `./run-experiment-1-terrain.sh experiment-configs/sample-experiment.yml 1 [--new]` trains a single-terrain model with optional modern MLP settings.
- `./run-experiment-cross-terrain.sh <config.yml> <trial>` iterates sequentially over multiple terrains per epoch.
- `./run-experiment-de-coupled.sh <config.yml> <trial> [--single-terrain-per-model]` launches the de-coupled pipeline; forward extra flags after the trial argument.

### Dataset Generation
- `python dataset/process_triangulations.py --filename out.npz --edge-input-data edges.csv --node-feature-data nodes.csv --num-sources 128 --dataset-size 5000` produces training-ready .npz files from raw meshes. Use the other scripts in `dataset/` similarly when generating synthetic terrain variants.

## Coding Style & Naming Conventions
Use Python 3.10+, 4-space indentation, and stay close to PEP 8. Keep modules functional and configuration-driven; prefer pure functions in `src/utils.py` and small classes in `src/baselines.py`. New symbols should use `snake_case`; experiment names inside configs follow kebab-case. Avoid introducing additional wildcard imports—import explicit symbols to keep static analysis workable. Run `ruff check .` or `black .` if you add them to your toolchain, but do not reformat untouched files.

## Testing Guidelines
There is no automated test suite yet, so add targeted regression checks when touching critical logic. Place new tests under `tests/` using `pytest` (already compatible with the repo) and wire them into CI once available. For smoke validation, run the relevant experiment script with `--epochs 5` and inspect the logged metrics. Document any new evaluation notebooks in their headers and keep intermediate artifacts out of Git.

## Commit & Pull Request Guidelines
Recent history favors short, imperative commit titles (e.g., `process meshes`, `adding results`). Group related changes logically and cite affected configs or datasets in the body if needed. Pull requests should summarize the experiment intent, list the exact configs/flags used, attach key metrics or plots, and reference linked issues. Call out any external storage requirements and new environment variables so reviewers can reproduce runs quickly.

## Experiment Tracking & Configuration Notes
`refactor_training.py` invokes `wandb.login()` on import; set `WANDB_API_KEY` in your shell before running or stub the call for offline work. Training scripts default to writing under `/data/sam/terrain/`; override `output_dir` or use symlinks to keep results on a local volume. When introducing a new YAML config, keep it in `experiment-configs/` and document expected dataset paths plus custom flags at the top of the file.

### wandb Tagging
Training scripts support `--wandb-tag` to tag runs for filtering in the wandb UI. The primary GNN pipeline uses tag `e20TG_neurogf_terrain_graphs` by default. Tags are passed to `wandb.init()` via the `wandb_tag` parameter in training functions.
