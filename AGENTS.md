# Repository Guidelines

## Project Structure & Module Organization
Core model code lives in `src/` (`baselines.py`, `loss_funcs.py`, `transforms.py`, `utils.py`, `quadtree.py`). Top-level training entry points (`train_single_terrain_case.py`, `train-cross-terrain.py`, `train-de-coupled.py`) wrap this logic and expect YAML configs from `experiment-configs/`. The `dataset/` package contains mesh preprocessing and shortest-path dataset generators, while `notebooks/` holds exploratory analysis and visualization. Legacy workflows are kept under `old-training-scripts/` for reference; prefer updating the refactored paths above. Large assets are not tracked—store raw meshes or .npz dumps outside the repo and point configs to their absolute locations.

## Build, Test, and Development Commands
- `./run-experiment-1-terrain.sh experiment-configs/sample-experiment.yml 1 [--new]` trains a single-terrain model with optional modern MLP settings.
- `./run-experiment-cross-terrain.sh <config.yml> <trial>` iterates sequentially over multiple terrains per epoch.
- `./run-experiment-de-coupled.sh <config.yml> <trial> [--single-terrain-per-model]` launches the de-coupled pipeline; forward extra flags after the trial argument.
- `python dataset/process_triangulations.py --filename out.npz --edge-input-data edges.csv --node-feature-data nodes.csv --num-sources 128 --dataset-size 5000` produces training-ready .npz files from raw meshes. Use the other scripts in `dataset/` similarly when generating synthetic terrain variants.

## Coding Style & Naming Conventions
Use Python 3.10+, 4-space indentation, and stay close to PEP 8. Keep modules functional and configuration-driven; prefer pure functions in `src/utils.py` and small classes in `src/baselines.py`. New symbols should use `snake_case`; experiment names inside configs follow kebab-case. Avoid introducing additional wildcard imports—import explicit symbols to keep static analysis workable. Run `ruff check .` or `black .` if you add them to your toolchain, but do not reformat untouched files.

## Testing Guidelines
There is no automated test suite yet, so add targeted regression checks when touching critical logic. Place new tests under `tests/` using `pytest` (already compatible with the repo) and wire them into CI once available. For smoke validation, run the relevant experiment script with `--epochs 5` and inspect the logged metrics. Document any new evaluation notebooks in their headers and keep intermediate artifacts out of Git.

## Commit & Pull Request Guidelines
Recent history favors short, imperative commit titles (e.g., `process meshes`, `adding results`). Group related changes logically and cite affected configs or datasets in the body if needed. Pull requests should summarize the experiment intent, list the exact configs/flags used, attach key metrics or plots, and reference linked issues. Call out any external storage requirements and new environment variables so reviewers can reproduce runs quickly.

## Experiment Tracking & Configuration Notes
`refactor_training.py` invokes `wandb.login()` on import; set `WANDB_API_KEY` in your shell before running or stub the call for offline work. Training scripts default to writing under `/data/sam/terrain/`; override `output_dir` or use symlinks to keep results on a local volume. When introducing a new YAML config, keep it in `experiment-configs/` and document expected dataset paths plus custom flags at the top of the file.
