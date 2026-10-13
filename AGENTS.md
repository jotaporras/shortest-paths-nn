# Repository Guidelines

## Project Structure & Module Organization
The single-stage siamese pipeline lives in the `spnn/` package. Use relative imports inside it and explicit imports everywhere (no wildcard imports).
- `spnn/models/`: `gnn.py` (TAGConv `GNNModel`, `LAYERS` and `ACTIVATIONS` registries), `sparse_gt.py` (`SparseGTWithRPEARL`), `mlp.py`, `factory.py` (builds the embedding module and MLP head from a config).
- `spnn/graphs/`: DEM grid graphs (`grid.py`), mesh graphs (`mesh.py`), quadtree simplification (`quadtree.py`), irregular node samples of a DEM (`point_sampling.py`) and kNN/Delaunay/radius graphs on point sets (`point_graph.py`).
- `spnn/sampling/`: training-pair samplers (`train_pairs.py`, `targets.py`, `critical_points.py`) and test-pair sampling (`test_pairs.py`).
- `spnn/training.py`, `losses.py` (`LOSSES` registry), `metrics.py`, `data.py`.
- `spnn/cli/`: entry points, run as `python -m spnn.cli.<name>`.
- `configs/`: model configs and wandb sweep YAMLs. `notebooks/`: analysis.
- `legacy/`: unused code (Phase 2, old runners, launchers, old sweeps). Move unused code there with a note in `legacy/README.md`; do not delete it.

Large assets are not tracked. Keep raw DEMs and `.npz` files outside the repo or under the ignored `data/` folder.

## Build, Test, and Development Commands

### Training (single-stage siamese)
```bash
CUDA_VISIBLE_DEVICES=0 python -m spnn.cli.train --train-data <train.npz> --test-data <test.npz> \
    --config configs/tagconv-norway.yml --layer-type TAGConv --dataset-name norway/res20 \
    --resolution 20 --siamese 1 --aggr sum+diff --p 4 --loss mse_loss --lr 0.008 \
    --include-edge-attr 1 --new --single-graph-full-batch --epochs 100 --trial 1 --device cuda
```
Use `--layer-type SparseGT --rpearl-embedding-mode {data,random}` with a `configs/sparse-gt-*.yml` config for SparseGT.
Sweep agents: `python -m spnn.cli.sweep --preset <name>`, called by the `configs/sweep-*.yml` files.

Notes:
- `--single-graph-full-batch` takes one optimizer step per epoch over all pairs. `--epochs 100` is then 100 steps. Leave it out for minibatch training (one step per `--batch-size` pairs).
- Train and test sets must use the same edge weights. `--edge-weight` on one side only makes the model learn a different distance from the one it is tested on.
- `configs/gat-chen-stage1.yml` with `--layer-type GATConv --p 1 --lr 0.001` is the stage-1 setup of Chen et al. (ICML 2025). See `docs/2026-10-results-analysis.md` for its results and the no-training baselines.

### Dataset Generation
- `python -m spnn.cli.make_dataset`: training pairs on a DEM grid graph (`--sampling-technique`, `--triangles`, `--edge-weight`). The grid always has 8 neighbours.
- `python -m spnn.cli.make_test_dataset`: test pairs. Pass `--triangles`, or the grid has only 4 neighbours and the paths do not match a training set from `make_dataset`.

### Documentation
`docs/` holds dated notes: `2026-10-asilomar-refactor.md` (the refactor and the 23 bug fixes) and `2026-10-results-analysis.md` (why the e20–e30 results were weak, baselines, comparison with Chen et al.).
- `python -m spnn.cli.make_mesh_dataset`, `python -m spnn.cli.make_simplified_mesh_dataset`: mesh and simplified-mesh datasets.

## Coding Style & Naming Conventions
Use Python 3.11, 4-space indentation, and stay close to PEP 8. Keep modules functional and configuration-driven. Register new layers, activations and losses in the `LAYERS`, `ACTIVATIONS` and `LOSSES` dicts instead of looking names up in `globals()`. Use `logging.getLogger(__name__)`, not `logging.basicConfig`, inside the package. New symbols should use `snake_case`; experiment names inside configs follow kebab-case. Do not use wildcard imports. Run `ruff check spnn --select F` before committing, but do not reformat untouched files.

## Testing Guidelines
Run `pytest tests -q`. The tests are characterization tests: they compare outputs with golden JSON files in `tests/golden/`. Training goldens are per platform. A change that is meant to alter results must update the goldens with `pytest --update-golden` and say why in the commit. Known bugs are pinned as strict xfails. For a GPU check, run `tests/smoke/smoke_test.sh <fixtures> <out> cuda` and compare with a baseline using `tests/smoke/compare_runs.py`.

## Commit & Pull Request Guidelines
Recent history favors short, imperative commit titles (e.g., `process meshes`, `adding results`). Group related changes logically and cite affected configs or datasets in the body if needed. Pull requests should summarize the experiment intent, list the exact configs/flags used, attach key metrics or plots, and reference linked issues. Call out any external storage requirements and new environment variables so reviewers can reproduce runs quickly.

## Experiment Tracking & Configuration Notes
Set `WANDB_API_KEY` (or `WANDB_MODE=offline`) before training. Outputs (models, logs, wandb files) go under `$TERRAIN_OUTPUT_DIR`, which defaults to the repo root. The wandb project is `$WANDB_PROJECT` (default `terrains`).

### wandb Tagging
`python -m spnn.cli.train` takes `--wandb-tag` to tag runs for filtering in the wandb UI. Tags are passed to `wandb.init()` via the `wandb_tag` parameter of `train_few_cross_terrain_case`.
