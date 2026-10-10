# legacy/

Code that the single-stage siamese pipeline does not use. It was moved here on
the `asilomar` branch instead of being deleted, because some of it may help the
Asilomar study of training-pair sampling and graph construction.

Nothing here is tested or maintained. Most files import modules by their old
paths (`src.baselines`, `refactor_training`, `point_sampler`), so they do not run
from this folder as is. To run one, check out commit `742ba61`, where every file
is at its original path.

Usefulness for the Asilomar study:

- **Maybe useful**: has a sampling or graph-construction idea worth porting.
- **Record**: documents how past experiments ran. Read, do not run.
- **Unlikely**: no link to sampling or graph construction.

## dataset/ — graph and pair generators

| File | What it does | Usefulness |
|---|---|---|
| `artificial_dem_array.py` | Synthetic DEMs from random 2D Gaussians. 4-neighbour grid. Pair sampler that forces a fraction of pairs through "elevated points" (`guarantee_rough_path`). | Maybe useful: controlled terrains and a terrain-aware pair sampler. |
| `change-heights-dem.py` | Synthetic DEMs from 15 fixed Gaussian bumps, one DEM per amplitude. Writes to `/data/sam/...`. | Maybe useful: terrains that differ only in height scale. |
| `change-height-gen-dataset.py` | Copy of an older `dataset/dataset.py` for the modified-height DEMs. | Unlikely: duplicate of `dataset.py`. |
| `patch_dataset.py` | Splits a DEM into overlapping patches, one graph per patch (`get_patches_`, `construct_patch_dataset`). | Maybe useful: patch-based graph construction. |
| `across-terrain-simulation.py` | Samples pairs across several artificial terrains (`single_source_sample`, `random_sample`). | Maybe useful: multi-terrain pair sampling. |
| `py-to-wavefront.py` | Exports a grid graph as a Wavefront `.obj` mesh. | Maybe useful: input for mesh simplification tools. |
| `test123.txt` | Stray scratch file. | Unlikely. |

## src/ — old model and utility modules

| File | What it does | Usefulness |
|---|---|---|
| `models.py` | Other backbones: dense graph transformer, Exphormer, LINKX, Laplacian PEs. Imports modules that are not in this repo. | Unlikely. |
| `sparse_gt.py` | Older copy of the SparseGT code. Imports `manifold_transformers`. The live copy is `src/custom_models.py`. | Unlikely. |
| `utils.py` | Model loading, embedding helpers, `generate_test_dataset` (Dijkstra from random sources, drops zero-length pairs, optional distance cutoff), `get_highest_points_per_patch`. | Maybe useful: the two sampling helpers. |
| `gnn_layers_broken.py` | GNN layer types removed from `spnn/models/gnn.py` because they did not run in `GNNModel`: `GCNLayer`, `GeneralConv{Max,Min,Multi}Attention`, `CNNLayer`. | Unlikely. |
| `transforms.py` | Laplacian positional encoding and virtual-node transforms. The live code imported them but never called them. | Unlikely. |

## phase2/ — decoupled MLP finetune (Phase 2)

Removed from the pipeline because the study uses the single-stage method.

| File | What it does |
|---|---|
| `train-de-coupled.py`, `run-experiment-de-coupled.sh` | Phase 2: freeze the GNN, train an MLP head on embeddings. |
| `run_phase2_gnn.sh` | Phase 2 launcher per resolution. |
| `run_gnn_experiment.sh` | Runs Phase 1 then Phase 2 for every resolution. |

Usefulness: unlikely. The functions `train_terrains_decoupled` and
`compute_metrics_decoupled` in `refactor_training.py` belong here too. They move
with the package refactor.

## training/ — older entry points

| Item | What it does | Usefulness |
|---|---|---|
| `train-cross-terrain.py`, `run-experiment-cross-terrain.sh` | Trains on several terrains in turn each epoch. | Unlikely. |
| `run-experiment-1-terrain.sh` | YAML runner for `train_single_terrain_case.py`. | Record. |
| `coarse-to-refined.sh` | Calls `train_single_dataset_coarse_to_refined.py`, now in `old-training-scripts/`. | Record. |
| `old-training-scripts/` | Pre-2025 training scripts, including artificial-terrain and patch training. | Record. |
| `experiment-configs/` | YAML configs for the runners above. | Record. |

## launchers/ — experiment launchers

Records of how the e20–e30 experiments and their datasets were run. They call
`train_single_terrain_case.py`, `dataset/dataset.py` and
`dataset/generate-test-dataset.py` with the CLI of commit `742ba61`, and several
hardcode paths on other machines. The flags they pass are the best record of
each experiment's settings.

| Files | Experiment |
|---|---|
| `run_phase1_gnn.sh` | Phase 1 (single-stage) launcher per Norway resolution. |
| `e20_*.sbatch`, `e21_*.sbatch` | e20/e21: all resolutions, with Phase 2. |
| `e23*`, `dev_e23_tg_sgt_test.sbatch` | e23: terrain graph 2 and SparseGT tuning. |
| `e24*` | e24: Philadelphia transferability data and training. |
| `e25TR_*` – `e30TR_*` | e25–e30: hybrid sampling, number of pairs, full test sets, Philly→Norway transfer. |
| `smaller_*.sh` | Data generation for the smaller test and transferability sets. |
| `sparse_gt_gridsearch.sh`, `sweep.sh` | Grid searches over SparseGT settings and TAGConv learning rate. |

## sweeps/ — the 6 original sweep agents

`sweep_{gnn,sparse_gt_data,sparse_gt_random}_{norway,philly}.py` are merged
into `spnn/cli/sweep.py`, one preset per script. `tests/test_sweep_configs.py`
checks that each preset passes the same arguments the script passed.
Usefulness: record.

## wandb_tools/ — one-off W&B maintenance

| File | What it does |
|---|---|
| `update_wandb_configs.py` | Back-fills run configs from logged CLI arguments. |
| `backfill_wandb_metrics.py` | Back-fills summary metrics from history means. Note: it averages sampled history, not the full history. |
| `update_wandb_tags.py` | Replaces run tags with `phase1`/`phase2` tags from the run name. |

Usefulness: unlikely.
