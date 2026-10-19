# e31: sampling-scheme transferability on Norway (2026-10-13)

**Status:** design agreed with JP on 2026-10-13. Data generated and pilot done on 2026-10-13. The 60 main runs ran on plaza from 17:31 on 2026-10-13 to 04:46 on 2026-10-14. Results are in section 6, the assessment in section 7 and the improvement plan in section 8. The plan is a draft for revision with JP.

## 1. Question

Does the node-sampling scheme of a coarse training graph change how well a model transfers to the fine grid? Does the SparseGT attention window (num_hops k) matter, and does it act through hop count or physical radius?

## 2. Design

JP fixed the factors: three models (GNN, SparseGT, MLP), all node-sampling schemes in `spnn/graphs/point_sampling.py`, SparseGT with k = 2 and k = 6, Norway only, and a one-day budget on plaza.

| Factor | Levels |
| --- | --- |
| Scheme | grid, uniform, Poisson-disk, slope-adaptive (0.5 uniform mix), quadtree |
| Partition (training graph) | stride 32: 3,969 nodes; stride 16: 15,625; stride 8: 62,500 |
| Model | MLP, TAGConv, SparseGT k=2, SparseGT k=6 |

60 runs, one seed (`--trial 1`).

Fixed across runs:
- **DEM:** `norway-smallest.txt`, 2000 × 2000 cells of 10 m (20 km × 20 km).
- **Test set:** `full_test-004.npz`, the stride-4 grid (250,000 nodes, 2.5M pairs, 10 sources × all nodes). The ground truth depends on the graph, so every run uses this one test graph.
- **Node count:** every scheme matches the node count of the grid at the same stride. Poisson-disk distance and quadtree threshold are calibrated to it, so they are within a few percent.
- **Edges:** grid uses its 8-neighbour edges. Uniform, Poisson-disk and slope-adaptive use symmetrized kNN with k = 8 in xy. kNN-8 equals the 8-neighbour grid 3 or more cells from the domain edge (`tests/test_schemes.py`), so these schemes differ from the grid only in node positions. Quadtree uses leaf sides. Small components are bridged to the main one by the shortest xy edge; the build fails if they hold more than 1% of the nodes. JP chose kNN over Delaunay, because Delaunay adds km-long convex-hull edges and has degree 6 instead of 8.
- **Edge weight:** unscaled 3D length (km) on both training and test graphs.
- **Training pairs:** 1,000 sources × 500 random targets (`single-source-random`), seed 0.
- **Input features:** node xyz in km. No other features.
- **Training:** siamese L1 head (`--p 1`), MSE loss, AdamW, minibatch 16,384 pairs (31 steps per epoch), 100 epochs = 3,100 steps, best-validation checkpoint on 0.5% of the training pairs. Learning rate per model, from the pilot in section 5: 3.5e-3 for MLP and SparseGT, 1e-3 for TAGConv. The design first had 1e-3 for all.

Models (all 128-dim node embeddings):

| Model | Config | Notes |
| --- | --- | --- |
| MLP | `configs/e31-mlp.yml` | 3 layers on xyz; no graph |
| TAGConv | `configs/e31-tagconv.yml` | 3 layers, K = 3 hops per layer |
| SparseGT k=2 | `configs/e31-sparse-gt-k2.yml` | `sparse-gt-data-norway.yml` with num_hops 2; RPEARL data mode |
| SparseGT k=6 | `configs/e31-sparse-gt-k6.yml` | same with num_hops 6 |

Receptive field of SparseGT: 6 layers × k hops.

| Partition | Spacing | k=2 radius (12 hops) | k=6 radius (36 hops) |
| --- | --- | --- | --- |
| stride 32 | 320 m | 3.8 km | 11.5 km |
| stride 16 | 160 m | 1.9 km | 5.8 km |
| stride 8 | 80 m | 1.0 km | 2.9 km |
| test, stride 4 | 40 m | 0.5 km | 1.4 km |

## 3. How to run

Data (CPU, about 5 minutes per 62.5k-node set):

```bash
python -m spnn.cli.make_dataset --name norway --raw-data ~/data/spnn/norway-smallest.txt \
    --filename ~/data/spnn/e31/norway_<scheme>_s<stride>.npz --graph-resolution <stride> \
    --dataset-size 500000 --num-sources 1000 --sampling-technique single-source-random \
    --scheme <scheme> --seed 0
```

Each set has a `.json` sidecar with N, E, mean degree, edge lengths, bridges and the calibrated parameter. The generator script is `~/data/spnn/e31/gen.sh` on plaza.

Training (plaza, three workers on one shared queue). MLP and TAGConv epochs take about 8 s even on 3,969 nodes. The e32 profiling (2026-10-14, `docs/2026-10-e32-sgt-tuning.md`) found the cause: the PyTorch DataLoader collates 16,384 scalar tensors one pair at a time, about 0.2 s per batch, so a SparseGT k=2 step ran at 0.34 s against 0.13 s of GPU work. The new `PairBatchLoader` in `spnn/data.py` (`--fast-loader`) removes this. In e31 the light models ran in a third worker that shares GPU 1:

```bash
MODELS="sgt6 sgt2" setsid nohup bash scripts/e31_sampling_transfer.sh 0 > ~/runs/e31/worker0.log 2>&1 < /dev/null &
MODELS="sgt6 sgt2" setsid nohup bash scripts/e31_sampling_transfer.sh 1 > ~/runs/e31/worker1.log 2>&1 < /dev/null &
MODELS="tagconv mlp" setsid nohup bash scripts/e31_sampling_transfer.sh 1 > ~/runs/e31/worker1_light.log 2>&1 < /dev/null &
```

Outputs: `~/runs/e31/<run>.json` (final and per-epoch validation metrics), `<run>.log`, and `<run>/models/.../preds.csv.gz` (test predictions, for detour bins). wandb tags `e31`, `e31-<scheme>`, `e31-s<stride>`, `e31-<model>`.

The runs were logged to wandb project `alelab/manifold-transformers-dev` by mistake. The launcher copied that default from `tests/smoke/smoke_test.sh`. JP's rule (2026-10-14): shortest-path runs go to `alelab/terrains`; `manifold-transformers-dev` is for graph benchmarks. The launcher default and a re-sync of the 60 runs with `wandb sync --project terrains` are pending.

## 4. Budget

Measured on plaza on 2026-10-13 at 62,500 nodes, 4,096 pairs per step:

| Model | s per step | Peak GPU memory |
| --- | --- | --- |
| SparseGT k=2 | 2.8 | 6.5 GB |
| SparseGT k=6 | 4.5 | 26.9 GB |

The forward on the whole training graph dominates each step, so cost scales with N and step count, not batch size. The k-hop window build takes 0.04 s at k=6, so caching it would not help. The k=6 window has 165 entries per node against 25 at k=2.

Estimated wall clock: SparseGT runs total about 44 GPU-hours, so about 22 h on two GPUs. The light worker needs about 8 h and runs alongside.

## 5. Data check and pilot

All 15 training graphs are connected before bridging, so no bridge was added. N is within 0.2% of the grid count.

| Scheme | Mean degree | Median edge, stride 8 |
| --- | --- | --- |
| grid | 7.8–8.0 | 113 m |
| uniform | 9.2 | 102 m |
| Poisson-disk | 8.6 | 105 m |
| slope-adaptive | 9.3 | 101 m |
| quadtree | 3.6–3.7 | 63 m |

Quadtree edges are long on flat ground (max 632 m at stride 8), because a flat leaf spans many cells.

Pilot: uniform, stride 32, 30 epochs, at two learning rates (output folders `~/runs/e31_pilot_lr1e-3` and `~/runs/e31_pilot_lr3.5e-3`). The rate with the lower validation nmae was chosen per model.

| Model | Val nmae, lr 1e-3 | Val nmae, lr 3.5e-3 | Test nmae at the chosen rate |
| --- | --- | --- | --- |
| MLP | 2.29% | 1.78% | 2.22% |
| TAGConv | 2.75% | 10.35% | 2.77% |
| SparseGT k=2 | 15.91% | 5.58% | 4.65% |

SparseGT at lr 1e-3 was still at 15% after 30 epochs, so it would be undertrained in 100 epochs. Its config was tuned at 3.5e-3. The MLP on xyz alone already beats octile+dz (2.40%) after 30 epochs.

## 6. Results

All 60 runs finished. Metrics are on the fixed test set, Norway stride 4, 250,000 nodes, 2.5M pairs. nmae is the mean of |pred − true| / true over pairs with true > 0, the same as Chen's mean relative error. Each run reports the checkpoint with the best validation nmae.

Test nmae / accuracy at 2%:

| Scheme | Stride | MLP | TAGConv | SparseGT k=2 | SparseGT k=6 |
| --- | --- | --- | --- | --- | --- |
| grid | 32 | **0.89% / 91.7%** | 1.48% / 78.7% | 1.11% / 85.0% | 1.24% / 79.2% |
| grid | 16 | **0.77% / 93.3%** | 1.09% / 86.8% | 1.00% / 89.0% | 1.03% / 87.9% |
| grid | 8 | **0.72% / 93.1%** | 1.15% / 86.6% | 1.04% / 85.9% | 1.01% / 88.2% |
| uniform | 32 | 1.96% / 61.7% | 2.43% / 55.5% | 1.95% / 61.8% | 1.95% / 61.1% |
| uniform | 16 | 1.80% / 67.2% | 1.98% / 63.4% | 1.75% / 66.5% | 1.87% / 63.0% |
| uniform | 8 | 1.74% / 66.0% | 1.89% / 64.6% | 1.78% / 64.6% | 1.75% / 65.7% |
| poisson-disk | 32 | 2.42% / 41.9% | 2.80% / 40.5% | 2.88% / 34.5% | 3.11% / 30.7% |
| poisson-disk | 16 | 2.27% / 45.1% | 2.68% / 40.4% | 2.38% / 42.2% | 2.51% / 40.6% |
| poisson-disk | 8 | 2.27% / 45.4% | 2.71% / 40.5% | 2.76% / 37.2% | 2.45% / 40.8% |
| slope-adaptive | 32 | 1.84% / 65.8% | 2.27% / 59.2% | 1.75% / 65.7% | 1.68% / 68.0% |
| slope-adaptive | 16 | 1.93% / 64.5% | 2.37% / 58.0% | 1.89% / 62.5% | 1.78% / 65.1% |
| slope-adaptive | 8 | 1.77% / 67.8% | 2.06% / 61.8% | 1.75% / 65.7% | 1.78% / 64.9% |
| quadtree | 32 | 20.01% / 2.0% | 20.08% / 2.7% | 19.71% / 7.2% | 19.73% / 7.3% |
| quadtree | 16 | 20.32% / 1.1% | 20.51% / 1.9% | 19.82% / 6.5% | 20.17% / 6.0% |
| quadtree | 8 | 20.35% / 1.5% | 20.01% / 2.8% | 20.10% / 4.5% | 20.24% / 4.2% |

Reference points on the same test set, from `docs/2026-10-results-analysis.md` section 7:

| Reference | nmae | acc<2% |
| --- | --- | --- |
| octile+dz, no training | 2.40% | 47.1% |
| octile+dz × c, one scalar on 100 pairs | 1.12% | 86.2% |
| GAT+L1 replication of Chen, grid stride 8, 500 minibatch epochs | 0.52% | 96.5% |
| Chen, Coarse-GAT+L1, paper Table 3, own test graph | 1.05% | 89.2% |

Observations:
- Only the grid scheme beats the one-scalar formula. On grid the MLP is the best model at every stride, and no e31 model reaches the GAT replication.
- Uniform, Poisson-disk and slope-adaptive sit at 1.7–3.1%, worse than the formula. More training nodes help little: from stride 32 to 8 the curves fall by at most a few tenths of a percent.
- SparseGT k=2 and k=6 are within noise of each other everywhere. k=6 is slightly better on grid stride 8 and slope-adaptive, slightly worse on Poisson-disk. One seed, so no conclusion on the window.
- Quadtree is at 20% for every model. Section 7 explains why.

Run times at stride 8: TAGConv about 13 min, MLP about 12 min, SparseGT k=2 about 27 min, SparseGT k=6 about 71 min. The whole set took 11 h 15 min on two GPUs.

## 7. Assessment (2026-10-14)

JP's question was whether the graph models overfit, since every curve is above the MLP and above Chen's numbers. They do not overfit. Three things explain the numbers: the runs are short, SparseGT training is unstable, and on the irregular schemes the training graph and the test graph measure different distances.

### 7.1 Train error equals test error

An overfit model has a low train error and a high test error. On grid, train, validation and test nmae are the same for every model. The graph models do not fit even their own training pairs as well as the MLP does.

| Grid stride 8 | Train nmae | Best val nmae | Test nmae |
| --- | --- | --- | --- |
| MLP | 0.74% | 0.70% | 0.72% |
| TAGConv | 1.11% | 1.06% | 1.15% |
| SparseGT k=2 | 1.08% | 1.03% | 1.04% |
| SparseGT k=6 | 1.03% | 0.99% | 1.01% |
| GAT+L1 replication | 0.46% | — | 0.52% |

### 7.2 The runs are 20× shorter than the replication

e31 takes 3,100 optimizer steps (batch 16,384, 100 epochs). The GAT replication took 61,000 (batch 4,096, 500 epochs). The validation curves are still falling at epoch 100, and the best epoch is 85–99 for most runs. Validation nmae on grid stride 8:

| Model | Epoch 0 | Epoch 40 | Epoch 80 | Best |
| --- | --- | --- | --- | --- |
| MLP | 6.4% | 1.07% | 0.90% | 0.70% |
| TAGConv | 15.0% | 1.43% | 1.13% | 1.06% |
| SparseGT k=2 | 112% | 3.43% | 1.23% | 1.03% |
| SparseGT k=6 | 109% | 3.44% | 1.63% | 0.99% |

### 7.3 SparseGT training is unstable

SparseGT starts at 110% error, so the initial scale of the L1 head is far off, and it spends about 40 epochs on the scale alone. The optimizer is AdamW at a constant 3.5e-3 with no warmup and no schedule. The validation curve spikes: grid stride 32 k=6 ends at 55% after a best of 0.88%, and quadtree stride 16 k=2 has a standard deviation of 37% over the last 10 epochs. The model also has dropout 0.5 and attention dropout 0.2, which is heavy for a regression of a deterministic function. Chen's GAT has no dropout. The best-validation checkpoint hides the spikes, but it is picked on 2,500 pairs, so the choice is noisy.

### 7.4 Irregular schemes: a metric mismatch, not a model error

Each training graph defines its own shortest-path distance. The ratio of path length to 3D Euclidean distance over the training pairs shows that the schemes do not agree with the test grid:

| Graph | Length / Euclidean, mean | Length / (octile+dz), mean |
| --- | --- | --- |
| Test grid, stride 4 | 1.080 | 1.025 |
| Grid, stride 8 | 1.082 | 1.026 |
| Uniform kNN-8 | 1.087 | 1.032 |
| Slope-adaptive kNN-8 | 1.091 | 1.036 |
| Poisson-disk kNN-8 | 1.063 | 1.009 |
| Quadtree, leaf edges | 1.304 | 1.237 |

The models learn their training metric well: the uniform stride-8 MLP has 0.99% train nmae and 1.74% test nmae, and the quadtree stride-8 MLP has 0.65% train nmae and 20.35% test nmae. The gap is the disagreement between the two graphs, not a failure to fit.

Median bias of the test predictions at stride 8, and the nmae after dividing the predictions by that median:

| Scheme | MLP | TAGConv | SparseGT k=2 | SparseGT k=6 |
| --- | --- | --- | --- | --- |
| grid | −0.06% → 0.72% | +0.24% → 1.12% | +0.01% → 1.04% | −0.09% → 1.01% |
| uniform | +0.12% → 1.74% | +0.07% → 1.89% | +0.20% → 1.77% | −0.18% → 1.75% |
| poisson-disk | −2.05% → 1.68% | −2.34% → 1.98% | −2.68% → 1.80% | −2.25% → 1.77% |
| slope-adaptive | +0.48% → 1.72% | +0.56% → 2.00% | +0.03% → 1.75% | +0.44% → 1.74% |
| quadtree | +20.56% → 7.90% | +20.30% → 7.95% | +20.40% → 8.19% | +20.67% → 7.97% |

Two findings:
- Poisson-disk is biased 2% short, and a single scalar brings it level with uniform. Quadtree is biased 20% long, because its leaf-side edges give Manhattan-like paths. A scalar leaves 8%, because the Manhattan pattern differs by direction from the grid's.
- After calibration all three kNN schemes stay at about 1.7%, against 0.7% for grid. The 8-neighbour grid has octile anisotropy: its paths are up to 8% longer than Euclidean depending on direction. A kNN graph on random points is isotropic. A model trained on an isotropic graph cannot learn the grid's directional pattern, so the test set favours the grid scheme by construction.

Consequence for the question in section 1: the fixed grid test set cannot rank sampling schemes fairly. It measures how close each scheme's metric is to the octile grid, not how well a model transfers.

### 7.5 What the graph models lack against Chen's GAT

Chen's GAT uses the edge weight inside attention. TAGConv and SparseGT in e31 receive the edge attribute and ignore it (`spnn/models/gnn.py:9`, `spnn/models/sparse_gt.py:598`). Their only information beyond the MLP's xyz is adjacency. On one terrain, xyz alone is enough for a smooth correction on top of the near-Euclidean distance, which is why the MLP wins.

### 7.6 Error by detour and distance bin (Opus, 2026-10-14)

Opus computed these tables on the plaza CPU from the saved test predictions of the 20 stride-8 runs. Script: `~/runs/e31/analysis/bins.py`. Raw numbers in long format: `~/runs/e31/analysis/bins_s08.csv`. The calibration numbers are in `~/runs/e31/analysis/calib_s08.csv`.

Method:
- Pairs: the 2,499,990 test pairs with true > 0. The 10 pairs with src = tar are dropped.
- nmae: mean of |pred − true| / true over the pairs in a bin, as in section 6.
- Detour: true / octile+dz, with octile+dz from the xyz of src and tar. Bins as in `docs/2026-10-results-analysis.md` section 7.
- Straight-line pairs have a detour up to 1e-7 below 1.00 because preds.csv stores float32 lengths. The script counts them in [1.00, 1.02). That bin then holds 45.9% instead of 45.8%.
- The octile+dz column is the formula itself, with no training. It is the same for every scheme.
- The "all" row repeats the test nmae of section 6.

grid, detour bin:

| Detour bin | Share | octile+dz | MLP | TAGConv | SparseGT k=2 | SparseGT k=6 |
| --- | --- | --- | --- | --- | --- | --- |
| [1.00,1.02) | 45.9% | 1.24% | 0.69% | 1.34% | 0.88% | 0.75% |
| [1.02,1.05) | 46.9% | 3.01% | 0.64% | 0.84% | 0.87% | 0.91% |
| [1.05,1.10) | 7.2% | 5.78% | 1.40% | 1.91% | 3.13% | 3.30% |
| [1.10,1.30) | 0.1% | 10.44% | 5.67% | 7.08% | 7.27% | 7.57% |
| **all** | 100.0% | 2.40% | 0.72% | 1.15% | 1.04% | 1.01% |

grid, distance bin:

| Distance | Share | octile+dz | MLP | TAGConv | SparseGT k=2 | SparseGT k=6 |
| --- | --- | --- | --- | --- | --- | --- |
| 0–2 km | 2.4% | 1.75% | 3.54% | 6.88% | 1.96% | 1.84% |
| 2–5 km | 9.9% | 2.30% | 1.71% | 2.56% | 1.59% | 1.52% |
| 5–10 km | 25.4% | 2.57% | 0.83% | 1.29% | 1.36% | 1.30% |
| 10–20 km | 54.3% | 2.42% | 0.42% | 0.67% | 0.82% | 0.81% |
| 20–30 km | 8.0% | 2.12% | 0.36% | 0.46% | 0.59% | 0.59% |
| **all** | 100.0% | 2.40% | 0.72% | 1.15% | 1.04% | 1.01% |

uniform, detour bin:

| Detour bin | Share | octile+dz | MLP | TAGConv | SparseGT k=2 | SparseGT k=6 |
| --- | --- | --- | --- | --- | --- | --- |
| [1.00,1.02) | 45.9% | 1.24% | 1.67% | 1.82% | 1.56% | 1.54% |
| [1.02,1.05) | 46.9% | 3.01% | 1.73% | 1.88% | 1.94% | 1.89% |
| [1.05,1.10) | 7.2% | 5.78% | 2.18% | 2.37% | 2.14% | 2.21% |
| [1.10,1.30) | 0.1% | 10.44% | 4.15% | 3.48% | 4.51% | 4.60% |
| **all** | 100.0% | 2.40% | 1.74% | 1.89% | 1.78% | 1.75% |

uniform, distance bin:

| Distance | Share | octile+dz | MLP | TAGConv | SparseGT k=2 | SparseGT k=6 |
| --- | --- | --- | --- | --- | --- | --- |
| 0–2 km | 2.4% | 1.75% | 4.18% | 5.90% | 2.65% | 2.54% |
| 2–5 km | 9.9% | 2.30% | 2.43% | 2.87% | 2.22% | 2.20% |
| 5–10 km | 25.4% | 2.57% | 1.75% | 1.97% | 1.92% | 1.90% |
| 10–20 km | 54.3% | 2.42% | 1.54% | 1.55% | 1.59% | 1.58% |
| 20–30 km | 8.0% | 2.12% | 1.48% | 1.52% | 1.83% | 1.71% |
| **all** | 100.0% | 2.40% | 1.74% | 1.89% | 1.78% | 1.75% |

poisson-disk, detour bin:

| Detour bin | Share | octile+dz | MLP | TAGConv | SparseGT k=2 | SparseGT k=6 |
| --- | --- | --- | --- | --- | --- | --- |
| [1.00,1.02) | 45.9% | 1.24% | 2.59% | 3.02% | 2.81% | 2.38% |
| [1.02,1.05) | 46.9% | 3.01% | 2.00% | 2.40% | 2.63% | 2.44% |
| [1.05,1.10) | 7.2% | 5.78% | 1.94% | 2.67% | 3.18% | 2.96% |
| [1.10,1.30) | 0.1% | 10.44% | 5.36% | 6.38% | 6.96% | 6.68% |
| **all** | 100.0% | 2.40% | 2.27% | 2.71% | 2.76% | 2.45% |

poisson-disk, distance bin:

| Distance | Share | octile+dz | MLP | TAGConv | SparseGT k=2 | SparseGT k=6 |
| --- | --- | --- | --- | --- | --- | --- |
| 0–2 km | 2.4% | 1.75% | 3.42% | 7.00% | 3.16% | 3.01% |
| 2–5 km | 9.9% | 2.30% | 2.43% | 3.77% | 3.15% | 2.94% |
| 5–10 km | 25.4% | 2.57% | 2.27% | 2.71% | 2.99% | 2.70% |
| 10–20 km | 54.3% | 2.42% | 2.21% | 2.39% | 2.62% | 2.28% |
| 20–30 km | 8.0% | 2.12% | 2.12% | 2.28% | 2.33% | 2.07% |
| **all** | 100.0% | 2.40% | 2.27% | 2.71% | 2.76% | 2.45% |

slope-adaptive, detour bin:

| Detour bin | Share | octile+dz | MLP | TAGConv | SparseGT k=2 | SparseGT k=6 |
| --- | --- | --- | --- | --- | --- | --- |
| [1.00,1.02) | 45.9% | 1.24% | 1.65% | 2.05% | 1.52% | 1.56% |
| [1.02,1.05) | 46.9% | 3.01% | 1.79% | 2.01% | 1.91% | 1.92% |
| [1.05,1.10) | 7.2% | 5.78% | 2.31% | 2.44% | 2.18% | 2.22% |
| [1.10,1.30) | 0.1% | 10.44% | 3.81% | 4.48% | 4.57% | 4.10% |
| **all** | 100.0% | 2.40% | 1.77% | 2.06% | 1.75% | 1.78% |

slope-adaptive, distance bin:

| Distance | Share | octile+dz | MLP | TAGConv | SparseGT k=2 | SparseGT k=6 |
| --- | --- | --- | --- | --- | --- | --- |
| 0–2 km | 2.4% | 1.75% | 4.53% | 7.36% | 2.56% | 2.62% |
| 2–5 km | 9.9% | 2.30% | 2.62% | 3.40% | 2.17% | 2.17% |
| 5–10 km | 25.4% | 2.57% | 1.79% | 2.04% | 1.87% | 1.90% |
| 10–20 km | 54.3% | 2.42% | 1.52% | 1.65% | 1.58% | 1.60% |
| 20–30 km | 8.0% | 2.12% | 1.43% | 1.65% | 1.77% | 1.85% |
| **all** | 100.0% | 2.40% | 1.77% | 2.06% | 1.75% | 1.78% |

quadtree, detour bin:

| Detour bin | Share | octile+dz | MLP | TAGConv | SparseGT k=2 | SparseGT k=6 |
| --- | --- | --- | --- | --- | --- | --- |
| [1.00,1.02) | 45.9% | 1.24% | 22.69% | 22.41% | 22.78% | 23.09% |
| [1.02,1.05) | 46.9% | 3.01% | 18.97% | 18.63% | 18.59% | 18.61% |
| [1.05,1.10) | 7.2% | 5.78% | 14.55% | 13.82% | 12.96% | 12.81% |
| [1.10,1.30) | 0.1% | 10.44% | 14.08% | 14.20% | 12.08% | 12.17% |
| **all** | 100.0% | 2.40% | 20.35% | 20.01% | 20.10% | 20.24% |

quadtree, distance bin:

| Distance | Share | octile+dz | MLP | TAGConv | SparseGT k=2 | SparseGT k=6 |
| --- | --- | --- | --- | --- | --- | --- |
| 0–2 km | 2.4% | 1.75% | 23.55% | 22.38% | 20.28% | 20.52% |
| 2–5 km | 9.9% | 2.30% | 21.38% | 20.08% | 19.81% | 20.08% |
| 5–10 km | 25.4% | 2.57% | 19.60% | 19.13% | 19.08% | 19.25% |
| 10–20 km | 54.3% | 2.42% | 19.38% | 19.25% | 19.49% | 19.63% |
| 20–30 km | 8.0% | 2.12% | 27.12% | 27.20% | 27.79% | 27.62% |
| **all** | 100.0% | 2.40% | 20.35% | 20.01% | 20.10% | 20.24% |

One-scalar calibration of plan item D2, raw nmae → calibrated nmae. The script draws 100 pairs with `np.random.default_rng(0)`, fits c = median(true / pred) on them, and scores c × pred on all pairs. Section 7.4 fitted c on all pairs instead.

| Scheme | MLP | TAGConv | SparseGT k=2 | SparseGT k=6 |
| --- | --- | --- | --- | --- |
| grid | 0.72% → 0.72% | 1.15% → 1.12% | 1.04% → 1.04% | 1.01% → 1.02% |
| uniform | 1.74% → 1.74% | 1.89% → 1.92% | 1.78% → 1.77% | 1.75% → 1.75% |
| poisson-disk | 2.27% → 1.68% | 2.71% → 2.00% | 2.76% → 1.80% | 2.45% → 1.77% |
| slope-adaptive | 1.77% → 1.72% | 2.06% → 2.03% | 1.75% → 1.75% | 1.78% → 1.74% |
| quadtree | 20.35% → 8.33% | 20.01% → 8.26% | 20.10% → 8.58% | 20.24% → 8.31% |

Fitted c:

| Scheme | MLP | TAGConv | SparseGT k=2 | SparseGT k=6 |
| --- | --- | --- | --- | --- |
| grid | 1.000 | 0.998 | 0.999 | 1.000 |
| uniform | 0.997 | 1.004 | 0.998 | 1.002 |
| poisson-disk | 1.020 | 1.027 | 1.027 | 1.023 |
| slope-adaptive | 0.995 | 0.998 | 0.999 | 0.994 |
| quadtree | 0.852 | 0.849 | 0.851 | 0.848 |

Observations:
- On grid, SparseGT has 3.1–3.3% nmae in the [1.05, 1.10) detour bin, against 1.40% for the MLP.
- On pairs of 0–2 km outside quadtree, the MLP (3.4–4.5%) and TAGConv (5.9–7.4%) are worse than octile+dz (1.75%), while SparseGT is at 1.8–3.2%.
- On pairs of 10–30 km, the MLP has the lowest nmae of the four models in 7 of the 8 non-quadtree cells. The exception is Poisson-disk 20–30 km, where SparseGT k=6 has 2.07% against 2.12%.
- On the three kNN schemes, nmae stays at 1.5–3.2% up to detour 1.10, while octile+dz rises from 1.24% to 5.78%.
- In the [1.00, 1.02) bin, every model trained on a kNN scheme is worse than octile+dz (1.24%).
- The 100-pair calibration changes nmae by more than 0.1 points only on Poisson-disk (to 1.68–2.00%) and quadtree (to 8.26–8.58%).

## 8. Improvement plan (draft, 2026-10-14)

To be revised with JP, then handed to Opus. Order matters: D1 and E change what every later number means.

**A. Training budget and schedule, on grid first.** Largest expected gain. Cost per step is the full-graph forward, so SparseGT steps cost 2.8 s (k=2) and 4.5 s (k=6) at 62,500 nodes whatever the batch size. These step times include the DataLoader cost found in e32 (see section 3), so with `--fast-loader` the estimates below are upper bounds. Items A3 and A4 are implemented in e32 (`--warmup-steps`, `--lr-schedule`, `--weight-decay`, `--val-pairs`).
1. MLP and TAGConv: batch 4,096, 500 epochs, the replication's setting. About 1–2 h per run.
2. SparseGT: batch 8,192, 200 epochs, which is 12,000 steps. About 9 h for k=2 and 15 h for k=6 at stride 8. Tune at stride 16 first, where a step is about 4× cheaper.
3. All models: linear warmup of 500 steps, then cosine decay to 1e-4 of the peak. Peak 1e-3 for SparseGT and TAGConv, 3.5e-3 for the MLP. `spnn/training.py` has no scheduler today.
4. Validation on 25,000 pairs instead of 2,500.

**B. SparseGT regularization and init.**
1. Dropout 0.5 → 0.1, attention dropout 0.2 → 0.0.
2. A learnable output scale, initialized so the first predictions are near the Euclidean distance. This removes the 110% start.

**C. Give the graph models the edge lengths.**
1. Add node features: mean and max incident edge length. This tells the model its local resolution, a direct lever for transferability.
2. SparseGT: add the 3D edge length as an attention bias in `SparseKHopGraphAttention`. Larger change.

**D. Fix the test ground truth.** Needed before any claim about which scheme transfers better.
1. Build the stride-4 test set with 16 neighbours (knight-move edges). Anisotropy drops from 8.2% to 2.8%, and grid-trained models then face a mismatch too.
2. Report calibrated nmae alongside raw, with the scalar fitted on 100 pairs.
3. Quadtree: add diagonal edges across leaf corners, or kNN-8 on leaf centres, or drop the scheme.

**E. Analysis now, no GPU.** Detour-bin and distance-bin tables per scheme from the saved `preds.csv.gz`.

Suggested order: D1 and E, then A and B as a 2 h tuning round of 4 runs on grid stride 16, then C1, then rerun the 60-run grid with the winning settings.
