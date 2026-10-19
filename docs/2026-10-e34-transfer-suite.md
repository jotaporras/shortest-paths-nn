# e34: full transferability suite on Norway (2026-10-15)

**Status:** done. All 90 runs finished 2026-10-16 13:57 EDT on plaza. Figures: `notebooks/e34_transfer_suite.ipynb`.

Transferability means: train on a coarse sample of the terrain, test on the fine stride-4 grid of the same terrain.
This is the e29 protocol of the previous paper, with the e31 sampling schemes added and the models
fixed by the e32 and e33 studies.

## 1. Decisions (JP, 2026-10-15)

- Models: MLP, TAGConv K=3, SparseGT k=2. Dropped: SparseGT k=6 (equal to k=2 at 4.7× the cost, e32),
  the edge-length variants (no gain, e33), GAT+L1 (kept as the reference number from the Chen replication).
- SparseGT trains for 200 epochs, MLP and TAGConv for 100.
- Suite C plus stride 4: 5 schemes × strides 32, 24, 16, 12, 8, 4 × 3 models = 90 runs. Stride 4 was
  first left out (in-distribution for the grid, 250k nodes, ~5 h per SparseGT run) and added back on JP's
  request as the end point of the curve, queued last.
- Test set: `~/data/spnn/full_test-004.npz` (stride-4 grid, 250k nodes, 2.5M pairs), the same as e29, e31, e32, e33.

## 2. Setup

| Item | Value |
| --- | --- |
| Training sets | `~/data/spnn/e31/norway_<scheme>_s<stride>.npz`, 1,000 sources × 500 targets, unweighted 3D edge length, kNN-8 for irregular schemes |
| New sets for e34 | strides 24, 12 and 4 for the 5 schemes, generated 2026-10-15 with `gen_s12_s24.sh` and `gen_s04.sh` in the data folder, same flags as e31 |
| Recipe | e32: batch 8192, AdamW weight decay 0.1, warmup 300 steps, constant lr, 25k validation pairs, fast loader, siamese L1 head, MSE |
| Learning rate | SparseGT 3.5e-3 and MLP 3.5e-3, TAGConv 1e-3 (MLP and TAGConv keep the e31 pilot choice; they were not tuned under the e32 recipe) |
| Configs | `configs/e31-mlp.yml`, `configs/e31-tagconv.yml`, `configs/e32-sparse-gt-k2.yml` |
| Launcher | `scripts/e34_transfer_suite.sh <gpu>`; two SparseGT workers (GPU 0, GPU 1) and one light-model worker on GPU 1 |
| Outputs | `~/runs/e34/<run>.json`, `<run>.log`, preds under `<run>/models`; wandb `alelab/terrains`, tags `e34`, `e34-<scheme>`, `e34-s<stride>`, `e34-<model>` |

Cost estimate from e32 timings (15 min at stride 16, 44 min at stride 8 per 100 SparseGT epochs): SparseGT about
15 GPU-h for strides 32–8, the light models about 4 GPU-h. A 2-epoch pilot of SparseGT k=2 on the stride-4 grid
(`~/runs/e34/pilot_s04/`, 2026-10-15 14:01) measured 2.1 min per epoch and about 31 GB of GPU memory, so each
stride-4 SparseGT run takes about 7 h and the five of them 36 GPU-h. Whole suite: about 55 GPU-h, 28 h on two GPUs.

## 3. Open issue carried from e31

The test ground truth is the 8-neighbour grid metric. Irregular kNN graphs are isotropic, so their models lose
about 1 point to the octile anisotropy (e31 section 7). A 16-neighbour stride-4 test set (e31 plan item D1)
is not part of e34; the numbers stay comparable with e31.

## 4. Results

Test nmae in %, MLP / TAGConv / SparseGT k=2. Full table: `python3 ~/runs/e34/analysis/summary.py`.

| Scheme | s32 | s24 | s16 | s12 | s8 | s4 |
| --- | --- | --- | --- | --- | --- | --- |
| grid | 0.83 / 1.04 / 0.87 | 0.75 / 0.97 / 0.70 | 0.70 / 0.83 / 0.65 | 0.69 / 0.79 / 0.60 | 0.69 / 0.77 / 0.62 | 0.71 / 0.83 / 0.62 |
| uniform | 1.81 / 2.11 / 2.02 | 1.74 / 1.83 / 1.85 | 1.76 / 1.75 / 1.66 | 1.76 / 1.79 / 1.72 | 1.69 / 1.73 / 1.62 | 1.67 / 1.74 / 1.63 |
| poisson-disk | 2.52 / 2.47 / 2.54 | 2.46 / 2.41 / 2.57 | 2.39 / 2.38 / 2.38 | 2.23 / 2.36 / 2.34 | 2.24 / 2.41 / 2.38 | 2.20 / 2.28 / 2.25 |
| slope-adaptive | 1.64 / 1.91 / 1.98 | 1.98 / 2.08 / 2.47 | 1.83 / 1.79 / 1.84 | 1.81 / 1.80 / 1.71 | 1.76 / 1.75 / 1.64 | 1.71 / 1.69 / 1.62 |
| quadtree | 19.8 / 19.9 / 19.7 | 19.8 / 19.8 / 19.8 | 20.2 / 20.0 / 20.0 | 20.2 / 20.0 / 20.0 | 20.1 / 20.0 / 19.9 | 20.0 / 20.0 / 19.9 |

Findings:
- On the grid, transfer saturates at stride 12 (28k nodes). SparseGT trained there equals SparseGT trained on the 250k-node test graph.
- The irregular schemes are flat at every density, also at 250k nodes. The models carry the metric of their training graph.
  The uniform error heatmap (notebook section 3) shows it: overestimation along the 0°, 45° and 90° grid directions, underestimation between them.
- MLP is within 0.1 of SparseGT at every grid stride; TAGConv is worse than MLP. One seed.

### 4.1 The slope-adaptive stride-24 outlier is a sampling-seed effect

All three models are worse on the seed-0 slope-adaptive stride-24 set than on its neighbours at strides 32 and 16.
The graph statistics of that set (degree, edge lengths, connectivity, path lengths) are in line with the other strides.
I generated a second point set with `--seed 1`, same flags, and ran the three models on it on 2026-10-16
(`~/runs/e34/seed1/`, wandb tag `e34seed1`). Test nmae in %:

| Seed | MLP | TAGConv | SparseGT k=2 |
| --- | --- | --- | --- |
| 0 (suite) | 1.98 | 2.08 | 2.47 |
| 1 | 1.75 | 1.89 | 1.82 |
| strides 32 / 16, seed 0 | 1.64 / 1.83 | 1.91 / 1.79 | 1.98 / 1.84 |

Seed 1 lands on the curve, so the outlier comes from the sampled point set and not from the stride. The spread between
the two seeds, 0.2 to 0.65 points, is of the same size as the differences between models and between strides on the
irregular schemes, so one seed cannot rank models or strides on the irregular schemes. The grid has no sampling seed.
