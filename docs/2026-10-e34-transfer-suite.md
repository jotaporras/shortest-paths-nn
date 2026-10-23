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

## 5. e34a: difficulty pair sampling on the grid (2026-10-19)

Question (JP): does the old `hybrid` pair sampler of e29 beat the uniform pairs of e34 on the grid? Hybrid draws half
of the sources with probability proportional to the elevation gradient and half of each source's targets with
probability 1/hops. Sets: `~/data/spnn/e34a/norway_grid_s16.npz` and `s12`, generated with `gen.sh` in that folder,
`--sampling-technique hybrid`, all other flags as e31. Runs: the e34 launcher with `DATA`, `DSET=e34a`, `OUT=~/runs/e34a`,
`TAG=e34a`, same recipe and epochs. Test nmae % / acc<2% %, uniform pairs (e34) → hybrid pairs (e34a):

| Grid stride | MLP | TAGConv | SparseGT k=2 |
| --- | --- | --- | --- |
| 16 | 0.70 / 93.9 → 0.70 / 94.0 | 0.83 / 91.3 → 0.81 / 92.1 | 0.65 / 94.6 → 0.62 / 95.4 |
| 12 | 0.69 / 93.9 → 0.70 / 93.9 | 0.79 / 92.3 → 0.77 / 92.5 | 0.60 / 95.7 → 0.62 / 95.6 |

No gain. The changes are at most 0.03 points in either direction, one seed each. The best SparseGT stays at 0.60 / 95.7
(uniform, stride 12), above the GAT+L1 replication at 0.52 / 96.5. The test pairs bend little (median detour ratio 1.02
over octile+dz), so 1/hops targets do not add hard pairs; they add short ones.

### 5.1 Train/validation curve audit

An Opus subagent audited the curves of all 93 runs (`~/runs/e34/analysis/curves_report.md`, figures `curves_*.pdf`).
No run overfits: train and best validation agree within -3% to +7% everywhere, and the test number comes from the
best-validation checkpoint. TAGConv has not converged in 30 of 31 runs (best epoch 87–99, best validation still falling 9%
over the last 25 epochs) and is the only model at lr 1e-3. MLP converges by epoch 40 and then oscillates at the constant
rate. SparseGT still gains about 1% per 20 epochs at epoch 200 and its validation spikes rise to 25% of the last 20
epochs. At an equal 100-epoch budget SparseGT's validation lead over MLP halves from 14% to 6.5%. Recommendation: same
step count for all models and a learning-rate decay, before the models are compared again.

### 5.2 e34b: cosine decay on SparseGT, grid stride 12 (2026-10-19)

One run, `~/runs/e34b/norway_grid_s12_sgt2_cosine.json` (launcher `run_cosine.sh` in that folder, wandb tag `e34b`):
the e34 grid stride-12 SparseGT k=2 run with `--lr-schedule cosine` (decay to 1e-4 × lr at the last step) instead
of constant. Everything else unchanged, 200 epochs.

| Schedule | Test nmae | acc<1% | acc<2% | Train nmae | Best val (epoch) | Val, last 20 epochs |
| --- | --- | --- | --- | --- | --- | --- |
| constant (e34) | 0.60 | 82.7 | 95.7 | 0.64 | 0.641 (189) | 0.64–1.87, spikes |
| cosine (e34b) | 0.62 | 82.4 | 95.1 | 0.63 | 0.627 (185) | 0.627–0.631, flat |

The decay removes the late validation spikes and lowers the best validation by 2%, but the test error does not
improve (0.60 → 0.62, within one-seed noise). The constant run's best checkpoint was a dip inside the spiky phase,
and it transfers as well as the cosine plateau. The schedule fixes stability, not transfer. On the grid, a
one-seed difference of 0.02 points cannot be read either way.

## 6. e34c: a test set per scheme, and the 90 × 5 cross evaluation (2026-10-19)

JP's hypothesis: the star-shaped error heatmaps and the grid's lead come from testing every model on the grid
metric. To test it without retraining, I built a full test set on the stride-4 graph of each irregular scheme
and evaluated all 90 e34 models on all 5 test sets.

- Test sets: `~/data/spnn/e34c/norway_<scheme>_test-004.npz`, built by `scripts/e34c_make_scheme_tests.py` from
  the e34 stride-4 training graphs (same 250k nodes, same kNN-8 edges and 3D lengths). Sources: the scheme node
  nearest in xy to each of the 10 grid sources, at most 45 m away (167 m on the quadtree). 10 sources × all nodes,
  about 2.5M pairs each. Quadtree pairs are 21% longer on average (14.56 km against 12.05 km), the Manhattan bias.
- Evaluation: `scripts/e34c_eval_cross.py`, which loads `final_model.pt`, embeds the test graph and predicts the L1
  distance. On the grid test it reproduces the e34 JSON nmae of all 90 runs to three decimals. 450 evaluations in
  6 min on two GPUs. Outputs `~/runs/e34c/metrics/<run>__<test>.json`, predictions `preds/*.npy` (float16),
  table `python ~/runs/e34c/summary.py [nmae|acc_2pct]`.

### 6.1 Matched test versus grid test, SparseGT k=2, test nmae in %

| Training scheme | s32 | s24 | s16 | s12 | s8 | s4 |
| --- | --- | --- | --- | --- | --- | --- |
| grid (its own test) | 0.87 | 0.70 | 0.65 | 0.60 | 0.62 | 0.62 |
| uniform, matched / grid test | 1.31 / 2.02 | 1.12 / 1.85 | 0.89 / 1.66 | 0.93 / 1.72 | 0.75 / 1.62 | 0.76 / 1.63 |
| poisson-disk, matched / grid test | 0.86 / 2.54 | 0.90 / 2.57 | 0.73 / 2.38 | 0.71 / 2.34 | 0.70 / 2.38 | 0.66 / 2.25 |
| slope-adaptive, matched / grid test | 1.27 / 1.98 | 1.84 / 2.47 | 1.20 / 1.84 | 0.88 / 1.71 | 0.80 / 1.64 | 0.73 / 1.62 |
| quadtree, matched / grid test | 0.63 / 19.7 | 0.57 / 19.8 | 0.53 / 20.0 | 0.51 / 20.0 | 0.51 / 19.9 | 0.49 / 19.9 |

MLP and TAGConv follow the same pattern (full tables from `summary.py`). Matched acc<2% for SparseGT: uniform
81–95%, poisson-disk 92–96%, slope-adaptive 72–94%, quadtree 95–97%, against 39–70% on the grid test.

Findings:
- The hypothesis holds. On its own test graph every scheme is within 0.3 points of the grid at strides 4–12, and
  the grid test added 0.9 points (uniform, slope-adaptive), 1.6 (poisson-disk) or 19 (quadtree) of pure metric
  mismatch. The grid's lead in section 4 was the test metric, not the sampling.
- The mismatch is symmetric. Grid-trained models get 1.7% on the uniform test and 2.1–2.2% on the poisson-disk
  test, the same as uniform- and poisson-disk-trained models get on the grid. Each model carries the metric of
  its training graph, whichever that is.
- Uniform and slope-adaptive are interchangeable: a model trained on one gets its matched error on the other
  (uniform s4 SparseGT: 0.76 on uniform, 0.78 on slope-adaptive). Poisson-disk is its own family, 2.0–2.2% from
  both. Its kNN-8 metric differs because its points are nearly equidistant.
- Quadtree is the easiest graph to fit, 0.49–0.63% matched, the lowest numbers of the suite. Its leaf-edge metric
  is axis-aligned and simple. That says nothing about fidelity to the terrain geodesic; see the caveat below.
- The irregular schemes now show a transfer curve: on the matched test, SparseGT improves from stride 32 to 4 by
  1.31 → 0.76 (uniform), 0.86 → 0.66 (poisson-disk), 0.63 → 0.49 (quadtree), where the grid goes 0.87 → 0.62 and
  saturates at stride 12. The slope-adaptive s24 seed outlier (section 4.1) is visible on every test set.
- The star pattern of the heatmaps is the mismatch. `docs/figures/e34c/heatmap_grid_vs_uniform_test.pdf` shows
  the uniform-trained SparseGT at source 239655: rays along 0°/45°/90° on the grid test, none on the uniform test;
  the grid-trained model shows the inverse rays on the uniform test.

Caveat: a matched test measures how well a model fits the metric of its own graph family. It does not say which
family is closest to the continuous terrain geodesic; the quadtree's 0.49% is a good fit of a Manhattan-like
metric. Ranking the schemes as approximations of the terrain needs a graph-independent ground truth (16-neighbour
grid or fast marching), the open item from e31.

## 7. e35: Laplacian eigenvectors instead of RPEARL (2026-10-20)

Purpose (JP): show that RPEARL transfers better than Laplacian eigenvector encodings (LapPE). The theory says the
transferability of this architecture depends on the eigengap, and LapPE exposes the eigenbasis directly.
A second output is the cost of computing the eigenvectors, which RPEARL does not pay.

### 7.1 Model and encodings

- Model `sgt2lap`: the e32 SparseGT k=2 architecture (6 layers, 128 hidden, 8 heads, attention support
  (A+I)^2) with `embedding_mode: lappe`, config `configs/e35-sparse-gt-k2-lappe.yml`. The RPEARL GCN (6 TAGConv
  layers on xyz) is replaced by one linear layer from the k eigenvector columns to the 128-dim embedding, added
  to the projected xyz as before. Everything else, recipe included, equals `sgt2`.
- Encodings (`spnn/graphs/spectral.py`): the k lowest non-trivial eigenvectors of the normalised Laplacian
  L = I − D^-1/2 A D^-1/2 of the unweighted, symmetrised graph without self loops, the operator TAGConv uses.
  ARPACK (`scipy.sparse.linalg.eigsh`, `which='LA'` on D^-1/2 A D^-1/2, tol 1e-6). Each vector is scaled by
  √n so its entries are O(1) on every graph size. The train and the test graph each get their own eigenvectors;
  nothing aligns them (signs, rotations in degenerate eigenspaces), which is the point of the experiment.
  Random sign flips per column in training (the standard augmentation), none at test time.
- Plumbing: `--rpearl-embedding-mode lappe --lappe-k K` in `spnn.cli.train`; the eigenvectors are appended to
  the node features and the model splits them off. Cache `~/data/spnn/lappe/lappe_<hash>.npz` keyed by the
  undirected edge set, so a training set and a test set on the same graph share one file.
  `python -m spnn.cli.make_lappe` precomputes and times them. `tests/test_lappe.py` (6 tests).
- Degeneracy on the grids: λ2 = λ3 to 1e-14 on grid s12 and on the stride-4 test grid (the two axis modes of a
  square domain), and more pairs follow (λ5 = λ6, λ7 = λ8, ...). Inside such a pair the eigsh basis is an
  arbitrary rotation, so the model cannot rely on a fixed meaning of column 2 versus column 3 across graphs.

### 7.2 Cost of the eigenvectors (plaza, 48 threads)

Recorded in the cache sidecars (`seconds`, `peak_rss_gb`, `host`, per-k `timings`) and copied into the wandb
config of every lappe run (`lappe_train_eig_seconds`, `lappe_test_eig_seconds`, eigenvalues, host).
The per-k timings below come from a second pass at nice 10 on 2026-10-21 (`~/runs/e35/retime_lappe.sh`) while
the two GPU jobs ran; the first pass (the `seconds` field, 64 vectors, with a concurrent eigsh) was within 2×.
Seconds per graph for k eigenvectors, peak RSS of the process, and the first non-trivial eigenvalue λ2 of the
normalised Laplacian with its gap to λ3. The stride-4 rows are also the five test graphs of e34c.

| stride | scheme | nodes | k=2 | k=8 | k=16 | k=32 | k=64 | peak RSS (GB) | λ2 | λ3 − λ2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 32 | grid | 3,969 | 0.0 | 0.1 | 0.1 | 0.1 | 7.7 | 0.08 | 9.56e-04 | 2.0e-15 |
| 32 | poisson-disk | 3,969 | 0.0 | 0.0 | 0.1 | 0.1 | 6.2 | 0.08 | 9.91e-04 | 1.3e-05 |
| 32 | quadtree | 3,972 | 0.0 | 0.1 | 0.1 | 0.1 | 9.1 | 0.07 | 4.94e-04 | 1.3e-04 |
| 32 | slope-adaptive | 3,969 | 0.0 | 0.1 | 0.1 | 0.1 | 7.7 | 0.08 | 7.57e-04 | 3.5e-05 |
| 32 | uniform | 3,969 | 0.0 | 0.0 | 0.1 | 0.1 | 8.4 | 0.08 | 7.72e-04 | 1.9e-05 |
| 24 | grid | 7,056 | 0.1 | 0.1 | 0.2 | 1.7 | 33.5 | 0.09 | 5.35e-04 | 2.8e-15 |
| 24 | poisson-disk | 7,043 | 0.1 | 0.1 | 0.1 | 1.8 | 27.7 | 0.09 | 5.56e-04 | 4.2e-06 |
| 24 | quadtree | 7,053 | 0.1 | 0.2 | 0.2 | 3.1 | 38.8 | 0.08 | 2.73e-04 | 8.6e-05 |
| 24 | slope-adaptive | 7,056 | 0.1 | 0.1 | 0.1 | 1.8 | 34.1 | 0.09 | 4.20e-04 | 3.3e-05 |
| 24 | uniform | 7,056 | 0.1 | 0.1 | 0.1 | 2.3 | 32.8 | 0.09 | 4.37e-04 | 3.6e-06 |
| 16 | grid | 15,625 | 0.3 | 0.4 | 20.4 | 34.5 | 60.2 | 0.11 | 2.40e-04 | 1.6e-14 |
| 16 | poisson-disk | 15,679 | 0.3 | 0.5 | 12.4 | 33.3 | 47.3 | 0.11 | 2.49e-04 | 2.5e-06 |
| 16 | quadtree | 15,628 | 0.4 | 1.0 | 21.5 | 46.5 | 70.6 | 0.11 | 1.23e-04 | 3.3e-05 |
| 16 | slope-adaptive | 15,625 | 0.4 | 0.6 | 12.9 | 38.4 | 56.1 | 0.11 | 1.90e-04 | 1.5e-05 |
| 16 | uniform | 15,625 | 0.4 | 0.6 | 14.7 | 34.0 | 55.2 | 0.11 | 1.97e-04 | 6.0e-06 |
| 12 | grid | 27,889 | 6.3 | 9.2 | 43.5 | 47.9 | 68.5 | 0.15 | 1.34e-04 | 2.4e-15 |
| 12 | poisson-disk | 27,868 | 6.9 | 10.9 | 36.1 | 45.0 | 58.5 | 0.15 | 1.40e-04 | 4.8e-07 |
| 12 | quadtree | 27,878 | 11.1 | 55.2 | 72.4 | 71.8 | 89.9 | 0.14 | 6.94e-05 | 1.7e-05 |
| 12 | slope-adaptive | 27,889 | 9.5 | 28.0 | 39.2 | 59.1 | 77.0 | 0.15 | 1.06e-04 | 8.6e-06 |
| 12 | uniform | 27,889 | 12.2 | 18.3 | 51.3 | 58.6 | 73.3 | 0.15 | 1.12e-04 | 1.5e-06 |
| 8 | grid | 62,500 | 27.0 | 69.9 | 120.0 | 82.0 | 105.2 | 0.25 | 5.96e-05 | 9.7e-15 |
| 8 | poisson-disk | 62,563 | 30.8 | 40.2 | 65.5 | 71.2 | 99.2 | 0.25 | 6.25e-05 | 1.0e-07 |
| 8 | quadtree | 62,441 | 71.4 | 127.5 | 173.6 | 123.8 | 145.5 | 0.24 | 3.15e-05 | 7.6e-06 |
| 8 | slope-adaptive | 62,500 | 56.0 | 77.3 | 106.2 | 94.2 | 114.3 | 0.26 | 4.74e-05 | 4.1e-06 |
| 8 | uniform | 62,500 | 35.3 | 61.8 | 96.3 | 84.8 | 110.1 | 0.26 | 5.01e-05 | 3.5e-07 |
| 4 | grid | 250,000 | 126.8 | 110.1 | 266.9 | 207.7 | 234.7 | 0.82 | 1.49e-05 | 3.4e-15 |
| 4 | poisson-disk | 249,832 | 170.8 | 120.4 | 324.9 | 160.6 | 234.5 | 0.82 | 1.56e-05 | 3.7e-08 |
| 4 | quadtree | 249,912 | 201.0 | 247.7 | 449.4 | 311.0 | 349.0 | 0.76 | 8.05e-06 | 1.9e-06 |
| 4 | slope-adaptive | 250,000 | 153.2 | 239.4 | 358.7 | 210.2 | 286.8 | 0.84 | 1.21e-05 | 1.1e-06 |
| 4 | uniform | 250,000 | 135.0 | 357.0 | 114.1 | 223.8 | 276.1 | 0.84 | 1.26e-05 | 6.9e-08 |

What the table says:

- Cost grows with n, from seconds at 4k nodes to 2–6 minutes at 250k nodes, and never needs more than 1 GB.
  A 200-epoch stride-4 training run takes 6 h on plaza, so the eigenvectors add about 1% to the wall time.
  The test graph adds the same again. RPEARL pays nothing here.
- The cost is not monotone in k. ARPACK converges slowly when the cutoff falls inside a cluster of nearly equal
  eigenvalues, which is the same eigengap that the theory ties to transferability. On the grids λ3 − λ2 is at
  machine precision (two axis modes), and the irregular graphs break the tie only by 1e-8 to 1e-4.
- The quadtree graphs are the most expensive at every size: their λ2 is half that of the other schemes, so the
  low end of the spectrum is more crowded.

### 7.3 Tuning k on grid stride 12 (plaza, both GPUs)

`scripts/e35_lappe_tune.sh`: `spnn.cli.tune --space lappe --sampler grid`, k ∈ {2, 4, 8, 16, 32, 64}, one trial
each, 100 epochs, e32 recipe, the e32 small test (100k pairs of the stride-4 grid test). Study
`~/runs/e35/tune_s12/e35-lappe-s12.db`, trials `tune_s12/trials/t*.json`, table `python ~/runs/e35/summarize_tune.py`.
Selection: best validation nmae (the test is only recorded).

Result (grid s12, 100 epochs, val nmae in %, small-test nmae and acc<2% in %):

| k | val nmae | best epoch | test nmae | acc<2% | wall s |
| --- | --- | --- | --- | --- | --- |
| 2 | 0.700 | 98 | 0.71 | 93.9 | 1308 |
| 4 | 0.709 | 90 | 0.71 | 93.7 | 1468 |
| 8 | 0.709 | 98 | 0.70 | 93.6 | 1395 |
| 16 | 0.706 | 79 | 0.70 | 93.7 | 1385 |
| 32 | 0.702 | 98 | 0.70 | 94.0 | 1303 |
| 64 | 0.691 | 83 | 0.68 | 94.1 | 1392 |

The landscape is flat: every k lands within 0.69–0.71% validation and 0.68–0.71% test, which is the run-to-run
noise seen in e34b (0.02 points). I took k = 64 by the stated rule (best validation). For comparison, the e34
RPEARL run on the same graph had 0.67% validation at epoch 100 and 0.60% test at epoch 200, so at stride 12 LapPE
is about 0.1 point behind RPEARL at equal epochs.

Note on the sampler (2026-10-20): Optuna's `GridSampler` shuffles its grid order with the seed, so two workers
with different seeds disagree on which grid id is which k, and a second worker also duplicates the last running
grid point when it runs out of unvisited ones. The first attempt ran k=4 twice; I killed the duplicate, moved
k ∈ {2, 8, 16, 32} to a second study (`tune_s12b`) with one seed for both workers, and killed one more
duplicate at the end. The stride-4 sweep runs as two single-worker studies with disjoint k sets instead.

### 7.4 Suite (all 30 runs on plaza, 2026-10-20/21)

Planned on betty: `scripts/e35_lappe_suite.sbatch`, 30 array tasks (5 schemes × 6 strides), 200 epochs,
k = 64, outputs under `$ALELAB_DRIVE/shortest-paths-nn/e35/`; staging from plaza with `~/runs/e35/sync_betty.sh`
(working tree to `~/sourcecode/shortest-paths-nn-asilomar`, e31 sets, test set, eigenvector cache). Betty stayed
unreachable from plaza (expired Kerberos ticket) on 2026-10-20 and 21, so the suite ran on plaza instead:
strides 32–8 (25 runs) with two workers of `scripts/e34_transfer_suite.sh` (`MODELS=sgt2lap TAG=e35 LAPPE_K=64`,
`~/runs/e35/suite/`) from 22:33 to 07:04, and the five stride-4 runs from 10:30 to 23:36 on 2026-10-21 (3.0–5.3 h
each). The sbatch was never submitted; it skips every run whose JSON is already staged, so it stays usable.

Result on the stride-4 grid test (nmae and acc<2% in %, 200 epochs; RPEARL = `sgt2` from section 3):

| scheme | stride | LapPE nmae | LapPE acc<2% | RPEARL nmae | RPEARL acc<2% | LapPE − RPEARL |
| --- | --- | --- | --- | --- | --- | --- |
| grid | 32 | 0.83 | 92.0 | 0.87 | 91.2 | −0.05 |
| grid | 24 | 0.82 | 92.9 | 0.70 | 94.1 | +0.12 |
| grid | 16 | 0.69 | 94.1 | 0.65 | 94.6 | +0.04 |
| grid | 12 | 0.70 | 94.1 | 0.60 | 95.7 | +0.10 |
| grid | 8 | 0.66 | 94.6 | 0.62 | 95.1 | +0.04 |
| grid | 4 | 0.68 | 94.8 | 0.62 | 95.6 | +0.07 |
| uniform | 32 | 1.87 | 63.7 | 2.02 | 60.9 | −0.14 |
| uniform | 24 | 1.79 | 64.7 | 1.85 | 64.1 | −0.06 |
| uniform | 16 | 1.67 | 68.1 | 1.66 | 69.0 | +0.01 |
| uniform | 12 | 1.70 | 67.2 | 1.72 | 67.0 | −0.02 |
| uniform | 8 | 1.65 | 67.8 | 1.62 | 70.4 | +0.03 |
| uniform | 4 | 1.67 | 67.7 | 1.63 | 69.2 | +0.04 |
| poisson-disk | 32 | 2.60 | 38.6 | 2.54 | 39.1 | +0.06 |
| poisson-disk | 24 | 2.53 | 39.2 | 2.57 | 38.5 | −0.04 |
| poisson-disk | 16 | 2.49 | 40.1 | 2.38 | 42.3 | +0.11 |
| poisson-disk | 12 | 2.33 | 43.2 | 2.34 | 43.2 | −0.01 |
| poisson-disk | 8 | 2.35 | 43.2 | 2.38 | 42.5 | −0.03 |
| poisson-disk | 4 | 2.27 | 44.8 | 2.25 | 45.5 | +0.02 |
| slope-adaptive | 32 | 1.99 | 63.1 | 1.98 | 63.2 | +0.01 |
| slope-adaptive | 24 | 2.24 | 57.5 | 2.47 | 57.1 | −0.22 |
| slope-adaptive | 16 | 1.72 | 66.6 | 1.84 | 65.0 | −0.11 |
| slope-adaptive | 12 | 1.69 | 67.8 | 1.71 | 67.4 | −0.02 |
| slope-adaptive | 8 | 1.68 | 68.4 | 1.64 | 69.9 | +0.04 |
| slope-adaptive | 4 | 1.63 | 69.4 | 1.62 | 70.0 | +0.02 |
| quadtree | 32 | 19.75 | 1.7 | 19.69 | 2.8 | +0.06 |
| quadtree | 24 | 19.75 | 2.0 | 19.79 | 2.0 | −0.04 |
| quadtree | 16 | 20.10 | 1.1 | 20.00 | 1.7 | +0.10 |
| quadtree | 12 | 19.81 | 1.7 | 19.99 | 1.7 | −0.18 |
| quadtree | 8 | 19.92 | 1.7 | 19.92 | 2.3 | +0.00 |
| quadtree | 4 | 19.78 | 2.3 | 19.90 | 2.8 | −0.12 |

Finding: LapPE and RPEARL are the same model on this test to within the run-to-run noise of 0.02–0.2 points.
The only consistent sign is on the grid training graphs, where RPEARL is ahead by 0.04–0.12 points at strides
24–4 and the acc<2% gap is 0.5–1.6 points. At stride 4 RPEARL is ahead on all four usable schemes, by 0.02–0.07
points, and on the four irregular schemes at coarser strides the sign flips from stride to stride. The unaligned eigenbases (section 7.1) do not hurt the grid-to-grid transfer that this test
measures. The expected failure, if it exists, has to show on the matched irregular tests of e34c, where train
and test eigenbases come from graphs with different tiny gaps (section 7.6).

### 7.5 k sweep on grid stride 4 (plaza, curiosity)

Two single-worker studies with disjoint k sets, 100 epochs each, about 2.4 h per trial
(`~/runs/e35/tune_s04`, records `trials/t*.json`; the k=2, 8 records were rebuilt from the study DB and wandb
after the two studies overwrote each other's files, see the `recovered` field).

| k | val nmae | best epoch | test nmae | acc<2% |
| --- | --- | --- | --- | --- |
| 2 | 0.754 | 96 | 0.68 | 94.5 |
| 4 | 0.764 | 89 | 0.69 | 94.5 |
| 8 | 0.766 | 89 | 0.67 | 94.7 |
| 16 | 0.750 | 96 | 0.67 | 94.7 |
| 32 | 0.756 | 94 | 0.68 | 94.5 |
| 64 | 0.740 | 98 | 0.66 | 95.0 |

Flat again, 0.74–0.77% validation for every k, and no mismatch with the stride-12 choice: k=64 is the best
value on both graphs, by 0.01 point at stride 12 and 0.01 at stride 4. Validation is 0.05 points worse than at
stride 12 while the test is the same (0.66–0.69), because at stride 4 the training graph is the test graph and
the validation pairs are the hard half of the budget. RPEARL on the same graph: 0.62% test at 200 epochs.

### 7.6 Matched tests (e34c cross evaluation of the 30 LapPE models, plaza, 2026-10-21/22)

`LAPPE_K=64 scripts/e34c_eval_cross.py --models sgt2lap --runs ~/runs/e35/suite --out ~/runs/e34c`
(150 metric files `norway_*_sgt2lap__<test>.json` next to the 450 of section 6). Here the test graph is the
stride-4 graph of the training scheme, so for the four irregular schemes the training and test eigenbases come
from different graphs with different tiny gaps. The table gives the matched-test nmae in % as LapPE / RPEARL
(difference in points):

| stride | grid | uniform | poisson-disk | slope-adaptive | quadtree |
| --- | --- | --- | --- | --- | --- |
| 32 | 0.83 / 0.87 (-0.05) | 1.19 / 1.31 (-0.12) | 0.99 / 0.86 (+0.13) | 1.29 / 1.27 (+0.02) | 0.66 / 0.63 (+0.03) |
| 24 | 0.82 / 0.70 (+0.12) | 1.19 / 1.12 (+0.06) | 0.91 / 0.90 (+0.01) | 1.66 / 1.84 (-0.18) | 0.70 / 0.57 (+0.13) |
| 16 | 0.69 / 0.65 (+0.04) | 0.93 / 0.89 (+0.04) | 0.85 / 0.73 (+0.13) | 1.10 / 1.20 (-0.10) | 0.67 / 0.53 (+0.14) |
| 12 | 0.70 / 0.60 (+0.10) | 0.98 / 0.93 (+0.05) | 0.77 / 0.71 (+0.06) | 0.95 / 0.88 (+0.07) | 0.62 / 0.51 (+0.11) |
| 8 | 0.66 / 0.62 (+0.04) | 0.85 / 0.75 (+0.09) | 0.75 / 0.70 (+0.05) | 0.91 / 0.80 (+0.11) | 0.63 / 0.51 (+0.12) |
| 4 | 0.68 / 0.62 (+0.07) | 0.83 / 0.76 (+0.07) | 0.75 / 0.66 (+0.09) | 0.83 / 0.73 (+0.10) | 0.63 / 0.49 (+0.14) |

Finding: on the matched tests LapPE is behind RPEARL at every stride from 16 down, 19 of 20 cells, by
0.04–0.14 points. At stride 4, where the training graph is the test graph and the eigenbases are identical,
the deficit is the same (0.07–0.14 points), so the unaligned bases are not the cause; LapPE is simply a weaker
encoding on this task. The relative gap is largest on quadtree (0.63 vs 0.49 at stride 4, +29%) and poisson-disk
(+14%), which are the two schemes with the most crowded low spectrum (section 7.2), and smallest on uniform
and slope-adaptive. At strides 32 and 24 the sign is mixed, as on the grid test. The effect is consistent but
small next to the sampling-scheme effect of section 6 (a factor of 2–3 between schemes), so the claim the
suite supports is "LapPE costs an eigendecomposition and is never better; it is slightly worse where the
eigengap is smallest", not a failure of LapPE. A direct test of the eigengap mechanism would evaluate the same
LapPE checkpoints with the test eigenvectors rotated inside their near-degenerate pairs; the models and the
cache make that a one-script experiment.

