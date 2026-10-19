# Are the e20–e30 results meaningful? (analysis, 2026-10-11)

**Status:** complete. Sections 1–6 are the analysis of the existing results. Section 7 reports the new runs on plaza. Section 8 has the conclusions and section 9 the ideas.

**Verdict.** The old single-stage results (e28–e30) are not usable as evidence of terrain awareness. Two data problems explain most of their error: the training sets have slope-weighted edges while the test set does not, and `--single-graph-full-batch` makes an "epoch" a single gradient step, so those models trained for 100 steps. With consistent data and the stage-1 setup of Chen et al., our single-stage code reaches 0.52% relative error on Norway, better than Chen's 0.90–1.05%. On Philly, a formula with no training already reaches 0.56%, so Philly cannot show terrain awareness at all.

All old model numbers predate the pass 2 bug fixes (see `2026-10-asilomar-refactor.md`). The new runs in section 7 use the fixed code.

## 1. What the notebooks show

The e23–e27 notebooks and the paper-plot notebooks average `test_batch_*` over the whole wandb history, not the final value. The e28–e30 tables use the summary `test_nmae`. nmae is the mean of |pred − true| / true over test pairs, the same quantity Chen et al. call relative error.

Best run per resolution, nmae, res 4 to 20:


| Experiment | Train → test                        | Model           | res4 | res10 | res20 | min–max   |
| ---------- | ----------------------------------- | --------------- | ---- | ----- | ----- | --------- |
| e29        | Norway → Norway `full_test-004`     | TAGConv         | .055 | .055  | .050  | .049–.057 |
|            |                                     | SparseGT data   | .042 | .041  | .037  | .037–.044 |
|            |                                     | SparseGT random | .033 | .033  | .030  | .030–.041 |
| e28        | Philly → Philly `philly_test_res04` | TAGConv         | .030 | .030  | .031  | .028–.034 |
|            |                                     | SparseGT data   | .022 | .022  | .022  | .021–.024 |
|            |                                     | SparseGT random | .028 | .028  | .035  | .027–.035 |
| e30        | Philly → Norway `full_test-004`     | TAGConv         | .030 | .037  | .032  | .030–.042 |
|            |                                     | SparseGT data   | .034 | .033  | .029  | .026–.049 |
|            |                                     | SparseGT random | .079 | .104  | .060  | .043–.126 |


Observations:

- The error is flat from res4 (250k nodes) to res20 (10k nodes).
- e30, trained on Philly, is as good on Norway as e29, trained on Norway.
- e26 (number of sources, 5 to 500): the number of training pairs barely changes the test MAE.
- Only 0–13% of pairs are within 1% of the true distance.
- No notebook has a baseline.



## 2. Baselines with no training

Definitions:

- **3D straight line:** the Euclidean distance between (x, y, z) of the two nodes.
- **Octile+dz:** the shortest 8-neighbour grid path on a flat grid, with the height difference added in quadrature. With grid spacing h and dx, dy in cells: sqrt((h · (max(dx, dy) + (√2 − 1) · min(dx, dy)))² + dz²). No fitted parameter.
- **× c:** the predictor times one scalar c, the median of true / predicted over 100 random pairs of the test set. The 100 pairs are left out of the metrics.

Full test sets, 8-neighbour grids, unweighted edges. Norway `full_test-004` has 250,000 nodes, 2.5M pairs (10 sources × all nodes), mean distance 12.0 km. Philly `philly_test-004` has 62,500 nodes, 625,000 pairs, mean distance 1.6 km. I generated the Philly set on plaza, because `philly_test_res04.npz` no longer exists on betty.


| Predictor        | Norway nmae | acc<1% | acc<2% | acc<5% | Philly nmae | acc<1% | acc<2% |
| ---------------- | ----------- | ------ | ------ | ------ | ----------- | ------ | ------ |
| 3D straight line | 7.33%       | 0.1%   | 0.9%   | 13.6%  | 5.58%       | 1.9%   | 8.5%   |
| Octile+dz        | **2.40%**   | 14.4%  | 47.1%  | 94.1%  | **0.56%**   | 88.7%  | 99.3%  |
| 3D line × c      | 1.74%       | 34.6%  | 66.4%  | 96.9%  | 1.96%       | 31.3%  | 66.3%  |
| Octile+dz × c    | **1.12%**   | 56.7%  | 86.2%  | 99.1%  | **0.29%**   | 97.5%  | 99.7%  |


Detour ratio true / octile+dz:


| Test set     | p50   | p90   | p99   | max   |
| ------------ | ----- | ----- | ----- | ----- |
| Norway res04 | 1.021 | 1.046 | 1.074 | 1.279 |
| Philly res04 | 1.005 | 1.011 | 1.019 | 1.096 |


Reading:

- On Norway, the grid explains most of the gap between straight line and true distance. The terrain adds 2% on the median pair and 5–8% on the top decile. That 2–8% is what a model must learn.
- On Philly, the terrain adds 0.5%. A model cannot show terrain awareness there, and neither can Chen's 0.21%.
- The old e29 models (3.0–5.7%) are worse than octile+dz (2.4%) on the same test set.



## 3. Why the old transferability curves are flat

The models learned a function close to a scaled straight-line distance of the coordinates.

- The node features are absolute (x, y, z) in km. Philly spans 0–3 km and Norway spans 0–20 km. A model that transfers between them without loss (e30) must be nearly linear in the coordinates.
- The grid factor (octile, median 1.08 over the straight line) is the same at every resolution and on every terrain. A model that learns it transfers for free.
- Training-pair count (e26) does not matter, which fits a simple target function.
- Section 7 confirms it: TAGConv with p=4 does not beat octile+dz on any detour bin.



## 4. Two data problems in the old runs

**Edge weights.** The launchers for e26, e27, e28 and e30 generate training data with `--edge-weight`, and `smaller_test_set.sh` generates `full_test-004` without it. The Norway `res##_hybrid.npz` files are weighted too.


| Data                                              | Edge weight / 3D length, median | max |
| ------------------------------------------------- | ------------------------------- | --- |
| Norway `res08_hybrid.npz` (e29 train)             | 1.17                            | 3.9 |
| Philly `philly_res08_hybrid.npz` (e28, e30 train) | 1.07                            | 2.0 |
| Norway `full_test-004.npz` (e29, e30 test)        | 1.00                            | 1.0 |


So e29 and e30 learned one distance and were tested on another. Bug 9 changed the weight formula, so regenerated data differs again.

**Full-batch epochs.** `--single-graph-full-batch` puts all training pairs in one batch and takes one optimizer step per epoch (`spnn/training.py`, `use_single_graph_full_batch` branch). e28–e30 ran with `--epochs 100`, so those models took 100 gradient steps. Chen's 500 epochs are minibatch epochs.

**A third trap, found while generating test data.** `make_test_dataset` builds a 4-neighbour grid unless `--triangles` is passed. `make_dataset` always builds an 8-neighbour grid. A test set made without `--triangles` has Manhattan paths, up to 41% longer than octile. A model trained on 8-neighbour data then scores 16% on it. `full_test-004` was made with `--triangles`, so the old results are not affected.

## 5. What the error levels mean in practice

Distances are in km.


| Terrain | Mean true distance | Old model nmae | Typical error | Terrain detour on the median pair |
| ------- | ------------------ | -------------- | ------------- | --------------------------------- |
| Norway  | 12 km              | 3–5.5%         | 360–660 m     | 2% = 240 m                        |
| Philly  | 1.6 km             | 2–3.5%         | 32–56 m       | 0.5% = 8 m                        |


The old errors are larger than the terrain effect they are meant to capture. A user who needs the detour cannot see it in the prediction. A 2.4% error on Norway is available from the octile formula with no training, and 1.1% with one calibration scalar.

## 6. Comparison with Chen, Agarwal and Wang (ICML 2025)

"De-coupled NeuroGF for Shortest Path Distance Approximations on Large Terrain Graphs", PMLR 267:7648–7663. Its code is `github.com/chens5/shortest-paths-nn`, the origin of this repo. It uses the same Norway and Philadelphia terrains. It trains one model per terrain and has no test on an unseen terrain.


| Item        | Chen, stage 1                                                  | Ours before this analysis                      |
| ----------- | -------------------------------------------------------------- | ---------------------------------------------- |
| Backbone    | GAT, 3 layers, 128 hidden, 64 out, edge weights in attention   | TAGConv K=3, 64 hidden, 3 layers, or SparseGT  |
| Head        | L1 norm of the embedding difference, MSE loss                  | p=4 norm, MSE loss                             |
| Optimizer   | AdamW, lr 1e-3, 500 epochs                                     | AdamW, lr 0.008 or 0.001, 100 full-batch steps |
| Coarse grid | 250×250 for Norway (= our res08), 50×50 for Philly             | res04–res20                                    |
| Pairs       | 1,000 sources × 500 targets                                    | hybrid, up to 500 sources                      |
| Metrics     | mean relative error; accuracy = share of pairs with error < 2% | nmae; acc<1%, acc<5%                           |


The right comparison for a single-stage run is her "Coarse-GAT+L1" column: the coarse-trained embedding applied directly to the full graph with no head retraining (her Table 3). M-CTR is her two-stage method.


| Setting                | Chen, Coarse-GAT+L1 | Chen, M-CTR   | Ours, old single stage | Ours, new (section 7) |
| ---------------------- | ------------------- | ------------- | ---------------------- | --------------------- |
| Norway, unweighted     | 1.05% / 89.2%       | 0.90% / 91.1% | 3.0–5.7% (e29)         | **0.52% / 96.5%**     |
| Philly, unweighted     | 2.07% / 30.1%       | 0.51% / 94.4% | —                      | **0.29% / 99.6%**     |
| Norway, slope-weighted | 2.30% / 62.1%       | 2.04% / 68.1% | —                      | —                     |
| Philly, slope-weighted | 2.43% / 61.4%       | 1.32% / 83.9% | 2.1–3.5% (e28)         | —                     |


Cells are relative error / accuracy at 2%.

## 7. New runs (plaza GPU 0, fixed code, 2026-10-11)

Data: unweighted 8-neighbour grids. Training sets have 1,000 sources × 500 random targets at res08, like Chen. Tests use the full sets of section 2. Config `configs/gat-chen-stage1.yml`; wandb tag `chen-replication` in `alelab/manifold-transformers-dev`; outputs in `~/smoke/chen/` on plaza. "Full batch" means `--single-graph-full-batch` (one step per epoch). "Minibatch" means batches of 4,096 pairs (122 steps per epoch).


| Run                            | Train → test    | Model, head, lr    | Training             | nmae      | acc<1% | acc<2% | acc<5% | train nmae |
| ------------------------------ | --------------- | ------------------ | -------------------- | --------- | ------ | ------ | ------ | ---------- |
| `gat_l1_norway08_mb`           | Norway → Norway | GAT, L1, 1e-3      | 500 minibatch epochs | **0.52%** | 88.0%  | 96.5%  | 99.5%  | 0.46%      |
| `gat_l1_norway08`              | Norway → Norway | GAT, L1, 1e-3      | 500 full-batch steps | 0.78%     | 76.2%  | 91.9%  | 99.0%  | 0.80%      |
| `tagconv_p4_norway08`          | Norway → Norway | TAGConv, p=4, 1e-3 | 100 full-batch steps | 2.45%     | 29.8%  | 57.3%  | 90.7%  | 3.12%      |
| `gat_l1_philly08_to_norway`    | Philly → Norway | GAT, L1, 1e-3      | 500 full-batch steps | 1.99%     | 31.1%  | 60.8%  | 94.4%  | 0.41%      |
| `gat_l1_philly08_to_norway_mb` | Philly → Norway | GAT, L1, 1e-3      | 200 minibatch epochs | 3.08%     | 24.8%  | 40.6%  | 81.5%  | 0.15%      |
| `gat_l1_philly08_mb`           | Philly → Philly | GAT, L1, 1e-3      | 200 minibatch epochs | **0.29%** | 97.9%  | 99.6%  | 99.9%  | 0.15%      |
| octile+dz                      | — → Norway      | no training        | —                    | 2.40%     | 14.4%  | 47.1%  | 94.1%  | —          |
| octile+dz × c                  | — → Norway      | 100 pairs          | —                    | 1.12%     | 56.7%  | 86.2%  | 99.1%  | —          |
| octile+dz                      | — → Philly      | no training        | —                    | 0.56%     | 88.7%  | 99.3%  | 100%   | —          |
| octile+dz × c                  | — → Philly      | 100 pairs          | —                    | 0.29%     | 97.5%  | 99.7%  | 100%   | —          |


Error by detour bin on Philly, nmae:


| Detour bin     | Share of pairs | octile+dz | GAT+L1 minibatch |
| -------------- | -------------- | --------- | ---------------- |
| [1.000, 1.005) | 51.0%          | 0.27%     | 0.32%            |
| [1.005, 1.010) | 37.4%          | 0.69%     | 0.24%            |
| [1.010, 1.020) | 10.8%          | 1.27%     | 0.28%            |
| [1.020, 1.100) | 0.8%           | 2.61%     | 1.35%            |


Error by detour bin on Norway, nmae. The bin is the ratio true / octile+dz.


| Detour bin   | Share of pairs | octile+dz | GAT+L1 minibatch | GAT+L1 full batch | TAGConv p=4 | GAT Philly→Norway, full batch |
| ------------ | -------------- | --------- | ---------------- | ----------------- | ----------- | ----------------------------- |
| [1.00, 1.02) | 45.8%          | 1.24%     | 0.52%            | 0.79%             | 2.61%       | 1.61%                         |
| [1.02, 1.05) | 46.8%          | 3.01%     | 0.47%            | 0.67%             | 2.23%       | 2.13%                         |
| [1.05, 1.10) | 7.2%           | 5.78%     | 0.83%            | 1.36%             | 2.80%       | 3.52%                         |
| [1.10, 1.30) | 0.1%           | 10.4%     | 3.13%            | 4.03%             | 4.04%       | 6.09%                         |


Error by distance bin on Norway, nmae:


| Distance | Share of pairs | octile+dz | GAT+L1 minibatch | TAGConv p=4 |
| -------- | -------------- | --------- | ---------------- | ----------- |
| 0–2 km   | 2.4%           | 1.75%     | 3.16%            | 8.71%       |
| 2–5 km   | 9.9%           | 2.30%     | 1.08%            | 4.71%       |
| 5–10 km  | 25.4%          | 2.57%     | 0.60%            | 2.61%       |
| 10–20 km | 54.3%          | 2.42%     | 0.32%            | 1.78%       |
| 20–30 km | 8.0%           | 2.12%     | 0.21%            | 1.79%       |


One-scalar calibration on 100 target pairs:


| Run                             | nmae before | nmae after |
| ------------------------------- | ----------- | ---------- |
| GAT Norway → Norway, minibatch  | 0.52%       | 0.52%      |
| GAT Philly → Norway, full batch | 1.99%       | 1.69%      |
| GAT Philly → Norway, minibatch  | 3.08%       | 2.92%      |




## 8. Conclusions

1. **Our single-stage code matches and beats Chen's stage 1 on Norway** once train and test data agree: 0.52% against her 1.05%, and better than her two-stage M-CTR at 0.90%. The old gap came from the data pipeline, not from the method.
2. **GAT+L1 learns the terrain.** On the 7% of pairs with a 5–10% detour, its error is 0.8% and the octile formula's is 5.8%. The gain over the formula grows with the detour.
3. **TAGConv with the p=4 head does not.** With the same data and 100 steps, it is at the octile level in every detour bin. Whether the cause is the head, the layer or the step count is not separated yet.
4. **Cross-terrain transfer works but is 4–6× worse than in-terrain**, 2.0–3.1% on Norway from a Philly model. More Philly training makes it worse: train error 0.15% and test error 3.1%. The model overfits Philly, which is nearly flat. One calibration scalar recovers only 0.2–0.3 points, so the gap is not a scale bias.
5. **Short pairs are the weak spot of the learned models.** Under 2 km, GAT is at 3.2% and octile at 1.8%. This is where a relative loss or a residual-on-octile head would help.
6. **Philly is not a useful terrain for this study.** The formula reaches 0.56% there with no training, and 0.29% with one scalar. The GAT reaches the same 0.29% (Chen: 0.21% full graph, 2.07% coarse). The model only beats the formula on the 11% of pairs with a detour above 1%, and the margin there is small in absolute terms.
7. **The old e20–e30 results should not go into the paper.** Rerun with unweighted or consistently weighted data, minibatch training, the octile baselines and the detour-bin table.



## 9. Ideas to beat Chen with one stage

Ordered by cost. Items 1 and 2 are done above.

1. ~~Replicate her stage 1.~~ Done: 0.52% on Norway.
2. ~~Add the baselines.~~ Done: octile+dz and octile+dz × c. Add them to `spnn/metrics.py` so every run reports them.
3. Separate the head from the layer: TAGConv with L1, GAT with p=4, both minibatch.
4. Slope-weighted edges on both sides, with the fixed formula (bug 9). The octile baseline is weaker there, so the terrain term is larger and the comparison with Chen's Table 4 is possible.
5. Predict the detour, not the distance: target = true / octile+dz, or head = octile+dz × (1 + g(u, v)). The straight-line part is then exact and short pairs stop dominating the error.
6. Relative or log-distance loss, so short pairs count as much as long pairs.
7. Translation- and scale-invariant inputs: slope, roughness and height above the local mean instead of absolute (x, y, z); the coordinate difference goes to the head. This targets the cross-terrain gap.
8. Sampling for detours (the Asilomar topic): stratify training pairs by detour ratio and by distance. The 0.1% of pairs above 1.10 carry the largest errors. `legacy/dataset/artificial_dem_array.py` (`guarantee_rough_path`) and the critical-point sampler are starting points.
9. Multi-resolution and multi-terrain training, which the fixed training loop supports (bug 7). Chen leaves cross-terrain transfer open.
10. Early stopping on the cross-terrain runs, or fewer epochs, since more training hurt transfer.

