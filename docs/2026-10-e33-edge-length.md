# e33: edge-length ablation (2026-10-15)

**Verdict.** Edge lengths, as implemented here, do not close the gap to GAT+L1.
The best e33 run, SparseGT k=2 with the Euclidean attention bias, reaches 0.65% test nmae.
The e32 SparseGT k=2 without lengths reached 0.68%, and GAT+L1 reached 0.52%.
The learned length parameters end where AdamW weight decay alone would put them (section 6.1).
So the runs do not show what the models do with a length signal that they are free to learn.

## 1. Question

JP asked: does giving our models the edge lengths close the gap to the GAT+L1 replication (0.52%)?
Chen's GAT sees the edge length. TAGConv and SparseGT ignored it in e31 (sampling-scheme transfer) and e32 (SparseGT recipe tuning).
Background: `docs/2026-10-e31-sampling-transfer.md` section 7.5 and `docs/2026-10-e32-sgt-tuning.md` section 11.

## 2. What Chen's GAT does with the length

We build Chen's GAT as PyG `GATConv(in, out, edge_dim=1)` with one head (`spnn/models/gnn.py`, `create_graph_layer`).
PyG computes the attention logit of edge (j → i) as (`torch_geometric/nn/conv/gat_conv.py:400-408`):

s_ij = LeakyReLU(a_sᵀ W x_j + a_tᵀ W x_i + a_eᵀ W_e e_ij),  α_ij = softmax_j(s_ij)

- e_ij is the edge length in km. W_e is a C×1 matrix, so a_eᵀ W_e e_ij = w_e · e_ij with one effective scalar w_e per layer and head.
- The softmax runs over the 1-hop neighbours plus a self-loop. PyG adds the self-loop with e_ii = mean of the incoming edge lengths (`fill_value='mean'`, line 140).
- The length term sits inside the LeakyReLU, so it is not a pure additive logit bias.

## 3. The two implementations

Fable (the previous model in this session) wrote both. They are uncommitted in the working tree.

### 3.1 TAGConv with an exponential edge-weight kernel

w_ij = exp(−e_ij / σ),  σ = softplus(ρ) in km, learnable, initial σ = 0.2 km

- TAGConv normalises with `gcn_norm`: Â = D^−1/2 W D^−1/2 with no self-loops. All K = 3 hops and all 3 layers use the same w.
- Code: `ExpEdgeKernel` (`spnn/models/gnn.py:18`), registry `EDGE_WEIGHT_KERNELS` (line 40), use in `GNNModel.forward` (line 148).
- Config key: `gnn.constr.edge_weight_kernel: exp` in `configs/e33-tagconv-ew.yml`. Needs `--include-edge-attr 1`.
- Parameters added: 1 (ρ).

### 3.2 SparseGT with a Euclidean attention bias

In every `SparseKHopGraphAttention` layer and head h:

logit_ij = (q_iᵀ k_j) / √F_h + m_h · ‖p_i − p_j‖,  for (i, j) in the k-hop window, −∞ outside

- p = the raw node xyz in km, passed as `Data.pos` (`spnn/models/sparse_gt.py:662`, read once at line 516).
- With num_hops = 1 the window is the edge set, so ‖p_i − p_j‖ is the edge length. With num_hops = 2 it is the straight-line distance to every node within 2 hops.
- Slopes m_h are learnable per layer and head. Initial values follow ALiBi: m_h = −10 · 2^−h per km, h = 0..7 (line 364).
- Code: `_csr_pair_distances` (line 328), bias built at line 424, added after the 1/√F_h scaling at line 305.
- Config key: `gnn.sparse_gt.attn_bias: euclidean` (`spnn/models/factory.py:13,65`). Configs `configs/e33-sparse-gt-k1-eb.yml` and `configs/e33-sparse-gt-k2-eb.yml`, copies of `configs/e32-sparse-gt-k2.yml` with only `num_hops` and `attn_bias` changed.
- Parameters added: 6 layers × 8 heads = 48.

Tests: `tests/test_edge_length_models.py` (4 tests). Fable reported 117 passed and the goldens unchanged.

## 4. Setup

Launcher: `scripts/e33_edge_length.sh <gpu>`, written and started by Fable on plaza. GPU 0 ran sgt1_eb then sgt2_eb. GPU 1 ran tagconv_ew then tagconv.

| Item | Value |
| --- | --- |
| Train data | `~/data/spnn/e31/norway_grid_s16.npz`: grid stride 16, 15,625 nodes, 500,000 pairs |
| Test data | `~/data/spnn/full_test-004.npz`: stride 4, 250,000 nodes, 2.5M pairs, the e31/e32 full test |
| Train edge length | 0.16–0.39 km, median 0.23 km |
| Test edge length | 0.04–0.27 km, median 0.057 km |
| Recipe | e32 recipe: batch 8,192, AdamW wd 0.1, 300 warmup steps, constant lr, 100 epochs = 5,800 steps |
| Validation | 25,000 held-out training pairs, best-validation checkpoint |
| Head | siamese L1 (`--p 1`), MSE loss, `--trial 1` |
| lr | TAGConv 1e-3 (e31 value). SparseGT 3.5e-3 (e31 value; e32 used 3.257e-3) |

Outputs: `~/runs/e33/e33_<run>_s16.json` and `.log`, run directories `~/runs/e33/e33_<run>_s16/`.
wandb `alelab/terrains-dev`, tag `e33`: tagconv `6nmx7vvw`, tagconv_ew `lt4kjtsb`, sgt1_eb `fqnr2yve`, sgt2_eb `mkfewmr1`.
Analysis script (Opus, plaza CPU): `~/runs/e33/analysis/compare.py`. Outputs `table.csv`, `bins.csv` and `params.json` in the same folder.

## 5. Results

Each run has one seed. We have no seed-variance estimate, so differences below about 0.03 points may be noise.

### 5.1 Test metrics

| Run | Edge lengths | Window | Test nmae | Acc<2% | Train nmae | Best val nmae (epoch) | Wall time |
| --- | --- | --- | --- | --- | --- | --- | --- |
| e33 tagconv (control) | no | K=3 hops | 0.81% | 91.5% | 0.84% | 0.859% (98) | 114 s |
| e33 tagconv_ew | exp kernel | K=3 hops | 0.77% | 92.7% | 0.83% | 0.849% (99) | 134 s |
| e33 sgt1_eb | logit bias | 1 hop, no self | 0.68% | 94.3% | 0.65% | 0.656% (98) | 751 s |
| e33 sgt2_eb | logit bias | 2 hops + self | **0.65%** | 94.6% | 0.65% | 0.654% (93) | 914 s |
| e32 sgt2 (reference) | no | 2 hops + self | 0.68% | 94.4% | 0.66% | 0.661% (84) | 882 s |
| e31 tagconv s16 (old recipe) | no | K=3 hops | 1.09% | 86.8% | 1.08% | 1.088% (98) | — |
| e31 mlp s16 | no | none | 0.77% | 93.3% | 0.70% | 0.722% (95) | — |
| GAT+L1, stride 8 | GATConv | 1 hop + self | 0.52% | 96.5% | 0.46% | 0.465% (490) | — |
| octile+dz | — | — | 2.40% | — | — | — | — |

The bins script gives 2.40% for octile+dz, while the task list quoted 1.12% for "octile+dz × c", the calibrated formula.
Wall time includes the 2.5M-pair test. The GAT+L1 run trained at stride 8 for 500 minibatch epochs of 4,096 pairs.

Most of the TAGConv gain comes from the recipe, not from the lengths:

| Change | Test nmae |
| --- | --- |
| e31 TAGConv → e33 tagconv control (e32 recipe) | 1.09% → 0.81% |
| e33 tagconv control → tagconv_ew (edge lengths) | 0.81% → 0.77% |
| e32 sgt2 → e33 sgt2_eb (edge lengths) | 0.68% → 0.65% |

### 5.2 Validation curves

Validation nmae at selected epochs. One epoch = 58 steps. The 300 warmup steps end in epoch 5.

| Run | 0 | 1 | 2 | 5 | 10 | 20 | 50 | 99 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tagconv | 119.2% | 26.8% | 18.8% | 10.7% | 6.78% | 3.69% | 1.36% | 0.88% |
| tagconv_ew | 119.4% | 27.0% | 19.5% | 10.8% | 6.84% | 3.60% | 1.31% | 0.85% |
| sgt1_eb | 114.9% | 50.0% | 52.2% | 33.2% | 8.13% | 2.51% | 0.73% | 0.68% |
| sgt2_eb | 113.8% | 49.1% | 50.0% | 33.3% | 8.15% | 2.59% | 0.74% | 0.66% |
| e32 sgt2 | 120.9% | 49.6% | 49.8% | 38.3% | 8.94% | 2.81% | 0.75% | 0.67% |

The bias does not give the GAT-like fast start.
GAT+L1 had 1.12% validation nmae after its first epoch (122 steps, no warmup).
The SparseGT runs were still near 50% after 116 steps, but their lr was still in warmup, so the comparison is not clean.

### 5.3 Learned parameters

AdamW multiplies each weight by (1 − lr · wd) at each step. Over the 5,650 effective steps of this recipe, the product is exp(−wd · lr · 5,650).
The table compares the learned values with that pure-decay value, where the gradient does nothing.

| Parameter | Initial | Pure weight decay only | Learned |
| --- | --- | --- | --- |
| tagconv_ew σ | 0.200 km | 0.354 km (ρ: −1.508 → −0.857) | 0.355 km (ρ = −0.853) |
| sgt1_eb m_0 (all 6 layers) | −10 /km | −1.38 /km (factor 0.138) | −1.41 /km |
| sgt2_eb m_0 (all 6 layers) | −10 /km | −1.38 /km | −1.56 /km |

All 8 heads keep the ALiBi ratio 2^−h. For example, sgt2_eb layer 0 ends at −1.56, −0.78, −0.39, −0.20, −0.10, −0.05, −0.025, −0.013 /km.
All slopes stay negative, so nearer nodes keep a higher logit.
The full values are in `~/runs/e33/analysis/params.json`.

What the final slopes mean in logits:
- sgt1_eb, train graph: an axial edge (0.16 km) and a diagonal edge (0.23 km) differ by 1.41 × 0.066 = 0.09 logits in head 0.
- On the stride-4 test graph the same pair of edges differs by 0.02 logits.

### 5.4 Error by detour bin

Method of `~/runs/e31/analysis/bins.py`: 2,499,990 test pairs with true > 0; detour = true / octile+dz.

| Detour bin | Share | octile+dz | tagconv | tagconv_ew | sgt1_eb | sgt2_eb | e32 sgt2 | e31 mlp s16 | GAT+L1 s08 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [1.00,1.02) | 45.9% | 1.24% | 0.69% | 0.63% | 0.50% | 0.48% | 0.46% | 0.65% | 0.52% |
| [1.02,1.05) | 46.9% | 3.01% | 0.74% | 0.72% | 0.68% | 0.64% | 0.69% | 0.74% | 0.47% |
| [1.05,1.10) | 7.2% | 5.78% | 1.98% | 1.93% | 1.79% | 1.77% | 1.92% | 1.63% | 0.83% |
| [1.10,1.30) | 0.1% | 10.44% | 6.11% | 6.06% | 5.49% | 5.87% | 5.53% | 5.25% | 3.13% |
| **all** | 100% | 2.40% | 0.81% | 0.77% | 0.68% | 0.65% | 0.68% | 0.77% | 0.52% |

The gap to GAT+L1 is in the detour bins ≥ 1.02, not in the straight-line bin.
In [1.05,1.10) the best e33 run has 1.77% and GAT+L1 has 0.83%.
The edge lengths lower this bin by 0.15 points for SparseGT k=2.

## 6. Review findings

Opus (a review subagent) read the diff and checked the points below on the plaza CPU.

### 6.1 Defect: weight decay sets the learned length parameters

- Where: `spnn/training.py:144`, `AdamW(parameters, lr=lr, weight_decay=weight_decay)`. All parameters are in one group with wd 0.1.
- What: ρ and m_h decay toward 0. For the SparseGT bias, 0 means no length information. For TAGConv, ρ = 0 means σ = 0.69 km, a flatter kernel.
- Evidence: the learned values are within 0.3–13% of the pure-decay values (table 5.3). The gradient barely moved them.
- Consequence: e33 does not test a learned length weighting. It tests a fixed schedule from strong to weak bias.
- Possible fix: put `rho` and `bias_slope` in a parameter group with weight_decay 0. I did not change the code.

### 6.2 Design issue: the TAGConv kernel changes between train and test graphs

`gcn_norm` removes any common scale of w on a near-regular grid. Only the ratio between edge weights matters.
That ratio depends on the edge length in units of σ, so it differs between the stride-16 train graph and the stride-4 test graph.

| Graph | Axial edge | Diagonal edge | Diagonal/axial weight at σ = 0.2 | at σ = 0.355 |
| --- | --- | --- | --- | --- |
| train, stride 16 | 0.160 km | 0.226 km | 0.72 | 0.83 |
| test, stride 4 | 0.040 km | 0.057 km | 0.92 | 0.95 |

The docstring at `spnn/models/gnn.py:24-26` says the km scale makes the weight "mean the same thing" on both graphs. Under `gcn_norm` it does not.
A scale-free weight such as 1/e_ij would keep the same ratio (0.71) on both graphs.

### 6.3 Design issue: the attention bias shrinks 4× on the test graph

The bias is m_h × distance in km. The test graph has 4× shorter edges, so logit differences inside the window are 4× smaller (section 5.3).
Chen's GAT has the same property, with a 2× change from stride 8 to stride 4. This is not a bug, but it is a train/test shift.

### 6.4 k = 1 has no self-loop

- `_get_attn_window` returns `data.edge_index` for num_hops ≤ 1 (`spnn/models/sparse_gt.py:488-489`). The grid has no self-loops, so node i does not attend to itself.
- With num_hops = 2 the window is (A + I)², so (i, i) is in the window with distance 0, which gives JP's w_ii = 0.
- At k = 1, x_i still reaches the output through the residual X + attn_out (line 520 onward). So the run is valid.
- It differs from GAT, which attends to a self-loop with e_ii = mean incoming length.

### 6.5 Checks that passed

| Check | Result |
| --- | --- |
| Bias order equals the CSR value order of `sampled_addmm` | Yes. I compared `_mha_sparse_attention` with a dense masked softmax, with unsorted and duplicate edges, k = 1 and 2. Max difference 1e-15 in float64 |
| `pos` is raw xyz in km in every layer | Yes. `pos = x` before projection (line 662). `KHopGTModel.forward` reads it once and passes it to every layer. Train and test npz xyz and `distances` are both in km, and ‖p_i − p_j‖ equals `distances` exactly |
| Interaction with `SparseValuesDropout(set_to_neg_inf=True)` | Dropped logits are −∞ before the bias, and −∞ + bias = −∞. The force-keep rule still prevents an all-−∞ row |
| Softmax over `row_index` | Unchanged. The bias is per value, so the softmax groups are the same |
| dtype and device | Slopes and `pos` are float32 on the model device. The bias is (H, nnz) |
| Eval path, `spnn/metrics.py:41` | Passes `edge_attr` of the test graph, so TAGConv gets test edge weights. SparseGT reads `pos` from `x` |
| `edge_attr` shape [E, 1] | `reshape(-1)` in `ExpEdgeKernel.forward` gives [E] |
| Behaviour with the options off | No change. `kwargs` is empty for TAGConv, `bias` is None, `pos` is unused, and the new parameters exist only when the option is on, so state dicts match |

Minor cost: `_csr_pair_distances` runs again in each of the 6 layers. One call in `KHopGTModel.forward` would do.

## 7. Who decided what

| Item | Decided by |
| --- | --- |
| Run the ablation to test whether edge lengths close the gap to GAT | JP |
| Logit idea: B_ij + w_ij with w_ij the edge weight, −∞ outside the neighbourhood | JP |
| Run the GNN (TAGConv) first | JP |
| Use the Euclidean distance over the whole k-hop window, not only edges | Fable |
| ALiBi-style per-head slopes, initial −10 · 2^−h per km | Fable |
| Exponential kernel exp(−e/σ) with learnable σ, initial 0.2 km, for TAGConv | Fable |
| lr 1e-3 for TAGConv and 3.5e-3 for SparseGT; the e32 recipe otherwise | Fable |
| Weight decay 0.1 on all parameters, including the new ones | Inherited from the e32 recipe; nobody chose it for e33 |
| Review, analysis and this note | Opus |

## 8. Round 2: no weight decay on the length parameters (Fable, 2026-10-15)

The review in section 6 found that AdamW weight decay 0.1 set the values of σ and the attention slopes. I fixed this in `spnn/training.py` (`param_groups`, `NO_WEIGHT_DECAY_SUFFIXES`): `edge_kernel.rho` and `bias_slope` now sit in a group with weight decay 0. Models without these parameters keep one group, so the e32 recipe is unchanged (`tests/test_edge_length_models.py::test_physical_scalars_get_no_weight_decay`). I reran the three runs with the same launcher, `OUT=~/runs/e33/nodecay TAG=e33nd`, from 12:29 to 12:57 EDT.

| Run | Round 1 test nmae | Round 2 test nmae | Round 2 acc<2% | Learned parameter, round 2 |
| --- | --- | --- | --- | --- |
| TAGConv control, no lengths | 0.81% | — | 91.5% | — |
| TAGConv + exp kernel | 0.77% | 0.81% | 92.2% | σ 0.202 km, from 0.200 |
| SparseGT k=2, e32 reference, no lengths | 0.68% | — | 94.4% | — |
| SparseGT k=1 + bias | 0.68% | 0.68% | 94.5% | slopes −10.0 … −0.08 /km, unchanged from init |
| SparseGT k=2 + bias | 0.65% | 0.66% | 94.6% | slopes −10.0 … −0.08 /km, unchanged from init |

Findings:
- With weight decay off, the length parameters do not move. σ changes by 1% and the slopes by under 2%. Their gradients are negligible, so the models do not use the length signal to fit the training pairs.
- The round-1 differences were the decay, not learning. TAGConv's round-1 0.77% came from a flattened kernel and is back at the control's 0.81%.
- Verdict: edge lengths, in these two forms, do not help on the Norway grid at stride 16. The 0.02–0.03 points between the SparseGT variants are within one-seed noise.
- The gap to the GAT+L1 replication (0.52% at stride 8, 61,000 steps) is not explained by the edge lengths. The remaining differences are training length (5,650 vs 61,000 steps), stride (16 vs 8) and the GAT's self-loop with mean edge length.
