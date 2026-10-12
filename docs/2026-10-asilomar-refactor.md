# Asilomar refactor (branch `asilomar`, 2026-10-09 to 2026-10-10)

The refactor is complete. The code is now the `spnn/` package. We fixed 22 of the 23 bugs that the audit found. Bug 4 needs no code change.

**Consequence for past results:** every SparseGT result and every result that chose a checkpoint on the test set (bug 2) must be run again before it goes in a paper. See [Bugs that change past results](#bugs-that-change-past-results).

## Stages

| Stage | Commits | What it did | Metrics change? |
|---|---|---|---|
| Step 0: test harness | 742ba61, 24f44c0 | Characterization tests. They save current outputs as goldens (JSON files in `tests/golden/`) and fail when an output changes. A GPU smoke check on plaza. | No |
| Pass 1: style | ecde65e – 9f56440 | Code moved into the `spnn/` package with explicit imports. Unused code moved to `legacy/` (see `legacy/README.md`). The 6 sweep scripts merged into `spnn/cli/sweep.py`. | No. The goldens did not change. |
| Pass 2: bug fixes | 127eb91 – 5843365 | One bug per commit. Each commit message says which metrics change and why. | Yes, for some bugs |

## Bugs that change past results

Past wandb runs have these bugs.

| Bug | Commit | What was wrong | Effect on past runs |
|---|---|---|---|
| 23 | ba9de85 | SparseGT never trained Q, K and V. The line `torch.utils.checkpoint.checkpoint(lambda: attn_out)` detached the attention output. The model learned only through the residual path. The line was there since the first SparseGT commit (fe9f513). | Every SparseGT run from this repo had frozen random attention projections. All SparseGT results must be run again. |
| 2 | 51dd601 | The validation set was 0.5% of the test pairs, on the test graph. | Early stopping and checkpoint choice used test data. This matters most for cross-terrain runs such as e30 (train Philadelphia, test Norway). |
| 3 | ceb9fdd | The seed was always 0, whatever the `--trial` value. | All trials had the same init and shuffle order. Error bars across trials are too small. |
| 1 | 127eb91 | TAGConv always used K=3. `tagconv-k5.yml` was a copy of `tagconv-k3.yml`. | Runs labelled K=5 used K=3. |
| 9 | 0afb97d | The `--edge-weight` factor was `1 + |arctan(dz)/dxy|`. That is not an angle, and its size depends on units. It is now `1 + arctan(|dz|/dxy)`. | Data made with `--edge-weight` changes when it is generated again. This includes the e26–e28 and e30 datasets. |
| 6 | 30960ee, 5376274 | Two samplers could pick the source as its own target. Metrics then counted zero-length pairs as misses or gave NaN. | test_acc_* was slightly too low on test sets with zero-length pairs. |
| 8 | 49e9fa4 | R-PEARL random mode drew new random inputs at every evaluation. | Noisy validation and checkpoint choice for SparseGT random. |
| 12 | 2cdd4bb | Simplified-mesh datasets could put the same pair in both train and test. | Test metrics on these datasets were too good. The data must be generated again. |

## Bugs with no effect on the current pipelines

These fixes change no metric for the runs we used. Other code paths now work.

| Bug | Commit | Fix |
|---|---|---|
| 7 | 6b4c918 | Metrics cover every graph, not only graph 0. |
| 10 | 5064518 | Non-square DEMs give correct graphs and pairs. |
| 14 | 5c6da05 | Final metrics show in the wandb history, not only in the summary. |
| 15 | cf7a26e | A dataloader with one pair no longer crashes `compute_metrics`. |
| 13, 16 | 47af874 | 10 of 15 GNN layer types crashed. GINConv and GeneralConv now work. The others moved to `legacy/src/gnn_layers_broken.py`. |
| 17 | 5a02ba5 | The critical-point-source sampler no longer crashes. |
| 18, 19 | c08603b | Mesh input to `make_dataset` now writes a dataset. An unused random draw no longer shifts the pair sampling. |
| 20 | a6b078f | Mesh graph node i is row i of the node file, so it matches the node features. |
| 21 | efa99f0 | Quadtree quadrants q2 and q4 get their own bounds. |
| 22 | 7d9f070 | The expanding-radius test sampler no longer crashes. |
| 5, 11 | 5843365 | Unknown config keys raise an error. Settings that have no effect give a warning. `--vn` and `--finetune` accept only 0. |
| 4 | none | `legacy/wandb_tools/backfill_wandb_metrics.py` averages sampled history. The script is in `legacy/` and `legacy/README.md` describes the problem. |

## Smoke check

The smoke check runs on plaza GPU 1 and logs to the wandb project `alelab/manifold-transformers-dev`.

```bash
CUDA_VISIBLE_DEVICES=1 PYTHON=~/miniconda3/envs/spnn-asilomar/bin/python \
    bash tests/smoke/smoke_test.sh ~/smoke/fixtures ~/smoke/out_<label> cuda
python tests/smoke/compare_runs.py ~/smoke/out_pass2_5843365 ~/smoke/out_<label>
```

It trains on one terrain and tests on another, like e30.

| Fixture | Taken from | Nodes | Edges | Pairs |
|---|---|---|---|---|
| `smoke_train.npz` | Philadelphia `philly_res20_hybrid.npz` | 2,500 | 19,404 | 2,000 |
| `smoke_test.npz` | Norway `full_test-020.npz` | 10,000 | 78,804 | 2,000 |

Models: TAGConv (`configs/tagconv-philly.yml`), SparseGT random (`sparse-gt-random-philly.yml`) and SparseGT data (`sparse-gt-data-philly.yml`). All use batch 4096, p=4, sum+diff, lr 0.008 and 5 epochs.

Results before pass 2 (07b0d9d) and after it (5843365):

| Model | test_nmae, before | test_nmae, after | test_acc_5pct, before | test_acc_5pct, after |
|---|---|---|---|---|
| TAGConv | 0.090 | 0.196 | 0.377 | 0.172 |
| SparseGT random | 0.338 | 0.188 | 0.245 | 0.133 |
| SparseGT data | 0.262 | 0.313 | 0.001 | 0 |

- These numbers show only that the pipeline runs. With 5 epochs and 2,000 pairs, they do not compare models.
- Several fixes act together, so the table does not show which fix caused which change.
- Before bug 2 was fixed, the checkpoint was chosen on test-terrain data. This can make the old test numbers look better.
- `compare_runs.py` prints FAIL when a value moves more than its noise tolerance. After a bug fix, FAIL is expected.
- SparseGT random scores poorly (best_val_nmae 2.11). The likely cause is the short run, but this is not confirmed.
- On shared plaza GPUs, SparseGT random best_val_nmae can vary by 4.4e-3 relative between runs of the same code. Run a control on HEAD before you blame a change.

## Tests

- Run `pytest tests -q` in the conda env `spnn-asilomar` (Python 3.11, torch 2.5.1, PyG 2.6.1, torch_sparse 0.6.18). 97 tests pass on plaza.
- Training goldens are per platform (darwin-arm64, linux-x86_64). SparseGT tests need MKL, so they skip on the Mac.
- A change that is meant to alter results updates the goldens with `pytest --update-golden`. The commit says why.

## Open items

- The darwin-arm64 training golden is out of date since bug 2. Regenerate it on a Mac with `pytest --update-golden`.
- `mt_squirrel_repro` still has the line that stops SparseGT Q, K and V from training (bug 23).
- The header comment in `configs/tagconv-philly.yml` says hidden 32 and 2 layers. The config sets 64 and 3.
- The data CLIs do not seed `np.random`, so dataset generation is not reproducible.
