# e32: Optuna tuning of the SparseGT training recipe (2026-10-14)

## 1. Question

Which optimizer settings make SparseGT k=2 train stably and reach the lowest validation error in the e31 budget of 100 epochs?
e31 (sampling-scheme transferability, `docs/2026-10-e31-sampling-transfer.md`) used a constant rate with no warmup and dropout 0.5.
Its section 7.3 shows that this setup starts at 110% error and has spikes in the validation curve.
e33 will reuse the winning recipe from the command line.

## 2. What JP decided and what I chose

JP decided:
- a budget of about 2.5 h on the two A6000 GPUs of plaza;
- a small test set of 100,000 stride-4 pairs instead of the full 2.5M-pair test;
- 25,000 validation pairs instead of 2,500;
- wandb project `terrains-dev` for tuning.

I chose:
- the TPE sampler with `constant_liar=True`, so two parallel workers do not sample the same point;
- sampler seed 0 for the GPU 0 worker and seed 1 for the GPU 1 worker, because with one seed both workers draw the same random startup trials;
- a study-wide cap of 100 trials, so in practice the timeout stops the study.

## 3. Fixed setup

| Item | Value |
| --- | --- |
| Training data | `~/data/spnn/e31/norway_grid_s16.npz`: grid stride 16, 15,625 nodes, 500,000 pairs |
| Validation | 25,000 of the training pairs, split seed 42; training uses the other 475,000 |
| Model | SparseGT k=2, `configs/e31-sparse-gt-k2.yml`, RPEARL data mode, siamese L1 head (`p = 1`), MSE loss |
| Epochs and seed | 100 epochs, model seed 1 (as `--trial 1` in e31) |
| Objective | best validation nmae over the run; nmae = mean of abs(pred - true) / true |
| Small test set | `~/data/spnn/e32/full_test-004_100k.npz`: the full stride-4 graph (250,000 nodes), 100,000 of the 2.5M pairs of `full_test-004.npz` |
| Small test use | evaluated once at the end of each completed trial; no `preds.csv` |
| wandb | `alelab/terrains-dev`, one run per trial, name `e32-t<trial number>`, tags `e32` and `e32-sgt2` |

The small test set keeps all 10 sources of the full test.
The pairs come from `np.random.default_rng(0).choice(2_500_000, 100_000, replace=False)`, sorted.
`scripts/e32_sgt_tune.sh` builds the file if it is missing.

## 4. Search space

| Factor | Range |
| --- | --- |
| Peak learning rate | log-uniform, 3e-4 to 1e-2 |
| Warmup steps | 0, 300, 1000 (linear from 0) |
| Schedule after warmup | constant, or cosine decay to 1e-4 x peak at the last step |
| Dropout (`sparse_gt.dropout`) | 0, 0.1, 0.3, 0.5 |
| Attention dropout | 0, 0.2 |
| Batch size (pairs per step) | 8192, 16384 |
| AdamW weight decay | 0, 0.01, 0.1 |

Sampler: TPE with 5 random startup trials.

Pruning:
- `MedianPruner(n_startup_trials=5, n_warmup_steps=15)` on the validation nmae of every epoch. SparseGT needs about 15 epochs to fix its output scale, so no trial is pruned before epoch 15.
- A trial also stops when its validation nmae is NaN or infinite, or above 50% from epoch 30 on.

## 5. Overhead measurement and what I removed

The task said that e31 runs spent about 15 min outside the training steps.
The measurement does not support this.
I timed a 3-epoch run with the e31 SparseGT k=2 arguments on a free GPU (stride 16).

| Phase | Time | Count |
| --- | --- | --- |
| Data loading (`np.load` + `npz_to_dataset`, train and test) | 0.1 s | 1 |
| wandb init | 1.1 s | 1 |
| wandb finish | 0.7 s | 1 |
| Training loop, per epoch (31 steps) | 10.5 s | per epoch |
| wandb.log per batch | 0.4 ms | per step |
| Validation, 2,500 pairs | 0.05 s | per epoch |
| Final evaluation, 497,500 training pairs | 6.1 s | 1 |
| Final evaluation, 2.5M test pairs | 29.7 s | 1 |
| `preds.csv` write (2.5M rows) | 16.7 s | 1 |
| Python start and imports | about 4 s | 1 |

The e31 log of the same run (`~/runs/e31/norway_grid_s16_sgt2.log`) agrees: 100 epochs took 17 min 56 s at 10.8 s per epoch, of a 19 min 12 s run.
So the fixed overhead is about 1 min per run, and the training steps take the rest.

The training steps were slow for a reason that can be removed.
One isolated SparseGT step (forward, backward, AdamW) costs 0.13 s, but the loop ran at 0.34 s per step.
The difference is the `DataLoader`. It collates 16,384 scalar tensors one pair at a time, which costs about 0.2 s per batch.
The same collate made the final evaluations slow, at about 0.2 s per batch.

What I changed:
- `PairBatchLoader` (`spnn/data.py`) slices batches from tensors on the GPU. It yields the same batches as the `DataLoader` and leaves the global RNG in the same state (`tests/test_pair_loader.py`).
- `spnn.cli.tune` loads the training graph and the small test set once per worker and runs every trial in-process.
- The tuner skips the test evaluation inside the training function, so no `preds.csv` is written. It evaluates the small test set itself, only for completed trials.

Result for the same 3-epoch run with `--fast-loader --val-pairs 25000`:

| Phase | Before | After |
| --- | --- | --- |
| Training loop, per epoch | 10.5 s | 4.1 s |
| Final evaluation, training pairs | 6.1 s | 0.1 s |
| Final evaluation, 2.5M test pairs | 29.7 s | 0.7 s |
| Small test set, 100,000 pairs (tuner) | n/a | 0.4 s |

Per-batch wandb logging costs 0.4 ms per step, so I did not add an option to log every N batches.
wandb init costs about 1 s, so the study logs online.
`preds.csv` still costs 17 s in `spnn.cli.train`, which has no flag to skip it.

Expected trial length at 100 epochs: about 7 min at batch 16,384 and about 13 min at batch 8,192.
Two workers for 2.5 h then give about 25–30 completed trials, and more with pruning.

## 6. Code changes

| File | Change | Why |
| --- | --- | --- |
| `spnn/data.py` | `PairBatchLoader`; `n_val` argument of `split_dataset_for_validation` | 2.6x faster epochs; 25,000 validation pairs |
| `spnn/training.py` | `lr_factor_fn`, `LR_SCHEDULES`; new arguments `weight_decay`, `warmup_steps`, `lr_schedule`, `on_epoch_end`, `finish_wandb` | the search space; pruning; the tuner logs the small test into the open wandb run |
| `spnn/cli/train.py` | flags `--val-pairs`, `--weight-decay`, `--warmup-steps`, `--lr-schedule`, `--fast-loader` | e33 can run the winning recipe from the command line |
| `spnn/cli/tune.py` | new: the Optuna study | e32 |
| `scripts/e32_sgt_tune.sh` | new: one worker per GPU | e32 launcher |
| `tests/test_pair_loader.py` | new | checks that the loader gives the same batches as `DataLoader` |

The defaults keep the old behaviour: weight decay 0.01 (the AdamW default used before), no warmup and a constant rate.
`pytest tests -q` passes (113 tests) with the goldens unchanged, and `ruff check spnn --select F` is clean.
Optuna 5.0.0 is installed in the `spnn-asilomar` env.

## 7. How to run

```bash
setsid nohup bash scripts/e32_sgt_tune.sh 0 > ~/runs/e32/worker0.log 2>&1 < /dev/null &
setsid nohup bash scripts/e32_sgt_tune.sh 1 > ~/runs/e32/worker1.log 2>&1 < /dev/null &
```

Each worker stops starting new trials 9,000 s (2.5 h) after it starts.
A trial that runs at that time still finishes, so the study can end up to about 15 min later.
Env overrides: `STUDY`, `OUT`, `TIMEOUT`, `EPOCHS`, `NTRIALS`, `EXTRA` (more `spnn.cli.tune` flags).

To resume, start the same commands again.
The study is `load_if_exists`, so new trials join the stored ones.
A trial that was running when a worker died stays in state RUNNING in the database, and the sampler ignores it.
After a restart with the same sampler seed, the random startup trials can repeat earlier points.

To read the study:

```python
import optuna
s = optuna.load_study(study_name='e32-sgt2', storage='sqlite:////home/jporras/runs/e32/e32-sgt2.db')
s.trials_dataframe().sort_values('value')
```

## 8. Outputs

| Path | Content |
| --- | --- |
| `~/runs/e32/e32-sgt2.db` | Optuna study (SQLite, shared by the workers) |
| `~/runs/e32/trials/t<NNN>.json` | params, best validation nmae and epoch, validation curve, small-test metrics, state, stop reason, wall time, wandb URL |
| `~/runs/e32/models/runs/e32-t<N>_<wandb id>/` | best-validation model (`final_model.pt`) and training log |
| `~/runs/e32/worker<gpu>.log` | worker logs |
| `~/runs/e32/wandb/` | local wandb files |
| `~/runs/e32-smoke/` | smoke-test study and the overhead profile runs; not part of e32 |

## 9. Launch

| Worker | GPU | Start (EDT) | PID of `bash scripts/e32_sgt_tune.sh` | PID of `python -m spnn.cli.tune` |
| --- | --- | --- | --- | --- |
| 0 | 0 | 2026-10-14 11:44:56 | 263588 | 263594 |
| 1 | 1 | 2026-10-14 11:45:16 | 263950 | 263956 |

Smoke test before the launch, in the separate study `~/runs/e32-smoke/smoke.db`:
- Run A: 2 trials of 2 epochs. Both completed, wrote `t000.json` and `t001.json` and logged to `alelab/terrains-dev`. The small test took 0.4 s.
- Run B: 1 trial with the abort threshold forced to 1% at epoch 0. It stopped at epoch 0, Optuna marked it PRUNED, and its JSON has `state: pruned` and no small-test metrics.
- The 3 smoke runs had the `e32` tags. I retagged them `e32-smoke` in wandb (ids `5x1xkdlg`, `551tukor`, `6z8k6a0m`). The 2 profile runs have the tag `e32-profile`.

Check at 11:56 EDT, 11 min after the launch: both workers run trials at 86–90% GPU utilization.

| Trial | Batch | State | Best val nmae | Best epoch | Small-test nmae | Wall time |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 16384 | complete | 3.61% | 90 | 3.64% | 378 s |
| 1 | 16384 | complete | 1.97% | 99 | 2.06% | 407 s |
| 2 | 8192 | running, epoch 38 at 285 s | | | | |
| 3 | 8192 | running, epoch 29 at 240 s | | | | |

## 10. Results

The study ended at 14:21 EDT.
Worker 0 ran for 9,124 s and worker 1 for 9,364 s.

| State | Trials |
| --- | --- |
| Complete | 16 |
| Pruned | 23 |
| Total | 39 |

Of the 23 pruned trials, the median pruner stopped 20 and the 50% threshold stopped 3 (trials 2, 6 and 11).

### 10.1 Top 10 complete trials

Sorted by best validation nmae.
Small test = the 100,000-pair subset of the full stride-4 test, evaluated with the best-validation checkpoint.

| Trial | Val nmae | Small-test nmae | Small-test acc<2% | Best epoch | Peak lr | Warmup | Schedule | Dropout | Attn dropout | Batch | Weight decay |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 22 | 0.654% | 0.68% | 94.6% | 97 | 3.26e-3 | 300 | constant | 0.1 | 0.2 | 8192 | 0.1 |
| 19 | 0.660% | 0.69% | 94.1% | 95 | 3.05e-3 | 1000 | constant | 0.1 | 0.2 | 8192 | 0.1 |
| 23 | 0.665% | 0.70% | 94.2% | 89 | 3.45e-3 | 1000 | constant | 0.1 | 0.0 | 8192 | 0.1 |
| 13 | 0.665% | 0.70% | 93.7% | 96 | 4.10e-3 | 1000 | constant | 0.1 | 0.2 | 8192 | 0.1 |
| 21 | 0.674% | 0.72% | 93.8% | 92 | 4.95e-3 | 1000 | cosine | 0.1 | 0.0 | 8192 | 0.1 |
| 32 | 0.674% | 0.70% | 93.8% | 74 | 4.89e-3 | 300 | constant | 0.1 | 0.2 | 8192 | 0.1 |
| 9 | 0.679% | 0.70% | 93.5% | 99 | 2.02e-3 | 1000 | constant | 0.1 | 0.2 | 8192 | 0.1 |
| 31 | 0.693% | 0.69% | 94.1% | 58 | 5.51e-3 | 300 | constant | 0.1 | 0.0 | 8192 | 0.1 |
| 20 | 0.695% | 0.71% | 93.7% | 77 | 4.72e-3 | 0 | constant | 0.0 | 0.2 | 8192 | 0.1 |
| 18 | 0.713% | 0.71% | 93.7% | 99 | 7.11e-3 | 1000 | constant | 0.1 | 0.2 | 8192 | 0.1 |

Reference points:

| Run | Val nmae | Test nmae | Test set |
| --- | --- | --- | --- |
| e31 SparseGT k=2, grid stride 16: lr 3.5e-3 constant, dropout 0.5, attn dropout 0.2, batch 16,384, weight decay 0.01 | 0.97% | 1.00% | full, 2.5M pairs |
| e31 MLP, grid stride 16 | 0.72% | 0.77% | full, 2.5M pairs |
| e32 trial 22 | 0.65% | 0.68% | small, 100,000 pairs |

The e31 validation set had 2,500 pairs, and e32 uses 25,000.

### 10.2 Pattern

- Weight decay 0.1 is in all 12 trials with a best validation nmae under 0.73%. This includes the pruned trial 28 (0.72% at epoch 75). With weight decay 0.01 the best values are 1.39% (trial 6, pruned) and 1.50% (trial 3, complete).
- Dropout 0.1 is in 11 of the 12 best trials. The exception is trial 20 with dropout 0 at 0.70%. No trial with dropout 0.3 or 0.5 got under 0.78%, and all four dropout 0.5 trials were pruned or reached only 1.97%.
- Batch 8,192 is in all 12 best trials. All 8 trials with batch 16,384 were pruned (6) or completed at 1.97% and 3.61% (2).
- The 8 best trials have a peak lr from 2.0e-3 to 5.5e-3. Trials 18 (7.1e-3) and 37 (8.8e-3) still reach 0.71% and 0.72%. The trials with lr under 2e-3 and weight decay 0.1 were pruned at epoch 15 or reached 1.88%.
- Cosine decay is not better than a constant rate. The best cosine trial (21) is fifth, at 0.67%.
- Warmup 300 and warmup 1,000 give the same result: 0.654% (trial 22) and 0.660% (trial 19). Warmup 0 also reached 0.70% (trial 20).
- Attention dropout does not matter. Seven of the top 10 have 0.2 and three have 0. The best trial with 0 (trial 23) is at 0.665%.
- The top 10 trials are within 0.06 percentage points on validation and 0.04 on the small test. Each trial has one seed, so this ranking inside the top 10 is not reliable.
- In 6 of the top 10 trials the best epoch is 89 or later. The curve of trial 22 stays between 0.65% and 0.69% from epoch 70 on.

### 10.3 GPU time saved by pruning

| Item | GPU time |
| --- | --- |
| 16 complete trials | 3.29 h |
| 23 pruned trials, as run (747 epochs) | 1.84 h |
| 23 pruned trials, extended to 100 epochs at the median epoch time of their batch size | 4.78 h |
| 23 pruned trials, extended to 100 epochs at each trial's own epoch time | 5.59 h |

The median epoch time was 8.1 s at batch 8,192 and 4.1 s at batch 16,384.
Pruning therefore saved about 2.9 h of GPU time, or 3.75 h with each trial's own epoch time.
The second estimate is higher because some short trials have wall times of 11–19 s per epoch, which includes their fixed start-up time.
The whole study used 5.1 GPU hours; without pruning the same 39 trials would have needed about 8.1 GPU hours.

## 11. Confirmation

JP decided to confirm the trial 22 recipe through the normal CLI before more tuning.
The runs use `spnn.cli.train` through `tests/smoke/run_with_metrics.py`, with the e31 arguments and the full test set `~/data/spnn/full_test-004.npz` (2.5M pairs).
So their test nmae is comparable with the e31 table (`docs/2026-10-e31-sampling-transfer.md`, section 6).

### 11.1 Recipe

| Item | Value | e31 value |
| --- | --- | --- |
| Peak lr | 3.257e-3, the value of trial 22 | 3.5e-3 |
| Warmup | 300 steps, linear | 0 |
| Schedule | constant | constant |
| Dropout (`sparse_gt.dropout`) | 0.1 | 0.5 |
| Attention dropout | 0.2 | 0.2 |
| AdamW weight decay | 0.1 | 0.01 |
| Batch | 8,192 | 16,384 |
| Validation pairs | 25,000 | 2,500 |
| Loader | `--fast-loader` (`PairBatchLoader`) | `DataLoader` |
| Epochs, seed | 100, `--trial 1` | 100, `--trial 1` |

Configs: `configs/e32-sparse-gt-k2.yml` and `configs/e32-sparse-gt-k6.yml`.
Each is a copy of the e31 config with `sparse_gt.dropout` changed from 0.5 to 0.1 and a new header comment.
`factory.py` reads the keys `sparse_gt.dropout` and `sparse_gt.attn_dropout`, the same keys that `spnn.cli.tune` overrides.

Launcher: `scripts/e32_confirm.sh <gpu>`, a queue like the e31 one (mkdir locks, a run with a JSON is skipped, `preds.csv` gzipped).
wandb: project `terrains-dev`, run name `e32c-<run>`, tags `e32`, `e32-confirm`, `e32-sgt<k>`, `e32-s<stride>`.
Outputs: `~/runs/e32/confirm/e32c_<run>.json` and `.log`, run directories `~/runs/e32/confirm/e32c_<run>/`, worker logs `~/runs/e32/confirm/worker<gpu>.log`.

### 11.2 Dry run

I ran the s16 k=2 command for 2 epochs on GPU 0 at 15:06 EDT, with `WANDB_MODE=offline` and outputs in the session scratchpad.
It finished in 52 s and wrote the JSON with `final` and `val` entries and a gzipped `preds.csv` of 2.5M rows.
The log showed 25,000 validation pairs, 250,000 test nodes and the config with dropout 0.1 and attention dropout 0.2.

| Epoch | Val nmae, dry run | Val nmae, trial 22 |
| --- | --- | --- |
| 0 | 120.87% | 120.84% |
| 1 | 49.67% | 49.80% |

I deleted the dry-run outputs before the launch.

### 11.3 Launch

| GPU | Runs, in order | Start (EDT) | PID of `bash scripts/e32_confirm.sh` |
| --- | --- | --- | --- |
| 0 | grid_s16_sgt2, grid_s16_sgt6, grid_s08_sgt2 | 2026-10-14 15:07:43 | 317682 |
| 1 | grid_s08_sgt6 | 2026-10-14 15:07:40 | 317620 |

The training processes are `python tests/smoke/run_with_metrics.py`, one per GPU (PIDs 317686 and 317624 for the first runs).

### 11.4 Results

Test = full stride-4 test, 2.5M pairs. Best-validation checkpoint.

| Run | Val nmae | Test nmae | Test acc<2% | Train nmae | Wall time | e31 test nmae, same data and k |
| --- | --- | --- | --- | --- | --- | --- |
| grid_s16_sgt2 | 0.661% (epoch 84) | 0.68% | 94.4% | 0.66% | 882 s | 1.00% |
| grid_s16_sgt6 | 0.649% (epoch 96) | 0.66% | 94.9% | 0.65% | 3,136 s | 1.03% |
| grid_s08_sgt2 | 0.680% (epoch 99) | 0.62% | 95.4% | 0.69% | 2,636 s | 1.04% |
| grid_s08_sgt6 | 0.678% (epoch 99) | 0.62% | 95.5% | 0.68% | 12,373 s | 1.01% |

Train nmae is on the training pairs of the run's own graph, with the best-validation checkpoint.
The wandb runs are `s76hm30k`, `f3my26ex`, `5tkphhig` and `x2mnv9k2`, in table order, in `alelab/terrains-dev`.

References on the same full test: e31 MLP at grid stride 16 0.77%, at stride 8 0.72%; GAT+L1 replication of Chen 0.52%.

First result, grid stride 16, SparseGT k=2:
- The full-test nmae is 0.68%, the same as the small-test 0.68% of trial 22. So the CLI path reproduces the tuner path. For this run the 100,000-pair small test and the full test agree.
- The recipe lowers the e31 value from 1.00% to 0.68% and the acc<2% from 89.0% to 94.4%.
- SparseGT k=2 now beats the e31 MLP at the same stride (0.77%). It is still above the GAT+L1 replication (0.52%).
- The wandb run is `alelab/terrains-dev/s76hm30k`.

All four runs finished. GPU 0 ended its queue at 16:58 EDT and GPU 1 at 18:33 EDT.
The grid_s08_sgt6 run took 123 s per epoch, about 3.4 h in total. Its e31 run took about 71 min.

## 12. Architecture study (e32-arch)

### 12.1 Question

Section 10 tuned the training recipe for one architecture, the e31 SparseGT k=2 model.
This study asks the next question.
With that recipe fixed, which SparseGT architecture gives the lowest validation nmae on grid stride 16?

The data, the objective and the trial procedure are the same as in the optimizer study (section 10):
- Training graph `~/data/spnn/e31/norway_grid_s16.npz`, 25,000 validation pairs split off with seed 42.
- Objective: the best validation nmae over 100 epochs.
- A complete trial is also tested on the small test set `~/data/spnn/e32/full_test-004_100k.npz` (100,000 pairs).

### 12.2 Fixed recipe

The recipe is the e32-sgt2 winner, with the peak lr rounded from 3.257e-3 to 3.5e-3.

| Item | Value |
| --- | --- |
| Peak lr | 3.5e-3 |
| Warmup | 300 steps, linear |
| Schedule | constant |
| Dropout (`sparse_gt.dropout`) | 0.1 |
| Attention dropout | 0.2 |
| AdamW weight decay | 0.1 |
| Batch | 8,192 |
| Loader | `PairBatchLoader` (the fast loader) |
| RPEARL embedding mode | data |
| Epochs, model seed | 100, 1 |

### 12.3 Search space

| Factor | Config key | Values |
| --- | --- | --- |
| Attention layers | `sparse_gt.num_layers` | 3, 4, 6, 8 |
| Width | `sparse_gt.hidden_dim` | 64, 128, 256 |
| Heads | `sparse_gt.num_heads` | 4, 8 |
| Attention window k | `sparse_gt.num_hops` | 2, 4, 6 |
| RPEARL GCN layers | `sparse_gt.rpearl_num_layers` | 2, 4, 6 |
| RPEARL TAGConv order | `sparse_gt.pe_k` | 1, 2, 3 |

The base config is `configs/e32-sparse-gt-k2.yml`.
The tuner sets `gnn.constr.output` and `gnn.constr.hidden` to the trial's `hidden_dim`, so the node embedding has the attention width.
The MLP head is not used, because the pair head is the L1 distance (p = 1).
The space has 1,296 combinations.

### 12.4 Sampler and pruner

- Sampler: TPE with `constant_liar`, 6 random start-up trials, seed 10 on the GPU-0 worker and 11 on the GPU-1 worker.
- Pruner: `MedianPruner(n_startup_trials=5, n_warmup_steps=15)`, on the validation nmae of each epoch.
- Abort: a trial stops when its validation nmae is not finite, or above 50% after epoch 30.
- Out of memory: a trial that raises a CUDA out-of-memory error is marked pruned. The trial JSON then has a `stop_reason` that starts with `CUDA out of memory`.
- Timeout: each worker starts no new trial 3 h after its own python process starts. A trial that runs at the timeout still finishes.
- Study-wide cap: 200 complete or pruned trials. The timeout normally stops the study first.

Expected trial times from the earlier runs:
- A 6-layer, 128-wide, k=2 trial takes about 12.5 min.
- k=6 costs about 4 times as much, and width 256 about 2 times as much. So the largest trials can take about 1.5 h.

Memory check: before the launch I ran one epoch of the largest architecture on GPU 0.
It had 8 layers, width 256, 4 heads, k=6, 6 RPEARL layers and `pe_k` 3.
The peak GPU memory from `nvidia-smi` was 8.6 GB of 48 GB.

### 12.5 Implementation

- `spnn/cli/tune.py` has a `--space {optimizer,architecture}` option, which selects a function from the `SEARCH_SPACES` registry. The default is `optimizer`, so the e32-sgt2 command in section 10 still runs the same study.
- With `--space architecture`, the flags `--lr`, `--warmup-steps`, `--lr-schedule`, `--batch-size`, `--weight-decay`, `--dropout` and `--attn-dropout` are required. With `--space optimizer` they are rejected.
- `--pruner-startup-trials` sets the pruner start-up count separately from the TPE start-up count.
- Each trial JSON has the keys `space`, `params` (the searched values) and `recipe` (the training values).
- Launcher: `scripts/e32_arch_tune.sh <gpu>`. The GPU-1 worker first waits for `~/runs/e32/confirm/e32c_grid_s08_sgt6.json`, polling every 60 s.

Smoke test before the launch, on GPU 0, with a separate study database in the session scratchpad and `WANDB_MODE=offline`:
- 2 trials of 2 epochs each finished in 37 s. Both wrote a trial JSON with the small-test metrics.
- Trial 0 had width 64, and the model printout showed 64-wide input and output projections.
- I deleted the smoke database and its outputs.
- `ruff check spnn --select F` passed. `pytest tests -q` gave 113 passed, and the goldens did not change.

### 12.6 How to run and resume

```bash
setsid nohup bash scripts/e32_arch_tune.sh 0 > ~/runs/e32/arch/worker0.log 2>&1 < /dev/null &
setsid nohup bash scripts/e32_arch_tune.sh 1 > ~/runs/e32/arch/worker1.log 2>&1 < /dev/null &
```

- Study `e32-arch` in `~/runs/e32/e32-arch.db`, separate from `e32-sgt2.db`.
- Trial JSONs: `~/runs/e32/arch/trials/t<NNN>.json`. Models: `~/runs/e32/arch/models/`. Worker logs: `~/runs/e32/arch/worker<gpu>.log`.
- wandb: project `terrains-dev`, tags `e32` and `e32-arch`, run names `e32a-t<trial number>`.
- To resume, start the same command again. The worker loads the study from the database. A trial that was running when a worker died stays in the state RUNNING, and the new worker does not repeat it.

### 12.7 Launch

| GPU | Start (EDT) | PID of `bash scripts/e32_arch_tune.sh` | PID of `python -m spnn.cli.tune` |
| --- | --- | --- | --- |
| 0 | 2026-10-14 21:57:21 | 398681 | 398687 |
| 1 | 2026-10-14 21:57:27 | 398881 | 398889 |

The confirmation JSON already existed, so the GPU-1 worker did not wait.
Both workers stop starting trials at about 01:00 EDT on 2026-10-15.

### 12.8 Results

Pending.
