#!/usr/bin/env bash
# e32: Optuna tuning of the SparseGT k=2 training recipe (design: docs/2026-10-e32-sgt-tuning.md).
# One worker per GPU; workers share the SQLite study, and a restart resumes it.
#   setsid nohup bash scripts/e32_sgt_tune.sh 0 > ~/runs/e32/worker0.log 2>&1 < /dev/null &
#   setsid nohup bash scripts/e32_sgt_tune.sh 1 > ~/runs/e32/worker1.log 2>&1 < /dev/null &
# Env overrides (smoke test): STUDY, OUT, TIMEOUT (s), EPOCHS, NTRIALS, EXTRA (more tune.py flags).
set -uo pipefail
GPU=${1:?usage: e32_sgt_tune.sh <gpu>}
REPO=/home/jporras/sourcecode/shortest-paths-nn
DATA=/home/jporras/data/spnn
OUT=${OUT:-/home/jporras/runs/e32}
STUDY=${STUDY:-e32-sgt2}
TIMEOUT=${TIMEOUT:-9000}   # 2.5 h; a trial running at the timeout still finishes
EPOCHS=${EPOCHS:-100}
NTRIALS=${NTRIALS:-100}    # study-wide cap; the timeout normally stops the study first
TEST=$DATA/e32/full_test-004_100k.npz
source ~/miniconda3/etc/profile.d/conda.sh
conda activate spnn-asilomar
export WANDB_PROJECT=terrains-dev
export TERRAIN_OUTPUT_DIR=$OUT   # wandb files go to $OUT/wandb, not the repo
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TQDM_DISABLE=1 PYTHONUNBUFFERED=1
mkdir -p "$OUT"
cd "$REPO"

# The small test set: the full stride-4 test graph, 100,000 of its 2.5M pairs (rng seed 0).
if [ ! -f "$TEST" ]; then
  mkdir -p "$DATA/e32"
  python - "$DATA/full_test-004.npz" "$TEST" <<'EOF'
import sys
import numpy as np
d = np.load(sys.argv[1])
idx = np.sort(np.random.default_rng(0).choice(len(d['srcs']), 100_000, replace=False))
out = {k: d[k] for k in d.files}
for k in ('srcs', 'tars', 'lengths'):
    out[k] = d[k][idx]
np.savez(sys.argv[2], **out)
EOF
fi

echo "$(date +%F_%T) gpu$GPU start study $STUDY pid $$"
# shellcheck disable=SC2086
CUDA_VISIBLE_DEVICES=$GPU python -m spnn.cli.tune \
  --study-name "$STUDY" --storage "$OUT/$STUDY.db" --out "$OUT" \
  --train-data "$DATA/e31/norway_grid_s16.npz" --test-data "$TEST" \
  --config configs/e31-sparse-gt-k2.yml --device cuda \
  --timeout "$TIMEOUT" --n-trials "$NTRIALS" --epochs "$EPOCHS" --gpu-tag "$GPU" ${EXTRA:-}
echo "$(date +%F_%T) gpu$GPU exit $?"
