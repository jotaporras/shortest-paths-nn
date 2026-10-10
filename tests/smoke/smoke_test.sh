#!/usr/bin/env bash
# Smoke test: three short single-stage runs on a cross-terrain pair (like e30).
#   models: TAGConv, SparseGT (RPEARL random), SparseGT (RPEARL data)
#   train:  <fixture_dir>/smoke_train.npz    test: <fixture_dir>/smoke_test.npz
# Each model writes <out_dir>/<model>.json (metrics), <out_dir>/<model>/ (preds.csv,
# final_model.pt, wandb files) and <out_dir>/<model>.log. Compare two out
# dirs with tests/smoke/compare_runs.py.
#
# Usage: bash tests/smoke/smoke_test.sh <fixture_dir> <out_dir> [device]
#   env: PYTHON (default: python), EPOCHS (default: 5),
#        WANDB_PROJECT (default: manifold-transformers-dev), WANDB_MODE (default: online)
# Runs are tagged "smoke" and "smoke-<out_dir name>" in wandb.
set -euo pipefail

FIX=$(cd "$1" && pwd)
mkdir -p "$2"
OUT=$(cd "$2" && pwd)
DEVICE=${3:-cuda}
PYTHON=${PYTHON:-python}
EPOCHS=${EPOCHS:-5}

cd "$(dirname "$0")/../.."
export WANDB_MODE=${WANDB_MODE:-online}
export WANDB_ENTITY=${WANDB_ENTITY:-alelab}
export WANDB_PROJECT=${WANDB_PROJECT:-manifold-transformers-dev}
LABEL=$(basename "$OUT")
# Record which code ran: git state, or the .sync_commit file written when the
# tree was copied to this machine without its .git directory.
if git rev-parse HEAD > "$OUT/commit.txt" 2>/dev/null; then
  git status --short >> "$OUT/commit.txt"
else
  cp .sync_commit "$OUT/commit.txt"
fi

run_model() {
  local name=$1
  shift
  local start=$SECONDS
  echo "running $name ..."
  DETERMINISTIC=${DETERMINISTIC:-0} TERRAIN_OUTPUT_DIR="$OUT/$name" "$PYTHON" tests/smoke/run_with_metrics.py "$OUT/$name.json" \
    --train-data "$FIX/smoke_train.npz" \
    --test-data "$FIX/smoke_test.npz" \
    --epochs "$EPOCHS" \
    --resolution 20 \
    --device "$DEVICE" \
    --batch-size 4096 \
    --dataset-name "smoke/$name" \
    --siamese 1 \
    --vn 0 \
    --aggr 'sum+diff' \
    --p 4 \
    --loss mse_loss \
    --finetune 0 \
    --include-edge-attr 1 \
    --trial 1 \
    --new \
    --single-graph-full-batch \
    --run-name "smoke-$name" \
    --wandb-tag smoke "smoke-$LABEL" \
    "$@" > "$OUT/$name.log" 2>&1
  echo "$name $((SECONDS - start))s" | tee -a "$OUT/timing.txt"
}

DETERMINISTIC=1 run_model tagconv --config configs/tagconv-philly.yml --layer-type TAGConv --lr 0.008
run_model sgt_random --config configs/sparse-gt-random-philly.yml --layer-type SparseGT \
  --rpearl-embedding-mode random --lr 0.008
run_model sgt_data --config configs/sparse-gt-data-philly.yml --layer-type SparseGT \
  --rpearl-embedding-mode data --lr 0.008

echo "done: $OUT"
