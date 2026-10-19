#!/usr/bin/env bash
# e33: edge-length ablation on grid stride 16 with the e32 recipe, full stride-4 test set
# (docs/2026-10-e33-edge-length.md). Runs:
#   tagconv      TAGConv, no edge lengths (control, e31 config)        lr 1e-3
#   tagconv_ew   TAGConv + exp edge-weight kernel (e33-tagconv-ew.yml) lr 1e-3
#   sgt2         SparseGT k=2, e32 config (reference, done in e32: 0.68%)
#   sgt1_eb      SparseGT k=1 + euclidean attention bias (JP's edge-only version)
#   sgt2_eb      SparseGT k=2 + euclidean attention bias
#   RUNS="tagconv tagconv_ew" setsid nohup bash scripts/e33_edge_length.sh 1 > ~/runs/e33/worker1.log 2>&1 < /dev/null &
# A run is skipped when its JSON exists or another worker holds its lock; delete $OUT/locks/<run> to retry.
# Env overrides: RUNS, EPOCHS, STRIDE, OUT, TAG.
set -uo pipefail
GPU=${1:?usage: e33_edge_length.sh <gpu>}
REPO=/home/jporras/sourcecode/shortest-paths-nn
PY=/home/jporras/miniconda3/envs/spnn-asilomar/bin/python
DATA=/home/jporras/data/spnn/e31
TEST=/home/jporras/data/spnn/full_test-004.npz
OUT=${OUT:-/home/jporras/runs/e33}
TAG=${TAG:-e33}
EPOCHS=${EPOCHS:-100}
STRIDE=${STRIDE:-16}
RUNS=${RUNS:-"tagconv tagconv_ew sgt1_eb sgt2_eb"}
export WANDB_PROJECT=${WANDB_PROJECT:-terrains-dev}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$OUT/locks"
cd "$REPO"

# model_args <run> -> the layer type, config and learning rate of each run
model_args() {
  case "$1" in
    tagconv)    echo "--layer-type TAGConv --config configs/e31-tagconv.yml --lr 0.001" ;;
    tagconv_ew) echo "--layer-type TAGConv --config configs/e33-tagconv-ew.yml --lr 0.001" ;;
    sgt2)       echo "--layer-type SparseGT --rpearl-embedding-mode data --config configs/e32-sparse-gt-k2.yml --lr 0.0035" ;;
    sgt1_eb)    echo "--layer-type SparseGT --rpearl-embedding-mode data --config configs/e33-sparse-gt-k1-eb.yml --lr 0.0035" ;;
    sgt2_eb)    echo "--layer-type SparseGT --rpearl-embedding-mode data --config configs/e33-sparse-gt-k2-eb.yml --lr 0.0035" ;;
    *) echo "unknown run $1" >&2; return 1 ;;
  esac
}

data="norway_grid_s$(printf %02d "$STRIDE")"
for r in $RUNS; do
  run="e33_${r}_s$(printf %02d "$STRIDE")"
  [ -f "$OUT/$run.json" ] && continue
  mkdir "$OUT/locks/$run" 2>/dev/null || continue   # another worker has it
  margs=$(model_args "$r") || { rmdir "$OUT/locks/$run"; continue; }
  echo "$(date +%F_%T) gpu$GPU start $run"
  start=$SECONDS
  # shellcheck disable=SC2086
  CUDA_VISIBLE_DEVICES=$GPU TERRAIN_OUTPUT_DIR="$OUT/$run" "$PY" tests/smoke/run_with_metrics.py "$OUT/$run.json.tmp" \
    --train-data "$DATA/$data.npz" --test-data "$TEST" $margs \
    --epochs "$EPOCHS" --batch-size 8192 --siamese 1 --p 1 --aggr sum+diff \
    --loss mse_loss --include-edge-attr 1 --new --trial 1 --device cuda \
    --val-pairs 25000 --weight-decay 0.1 --warmup-steps 300 --lr-schedule constant --fast-loader \
    --resolution "$STRIDE" --dataset-name "norway/e31/$data" \
    --run-name "$TAG-$r-s$STRIDE" --wandb-tag "$TAG" "$TAG-$r" \
    > "$OUT/$run.log" 2>&1
  status=$?
  if [ $status -eq 0 ] && [ -f "$OUT/$run.json.tmp" ]; then
    mv "$OUT/$run.json.tmp" "$OUT/$run.json"
    find "$OUT/$run" -name preds.csv -exec gzip -f {} \;
    echo "$(date +%F_%T) gpu$GPU done  $run $((SECONDS - start))s"
  else
    echo "$(date +%F_%T) gpu$GPU FAILED $run (exit $status) $((SECONDS - start))s, see $OUT/$run.log"
  fi
done
echo "$(date +%F_%T) gpu$GPU queue empty"
