#!/usr/bin/env bash
# e32 confirmation: the trial 22 recipe of the e32 study through the normal CLI, on the full
# stride-4 test set, comparable with the e31 table (docs/2026-10-e32-sgt-tuning.md, "Confirmation").
#   RUNS="grid_s08_sgt6" setsid nohup bash scripts/e32_confirm.sh 1 > ~/runs/e32/confirm/worker1.log 2>&1 < /dev/null &
#   RUNS="grid_s16_sgt2 grid_s16_sgt6 grid_s08_sgt2" setsid nohup bash scripts/e32_confirm.sh 0 > ~/runs/e32/confirm/worker0.log 2>&1 < /dev/null &
# A run is <scheme>_s<stride>_sgt<k>. It is skipped when its JSON exists or another worker holds
# its lock. To retry a failed run, delete $OUT/locks/<run>.
# Env overrides: RUNS, EPOCHS, LR, OUT, TAG.
set -uo pipefail
GPU=${1:?usage: e32_confirm.sh <gpu>}
REPO=/home/jporras/sourcecode/shortest-paths-nn
PY=/home/jporras/miniconda3/envs/spnn-asilomar/bin/python
DATA=/home/jporras/data/spnn/e31
TEST=/home/jporras/data/spnn/full_test-004.npz
OUT=${OUT:-/home/jporras/runs/e32/confirm}
TAG=${TAG:-e32}
EPOCHS=${EPOCHS:-100}
LR=${LR:-0.003257}   # trial 22: 3.257e-3
RUNS=${RUNS:-"grid_s16_sgt2 grid_s16_sgt6 grid_s08_sgt2 grid_s08_sgt6"}
export WANDB_PROJECT=${WANDB_PROJECT:-terrains-dev}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$OUT/locks"
cd "$REPO"

for r in $RUNS; do
  scheme=${r%%_s*}; rest=${r#*_s}; stride=${rest%%_*}; model=${r##*_}; k=${model#sgt}
  data="norway_${scheme}_s${stride}"
  run="e32c_$r"
  [ -f "$OUT/$run.json" ] && continue
  mkdir "$OUT/locks/$run" 2>/dev/null || continue   # another worker has it
  if [ ! -f "$DATA/$data.npz" ]; then
    echo "$(date +%F_%T) gpu$GPU $run MISSING $DATA/$data.npz"; rmdir "$OUT/locks/$run"; continue
  fi
  echo "$(date +%F_%T) gpu$GPU start $run"
  start=$SECONDS
  CUDA_VISIBLE_DEVICES=$GPU TERRAIN_OUTPUT_DIR="$OUT/$run" "$PY" tests/smoke/run_with_metrics.py "$OUT/$run.json.tmp" \
    --train-data "$DATA/$data.npz" --test-data "$TEST" \
    --layer-type SparseGT --rpearl-embedding-mode data --config "configs/e32-sparse-gt-k$k.yml" \
    --epochs "$EPOCHS" --batch-size 8192 --lr "$LR" --siamese 1 --p 1 --aggr sum+diff \
    --loss mse_loss --include-edge-attr 1 --new --trial 1 --device cuda \
    --val-pairs 25000 --weight-decay 0.1 --warmup-steps 300 --lr-schedule constant --fast-loader \
    --resolution "$((10#$stride))" --dataset-name "norway/e31/$data" \
    --run-name "e32c-$r" --wandb-tag "$TAG" "$TAG-confirm" "$TAG-$model" "$TAG-s$stride" \
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
