#!/usr/bin/env bash
# e31: sampling-scheme transferability on Norway (design: docs/2026-10-e31-sampling-transfer.md).
# Train on one scheme/partition graph, test on the fixed stride-4 grid full_test-004 (250k nodes).
#
# Workers pull from one queue (atomic mkdir locks). On plaza, two SparseGT workers
# and one worker for the light models, which share GPU 1 (their epochs are bound
# by data loading, not the GPU):
#   MODELS="sgt6 sgt2" setsid nohup bash scripts/e31_sampling_transfer.sh 0 > ~/runs/e31/worker0.log 2>&1 < /dev/null &
#   MODELS="sgt6 sgt2" setsid nohup bash scripts/e31_sampling_transfer.sh 1 > ~/runs/e31/worker1.log 2>&1 < /dev/null &
#   MODELS="tagconv mlp" setsid nohup bash scripts/e31_sampling_transfer.sh 1 > ~/runs/e31/worker1_light.log 2>&1 < /dev/null &
# A run is skipped when its JSON exists. To retry a failed run, delete $OUT/locks/<run>.
# Env overrides (pilot): EPOCHS, LR, STRIDES, SCHEMES, MODELS, OUT, TAG.
set -uo pipefail
GPU=${1:?usage: e31_sampling_transfer.sh <gpu>}
REPO=/home/jporras/sourcecode/shortest-paths-nn
PY=/home/jporras/miniconda3/envs/spnn-asilomar/bin/python
DATA=/home/jporras/data/spnn/e31
TEST=/home/jporras/data/spnn/full_test-004.npz
OUT=${OUT:-/home/jporras/runs/e31}
TAG=${TAG:-e31}
EPOCHS=${EPOCHS:-100}
LR=${LR:-}                               # empty: per-model rate from model_lr
STRIDES=${STRIDES:-"32 16 8"}            # small graphs first, so a late stop only loses large runs
SCHEMES=${SCHEMES:-"grid uniform poisson-disk slope-adaptive quadtree"}
MODELS=${MODELS:-"sgt6 sgt2 tagconv mlp"}  # longest first within a partition
export WANDB_PROJECT=${WANDB_PROJECT:-terrains}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$OUT/locks"
cd "$REPO"

model_args() {
  case $1 in
    mlp)     echo "--layer-type MLP --config configs/e31-mlp.yml" ;;
    tagconv) echo "--layer-type TAGConv --config configs/e31-tagconv.yml" ;;
    sgt2)    echo "--layer-type SparseGT --rpearl-embedding-mode data --config configs/e31-sparse-gt-k2.yml" ;;
    sgt6)    echo "--layer-type SparseGT --rpearl-embedding-mode data --config configs/e31-sparse-gt-k6.yml" ;;
    *) echo "unknown model $1" >&2; return 1 ;;
  esac
}

# Learning rate per model, the better of 1e-3 and 3.5e-3 on validation in the
# 30-epoch pilot (uniform, stride 32; docs/2026-10-e31-sampling-transfer.md).
model_lr() {
  case $1 in
    tagconv) echo 0.001 ;;
    *)       echo 0.0035 ;;
  esac
}

for stride in $STRIDES; do
  for scheme in $SCHEMES; do
    for model in $MODELS; do
      data=$(printf "norway_%s_s%02d" "$scheme" "$stride")
      run="${data}_${model}"
      [ -f "$OUT/$run.json" ] && continue
      mkdir "$OUT/locks/$run" 2>/dev/null || continue   # another worker has it
      if [ ! -f "$DATA/$data.npz" ]; then
        echo "$(date +%F_%T) gpu$GPU $run MISSING $DATA/$data.npz"; rmdir "$OUT/locks/$run"; continue
      fi
      echo "$(date +%F_%T) gpu$GPU start $run"
      start=$SECONDS
      # shellcheck disable=SC2046
      CUDA_VISIBLE_DEVICES=$GPU TERRAIN_OUTPUT_DIR="$OUT/$run" "$PY" tests/smoke/run_with_metrics.py "$OUT/$run.json.tmp" \
        --train-data "$DATA/$data.npz" --test-data "$TEST" \
        $(model_args "$model") \
        --epochs "$EPOCHS" --batch-size 16384 --lr "${LR:-$(model_lr "$model")}" --siamese 1 --p 1 --aggr sum+diff \
        --loss mse_loss --include-edge-attr 1 --new --trial 1 --device cuda \
        --resolution "$stride" --dataset-name "norway/e31/$data" \
        --run-name "$TAG-$run" --wandb-tag "$TAG" "$TAG-$scheme" "$TAG-s$stride" "$TAG-$model" \
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
  done
done
echo "$(date +%F_%T) gpu$GPU queue empty"
