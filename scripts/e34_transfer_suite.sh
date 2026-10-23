#!/usr/bin/env bash
# e34: full transferability suite on Norway (docs/2026-10-e34-transfer-suite.md).
# Train on one scheme/stride graph, test on the fixed stride-4 grid full_test-004 (250k nodes).
# Suite C + stride 4 (JP, 2026-10-15): 5 schemes x strides 32 24 16 12 8 4 x {MLP, TAGConv K=3, SparseGT k=2},
# all with the e32 recipe (batch 8192, wd 0.1, warmup 300, constant lr, 25k val pairs, fast loader).
# SparseGT runs 200 epochs, the light models 100.
#
# Workers pull from one queue (atomic mkdir locks). On plaza, two SparseGT workers
# and one worker for the light models, which share GPU 1:
#   MODELS=sgt2 setsid nohup bash scripts/e34_transfer_suite.sh 0 > ~/runs/e34/worker0.log 2>&1 < /dev/null &
#   MODELS=sgt2 setsid nohup bash scripts/e34_transfer_suite.sh 1 > ~/runs/e34/worker1.log 2>&1 < /dev/null &
#   MODELS="tagconv mlp" setsid nohup bash scripts/e34_transfer_suite.sh 1 > ~/runs/e34/worker1_light.log 2>&1 < /dev/null &
# A run is skipped when its JSON exists. To retry a failed run, delete $OUT/locks/<run>.
# A run whose training set does not exist yet is skipped without a lock; the worker
# makes another pass every $WAIT seconds until every set exists.
# Env overrides: EPOCHS (all models), LR, STRIDES, SCHEMES, MODELS, OUT, TAG, WAIT, LAPPE_K (sgt2lap), SPNN_LAPPE_CACHE.
set -uo pipefail
GPU=${1:?usage: e34_transfer_suite.sh <gpu>}
REPO=/home/jporras/sourcecode/shortest-paths-nn
PY=/home/jporras/miniconda3/envs/spnn-asilomar/bin/python
DATA=${DATA:-/home/jporras/data/spnn/e31}  # training sets; DSET labels them in wandb
DSET=${DSET:-e31}
TEST=/home/jporras/data/spnn/full_test-004.npz
OUT=${OUT:-/home/jporras/runs/e34}
TAG=${TAG:-e34}
EPOCHS=${EPOCHS:-}                       # empty: per-model count from model_epochs
LR=${LR:-}                               # empty: per-model rate from model_lr
WAIT=${WAIT:-120}
STRIDES=${STRIDES:-"32 24 16 12 8 4"}    # small graphs first; stride 4 (250k nodes) last
SCHEMES=${SCHEMES:-"grid uniform poisson-disk slope-adaptive quadtree"}
MODELS=${MODELS:-"sgt2 tagconv mlp"}
export WANDB_PROJECT=${WANDB_PROJECT:-terrains}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$OUT/locks"
cd "$REPO"

model_args() {
  case $1 in
    mlp)     echo "--layer-type MLP --config configs/e31-mlp.yml" ;;
    tagconv) echo "--layer-type TAGConv --config configs/e31-tagconv.yml" ;;
    sgt2)    echo "--layer-type SparseGT --rpearl-embedding-mode data --config configs/e32-sparse-gt-k2.yml" ;;
    # e35: same SparseGT with Laplacian eigenvector encodings; LAPPE_K from the e35 stride-12 tuning.
    sgt2lap) echo "--layer-type SparseGT --rpearl-embedding-mode lappe --lappe-k ${LAPPE_K:-64} --config configs/e35-sparse-gt-k2-lappe.yml" ;;
    *) echo "unknown model $1" >&2; return 1 ;;
  esac
}

# Learning rate per model: SparseGT from the e32 Optuna study, MLP and TAGConv
# from the e31 pilot (the better of 1e-3 and 3.5e-3 on validation).
model_lr() {
  case $1 in
    tagconv) echo 0.001 ;;
    *)       echo 0.0035 ;;
  esac
}

model_epochs() {
  case $1 in
    sgt2|sgt2lap) echo 200 ;;
    *)    echo 100 ;;
  esac
}

while :; do
  missing=0
  for stride in $STRIDES; do
    for scheme in $SCHEMES; do
      for model in $MODELS; do
        data=$(printf "norway_%s_s%02d" "$scheme" "$stride")
        run="${data}_${model}"
        [ -f "$OUT/$run.json" ] && continue
        if [ ! -f "$DATA/$data.npz" ] || [ ! -f "$DATA/$data.json" ]; then
          missing=$((missing + 1)); continue   # still being generated; retried next pass
        fi
        mkdir "$OUT/locks/$run" 2>/dev/null || continue   # another worker has it
        echo "$(date +%F_%T) gpu$GPU start $run"
        start=$SECONDS
        # shellcheck disable=SC2046
        CUDA_VISIBLE_DEVICES=$GPU TERRAIN_OUTPUT_DIR="$OUT/$run" "$PY" tests/smoke/run_with_metrics.py "$OUT/$run.json.tmp" \
          --train-data "$DATA/$data.npz" --test-data "$TEST" \
          $(model_args "$model") \
          --epochs "${EPOCHS:-$(model_epochs "$model")}" --batch-size 8192 --lr "${LR:-$(model_lr "$model")}" \
          --siamese 1 --p 1 --aggr sum+diff --loss mse_loss --include-edge-attr 1 --new --trial 1 --device cuda \
          --val-pairs 25000 --weight-decay 0.1 --warmup-steps 300 --lr-schedule constant --fast-loader \
          --resolution "$stride" --dataset-name "norway/$DSET/$data" \
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
  [ "$missing" -eq 0 ] && break
  echo "$(date +%F_%T) gpu$GPU $missing runs wait for their training set; next pass in ${WAIT}s"
  sleep "$WAIT"
done
echo "$(date +%F_%T) gpu$GPU queue empty"
