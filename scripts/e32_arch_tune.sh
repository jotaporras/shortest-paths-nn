#!/usr/bin/env bash
# e32-arch: Optuna study of the SparseGT architecture with the e32 recipe fixed
# (design: docs/2026-10-e32-sgt-tuning.md, section 12).
# One worker per GPU; workers share the SQLite study, and a restart resumes it.
#   setsid nohup bash scripts/e32_arch_tune.sh 0 > ~/runs/e32/arch/worker0.log 2>&1 < /dev/null &
#   setsid nohup bash scripts/e32_arch_tune.sh 1 > ~/runs/e32/arch/worker1.log 2>&1 < /dev/null &
# The GPU-1 worker first waits for the e32 confirmation run grid_s08_sgt6 (WAIT_FOR, polled every 60 s).
# The timeout counts from the start of this worker's python process, after the wait.
# Env overrides (smoke test): STUDY, OUT, TIMEOUT (s), EPOCHS, NTRIALS, WAIT_FOR, EXTRA (more tune.py flags).
set -uo pipefail
GPU=${1:?usage: e32_arch_tune.sh <gpu>}
REPO=/home/jporras/sourcecode/shortest-paths-nn
DATA=/home/jporras/data/spnn
OUT=${OUT:-/home/jporras/runs/e32/arch}
STUDY=${STUDY:-e32-arch}
STORAGE=${STORAGE:-/home/jporras/runs/e32/$STUDY.db}
TIMEOUT=${TIMEOUT:-10800}  # 3 h; a trial running at the timeout still finishes
EPOCHS=${EPOCHS:-100}
NTRIALS=${NTRIALS:-200}    # study-wide cap; the timeout normally stops the study first
TEST=$DATA/e32/full_test-004_100k.npz
if [ "$GPU" = 1 ]; then
  WAIT_FOR=${WAIT_FOR:-/home/jporras/runs/e32/confirm/e32c_grid_s08_sgt6.json}
else
  WAIT_FOR=${WAIT_FOR:-}
fi
source ~/miniconda3/etc/profile.d/conda.sh
conda activate spnn-asilomar
export WANDB_PROJECT=terrains-dev
export TERRAIN_OUTPUT_DIR=$OUT   # wandb files go to $OUT/wandb, not the repo
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TQDM_DISABLE=1 PYTHONUNBUFFERED=1
mkdir -p "$OUT"
cd "$REPO"

if [ -n "$WAIT_FOR" ]; then
  echo "$(date +%F_%T) gpu$GPU waiting for $WAIT_FOR"
  until [ -f "$WAIT_FOR" ]; do sleep 60; done
  echo "$(date +%F_%T) gpu$GPU found $WAIT_FOR"
fi

echo "$(date +%F_%T) gpu$GPU start study $STUDY pid $$"
# Fixed recipe = the e32-sgt2 winner (section 10), lr rounded to 3.5e-3.
# shellcheck disable=SC2086
CUDA_VISIBLE_DEVICES=$GPU python -m spnn.cli.tune --space architecture \
  --study-name "$STUDY" --storage "$STORAGE" --out "$OUT" \
  --train-data "$DATA/e31/norway_grid_s16.npz" --test-data "$TEST" \
  --config configs/e32-sparse-gt-k2.yml --device cuda \
  --lr 3.5e-3 --warmup-steps 300 --lr-schedule constant --batch-size 8192 \
  --weight-decay 0.1 --dropout 0.1 --attn-dropout 0.2 \
  --sampler-seed 10 --n-startup-trials 6 --pruner-startup-trials 5 --pruner-warmup-epochs 15 \
  --abort-after-epoch 30 --abort-nmae 0.5 \
  --wandb-project terrains-dev --wandb-tags e32 e32-arch --run-prefix e32a \
  --timeout "$TIMEOUT" --n-trials "$NTRIALS" --epochs "$EPOCHS" --gpu-tag "$GPU" ${EXTRA:-}
status=$?
echo "$(date +%F_%T) gpu$GPU exit $status"
