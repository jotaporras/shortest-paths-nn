#!/usr/bin/env bash
# e35: tune the number of Laplacian eigenvectors (lappe_k in 2 4 8 16 32 64) for the SparseGT k=2 model
# with the e32 recipe fixed (lr 3.5e-3, warmup 300, constant, batch 8192, wd 0.1, dropout 0.1/0.2).
# A grid sampler enumerates the six values once; two workers share the SQLite study.
#   setsid nohup bash scripts/e35_lappe_tune.sh 0 > ~/runs/e35/tune_s12/worker0.log 2>&1 < /dev/null &
#   setsid nohup bash scripts/e35_lappe_tune.sh 1 > ~/runs/e35/tune_s12/worker1.log 2>&1 < /dev/null &
# The curiosity sweep on stride 4: STRIDE=04 OUT=~/runs/e35/tune_s04 STUDY=e35-lappe-s04.
# CAUTION: trial records are $OUT/trials/t<NNN>.json, numbered per study. Two studies that share $OUT
# overwrite each other's records (happened on 2026-10-20 with the two single-worker stride-4 studies);
# give every study its own OUT.
# Env overrides: STRIDE, OUT, STUDY, EPOCHS, EXTRA (more tune.py flags).
set -uo pipefail
GPU=${1:?usage: e35_lappe_tune.sh <gpu>}
REPO=/home/jporras/sourcecode/shortest-paths-nn
DATA=/home/jporras/data/spnn
STRIDE=${STRIDE:-12}
OUT=${OUT:-/home/jporras/runs/e35/tune_s$STRIDE}
STUDY=${STUDY:-e35-lappe-s$STRIDE}
EPOCHS=${EPOCHS:-100}
TEST=$DATA/e32/full_test-004_100k.npz   # the e32 small test set (100k pairs of the stride-4 grid test)
source ~/miniconda3/etc/profile.d/conda.sh
conda activate spnn-asilomar
export WANDB_PROJECT=terrains-dev
export TERRAIN_OUTPUT_DIR=$OUT
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TQDM_DISABLE=1 PYTHONUNBUFFERED=1
mkdir -p "$OUT"
cd "$REPO"

echo "$(date +%F_%T) gpu$GPU start study $STUDY pid $$"
# shellcheck disable=SC2086
CUDA_VISIBLE_DEVICES=$GPU python -m spnn.cli.tune \
  --study-name "$STUDY" --storage "$OUT/$STUDY.db" --out "$OUT" \
  --train-data "$DATA/e31/norway_grid_s$STRIDE.npz" --test-data "$TEST" \
  --config configs/e35-sparse-gt-k2-lappe.yml --space lappe --sampler grid --device cuda \
  --lr 0.0035 --warmup-steps 300 --lr-schedule constant --batch-size 8192 --weight-decay 0.1 \
  --dropout 0.1 --attn-dropout 0.2 \
  --n-trials 6 --epochs "$EPOCHS" --gpu-tag "$GPU" --abort-after-epoch 1000 --pruner-startup-trials 6 \
  --wandb-tags e35 "e35-lappe-s$STRIDE" --run-prefix "e35-s$STRIDE" ${EXTRA:-}
echo "$(date +%F_%T) gpu$GPU exit $?"
