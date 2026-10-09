#!/usr/bin/env bash
# Usage:
#   CUDA_VISIBLE_DEVICES=0 bash scripts/e25TR_norway_hybrid_sampling.sh even
#   CUDA_VISIBLE_DEVICES=1 bash scripts/e25TR_norway_hybrid_sampling.sh odd
set -euo pipefail

SPLIT=${1:-}
if [ "$SPLIT" != "even" ] && [ "$SPLIT" != "odd" ]; then
  echo "Usage: CUDA_VISIBLE_DEVICES=X bash scripts/e25TR_norway_hybrid_sampling.sh <even|odd>"
  exit 1
fi

if [ "$SPLIT" = "even" ]; then
  #RESOLUTIONS=(20 18 16 14 12 10 8 6 4 2)
  RESOLUTIONS=(10)
else
  RESOLUTIONS=(19 17 15 13 11 9 7 5 3 1)
fi

PROJECT_ROOT="/home/teresa/shortest-paths-nn"
PYTHON_BIN="/home/shared/manifold-transformers/bin/python"
TEST_FILE="generated2/full_test-004.npz"

cd "$PROJECT_ROOT"

for RES in "${RESOLUTIONS[@]}"; do
  RES_PADDED=$(printf "%02d" "$RES")
  TRAIN_FILE_ABS="${PROJECT_ROOT}/data/res${RES_PADDED}_hybrid.npz"
  TRAIN_FILE_REL="res${RES_PADDED}_hybrid.npz"
  DATASET_NAME="norway/res${RES_PADDED}"

  echo "=================================================="
  echo "Resolution ${RES_PADDED}  |  split=${SPLIT}"
  echo "Train file: ${TRAIN_FILE_ABS}"
  echo "Test file:  ${TEST_FILE}"
  echo "=================================================="

  if [ ! -f "$TRAIN_FILE_ABS" ]; then
    echo "Missing dataset: $TRAIN_FILE_ABS"
    echo "Expected all Norway datasets to already exist; aborting."
    exit 1
  fi

  echo "=================================================="
  echo "Training TAGConv for resolution ${RES}"
  echo "=================================================="

  WANDB_PROJECT=manifold-transformers-dev "$PYTHON_BIN" train_single_terrain_case.py \
    --train-data "$TRAIN_FILE_REL" \
    --test-data "$TEST_FILE" \
    --epochs 100 \
    --resolution "$RES" \
    --device cuda \
    --batch-size 32 \
    --dataset-name "$DATASET_NAME" \
    --config configs/tagconv-k5.yml \
    --siamese 1 \
    --vn 0 \
    --layer-type TAGConv \
    --aggr 'sum+diff' \
    --p 1 \
    --loss mse_loss \
    --finetune 0 \
    --include-edge-attr 1 \
    --lr 0.0001 \
    --trial 1 \
    --new \
    --single-graph-full-batch \
    --wandb-tag e25TR_norway_hyprid_sampling

  echo "=================================================="
  echo "Training SparseGT for resolution ${RES}"
  echo "=================================================="

  WANDB_PROJECT=manifold-transformers-dev "$PYTHON_BIN" train_single_terrain_case.py \
    --train-data "$TRAIN_FILE_REL" \
    --test-data "$TEST_FILE" \
    --epochs 10 \
    --resolution "$RES" \
    --device cuda \
    --batch-size 32 \
    --dataset-name "$DATASET_NAME" \
    --config configs/sparse-gt-rpearl-k5.yml \
    --siamese 1 \
    --vn 0 \
    --layer-type SparseGT \
    --aggr 'sum+diff' \
    --p 1 \
    --loss mse_loss \
    --finetune 0 \
    --include-edge-attr 1 \
    --lr 0.0001 \
    --trial 1 \
    --new \
    --single-graph-full-batch \
    --wandb-tag e25TR_norway_hyprid_sampling

done