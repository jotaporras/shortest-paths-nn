#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/teresa/shortest-paths-nn"
PYTHON_BIN="/home/shared/manifold-transformers/bin/python"
RAW_DATA="/home/teresa/terrain-data/data/norway-smallest.txt"
TEST_FILE="generated2/full_test-004.npz"

cd "$PROJECT_ROOT"

for RES in $(seq 20 -1 1); do
  RES_PADDED=$(printf "%02d" "$RES")
  TRAIN_FILE_ABS="${PROJECT_ROOT}/data/res${RES_PADDED}_hybrid.npz"
  TRAIN_FILE_REL="res${RES_PADDED}_hybrid.npz"
  DATASET_NAME="norway/res${RES_PADDED}"

  echo "=================================================="
  echo "Generating dataset for resolution ${RES}"
  echo "Train file: ${TRAIN_FILE_ABS}"
  echo "Test file:  ${TEST_FILE}"
  echo "=================================================="

  "$PYTHON_BIN" dataset/dataset.py \
    --name norway \
    --raw-data "$RAW_DATA" \
    --filename "$TRAIN_FILE_ABS" \
    --graph-resolution "$RES" \
    --dataset-size 500000 \
    --num-sources 500 \
    --sampling-technique hybrid \
    --edge-weight

  echo "=================================================="
  echo "Training TAGConv for resolution ${RES}"
  echo "=================================================="

  WANDB_PROJECT=terrains "$PYTHON_BIN" train_single_terrain_case.py \
    --train-data "$TRAIN_FILE_REL" \
    --test-data "$TEST_FILE" \
    --epochs 100 \
    --device cuda \
    --batch-size 32 \
    --dataset-name "$DATASET_NAME" \
    --config configs/tagconv-k5.yml \
    --siamese 1 \
    --vn 0 \
    --layer-type TAGConv \
    --aggr 'sum+diff' \
    --p 4 \
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

  WANDB_PROJECT=terrains "$PYTHON_BIN" train_single_terrain_case.py \
    --train-data "$TRAIN_FILE_REL" \
    --test-data "$TEST_FILE" \
    --epochs 1 \
    --device cuda \
    --batch-size 32 \
    --dataset-name "$DATASET_NAME" \
    --config configs/sparse-gt-rpearl-k5.yml \
    --siamese 1 \
    --vn 0 \
    --layer-type SparseGT \
    --aggr 'sum+diff' \
    --p 4 \
    --loss mse_loss \
    --finetune 0 \
    --include-edge-attr 1 \
    --lr 0.0001 \
    --trial 1 \
    --new \
    --single-graph-full-batch \
    --wandb-tag e25TR_norway_hyprid_sampling

done