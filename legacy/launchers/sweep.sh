#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/teresa/shortest-paths-nn"
PYTHON_BIN="/home/shared/manifold-transformers/bin/python"
TRAIN_FILE="res04_hybrid.npz"
TEST_FILE="generated2/full_test-004.npz"
DATASET_NAME="norway/res04"
WANDB_TAG="e27TR_lr_sweep"

cd "$PROJECT_ROOT"

LR_VALUES=(0.01 0.005 0.001 0.0005 0.0001 0.00005 0.00001)

for LR in "${LR_VALUES[@]}"; do
  echo "=================================================="
  echo "TAGConv  |  lr=${LR}"
  echo "=================================================="

  WANDB_PROJECT=manifold-transformers-dev "$PYTHON_BIN" train_single_terrain_case.py \
    --train-data "$TRAIN_FILE" \
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
    --lr "$LR" \
    --trial "lr${LR}" \
    --new \
    --single-graph-full-batch \
    --wandb-tag "$WANDB_TAG" TAGConv "lr-${LR}"

  echo "=================================================="
  echo "SparseGT  |  lr=${LR}"
  echo "=================================================="

  WANDB_PROJECT=manifold-transformers-dev "$PYTHON_BIN" train_single_terrain_case.py \
    --train-data "$TRAIN_FILE" \
    --test-data "$TEST_FILE" \
    --epochs 10 \
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
    --lr "$LR" \
    --trial "lr${LR}" \
    --new \
    --single-graph-full-batch \
    --wandb-tag "$WANDB_TAG" SparseGT "lr-${LR}"
done
