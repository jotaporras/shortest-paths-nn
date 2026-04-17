#!/usr/bin/env bash
# Usage:
#   CUDA_VISIBLE_DEVICES=0 bash scripts/e27TR_philly_hybrid_sampling.sh even
#   CUDA_VISIBLE_DEVICES=1 bash scripts/e27TR_philly_hybrid_sampling.sh odd
set -euo pipefail

SPLIT=${1:-}
if [ "$SPLIT" != "even" ] && [ "$SPLIT" != "odd" ]; then
  echo "Usage: CUDA_VISIBLE_DEVICES=X bash scripts/e27TR_philly_hybrid_sampling.sh <even|odd>"
  exit 1
fi

if [ "$SPLIT" = "even" ]; then
  RESOLUTIONS=(6 4)
else
  RESOLUTIONS=(1)
fi

PROJECT_ROOT="/home/teresa/shortest-paths-nn"
PYTHON_BIN="/home/shared/manifold-transformers/bin/python"
RAW_DATA="/home/teresa/terrain-data/data/flat-phil-data-1.txt"
TEST_FILE_ABS="data/generated2/philly_test_res04.npz"
TEST_FILE_REL="generated2/philly_test_res04.npz"

cd "$PROJECT_ROOT"

# Generate Philadelphia test dataset at res=4 (once, before training loop)
if [ ! -f "$TEST_FILE_ABS" ]; then
  echo "Generating Philadelphia test dataset: $TEST_FILE_ABS"
  mkdir -p "$(dirname "$TEST_FILE_ABS")"
  "$PYTHON_BIN" dataset/dataset.py \
    --name phil \
    --raw-data "$RAW_DATA" \
    --filename "$TEST_FILE_ABS" \
    --graph-resolution 4 \
    --dataset-size 50000 \
    --num-sources 100 \
    --sampling-technique single-source-random \
    --edge-weight
fi

for RES in "${RESOLUTIONS[@]}"; do
  RES_PADDED=$(printf "%02d" "$RES")
  TRAIN_FILE_ABS="${PROJECT_ROOT}/data/philly_res${RES_PADDED}_hybrid.npz"
  TRAIN_FILE_REL="philly_res${RES_PADDED}_hybrid.npz"
  DATASET_NAME="philadelphia/res${RES_PADDED}"

  echo "=================================================="
  echo "Resolution ${RES_PADDED}  |  split=${SPLIT}"
  echo "Train file: ${TRAIN_FILE_ABS}"
  echo "Test file:  ${TEST_FILE_ABS}"
  echo "=================================================="

  if [ ! -f "$TRAIN_FILE_ABS" ]; then
    "$PYTHON_BIN" dataset/dataset.py \
      --name phil \
      --raw-data "$RAW_DATA" \
      --filename "$TRAIN_FILE_ABS" \
      --graph-resolution "$RES" \
      --dataset-size 500000 \
      --num-sources 500 \
      --sampling-technique hybrid \
      --edge-weight
  else
    echo "Dataset already exists, skipping generation."
  fi

  echo "=================================================="
  echo "Training TAGConv for resolution ${RES}"
  echo "=================================================="

  WANDB_PROJECT=terrains "$PYTHON_BIN" train_single_terrain_case.py \
    --train-data "$TRAIN_FILE_REL" \
    --test-data "$TEST_FILE_REL" \
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
    --p 4 \
    --loss mse_loss \
    --finetune 0 \
    --include-edge-attr 1 \
    --lr 0.0001 \
    --trial 1 \
    --new \
    --single-graph-full-batch \
    --wandb-tag e27TR_philly_hybrid_sampling

  echo "=================================================="
  echo "Training SparseGT-Random for resolution ${RES}"
  echo "=================================================="

  WANDB_PROJECT=terrains "$PYTHON_BIN" train_single_terrain_case.py \
    --train-data "$TRAIN_FILE_REL" \
    --test-data "$TEST_FILE_REL" \
    --epochs 100 \
    --resolution "$RES" \
    --device cuda \
    --batch-size 32 \
    --dataset-name "$DATASET_NAME" \
    --config configs/sparse-gt-rpearl-philly.yml \
    --rpearl-embedding-mode data \
    --siamese 1 \
    --vn 0 \
    --layer-type SparseGT \
    --aggr 'sum+diff' \
    --p 4 \
    --loss mse_loss \
    --finetune 0 \
    --include-edge-attr 1 \
    --lr 0.01 \
    --trial 1 \
    --new \
    --single-graph-full-batch \
    --wandb-tag e27TR_philly_hybrid_sampling

done
