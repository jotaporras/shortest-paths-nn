#!/usr/bin/env bash
# Usage:
#   CUDA_VISIBLE_DEVICES=0 bash scripts/e28TR_philly_hybrid_sampling_full_test.sh
set -euo pipefail

RESOLUTIONS=(20 19 18 17 16 15 14 13 12 11 10 9 8 7 6 5 4 3 2 1)

PROJECT_ROOT="/home/teresa/shortest-paths-nn"
PYTHON_BIN="/home/teresa/miniconda3/envs/manifold-transformers/bin/python"
RAW_DATA="/home/teresa/terrain-data/data/flat-phil-data-1.txt"
TEST_FILE_ABS="data/generated2/philly_test_1k_sources.npz"
TEST_FILE_REL="generated2/philly_test_1k_sources.npz"

cd "$PROJECT_ROOT"

for RES in "${RESOLUTIONS[@]}"; do
  RES_PADDED=$(printf "%02d" "$RES")
  TRAIN_FILE_ABS="${PROJECT_ROOT}/data/philly_res${RES_PADDED}_hybrid.npz"
  TRAIN_FILE_REL="philly_res${RES_PADDED}_hybrid.npz"
  DATASET_NAME="philadelphia/res${RES_PADDED}"

  echo "=================================================="
  echo "Resolution ${RES_PADDED}"
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
    --wandb-tag e28TR_philly_hybrid_sampling_full_test

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
    --rpearl-embedding-mode random \
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
    --wandb-tag e28TR_philly_hybrid_sampling_full_test

  echo "=================================================="
  echo "Training SparseGT-Data for resolution ${RES}"
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
    --wandb-tag e28TR_philly_hybrid_sampling_full_test
done
