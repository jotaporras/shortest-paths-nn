# LOG NUMBER OF SOURCES TO WANGB NEXT TIME

#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/teresa/shortest-paths-nn"
PYTHON_BIN="/home/shared/manifold-transformers/bin/python"
RAW_DATA="/home/teresa/terrain-data/data/norway-smallest.txt"
TEST_FILE="generated2/full_test-004.npz"

cd "$PROJECT_ROOT"

for num_sources in $(seq 5 5 55); do
  TRAIN_FILE_ABS="${PROJECT_ROOT}/data/res04_${num_sources}_sources.npz"
  TRAIN_FILE_REL="res04_${num_sources}_sources.npz"
  DATASET_NAME="norway/res04_${num_sources}_sources"
  NUM_PAIRS=$((num_sources*100))

  echo "=================================================="
  echo "Generating dataset for ${num_sources} sources"
  echo "Train file: ${TRAIN_FILE_ABS}"
  echo "Test file:  ${TEST_FILE}"
  echo "=================================================="

  "$PYTHON_BIN" dataset/dataset.py \
    --name norway \
    --raw-data "$RAW_DATA" \
    --filename "$TRAIN_FILE_ABS" \
    --graph-resolution 4 \
    --dataset-size $NUM_PAIRS \
    --num-sources "$num_sources" \
    --sampling-technique hybrid \
    --edge-weight

  echo "=================================================="
  echo "Training TAGConv for ${num_sources} sources"
  echo "=================================================="

  WANDB_PROJECT=terrains "$PYTHON_BIN" train_single_terrain_case.py \
    --train-data "$TRAIN_FILE_REL" \
    --test-data "$TEST_FILE" \
    --epochs 30 \
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
    --wandb-tag e26TR_norway_transferability_num_pairs

  echo "=================================================="
  echo "Training SparseGT for ${num_sources} sources"
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
    --wandb-tag e26TR_norway_transferability_num_pairs

done

