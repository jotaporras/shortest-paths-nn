#!/bin/bash
#SBATCH --job-name=e28TR_philly_hybrid_sampling_full_test
#SBATCH --output=/vast/projects/aribeiro/alelab/jporras/manifold-transformers/slurm-%A_%a.out
#SBATCH --time=55:00:00
#SBATCH --partition=b200-mig45
#SBATCH --gpus=1
#SBATCH --cpus-per-task=14
#SBATCH --mem=256G
#SBATCH --array=0-19
set -euo pipefail

# ── Cluster-specific paths (edit these) ───────────────────────────
CONDA_ENV="manifold-transformers"       # conda env name
PROJECT_ROOT="$HOME/shortest-paths-nn"

# ── Data staging ──────────────────────────────────────────────────
# Option A (default): read data directly from the shared filesystem.
# Option B: copy to node-local scratch for faster I/O.
#   Set USE_LOCAL_SCRATCH=1 and make sure $TMPDIR or $SLURM_TMPDIR exists.
USE_LOCAL_SCRATCH=0

if [ "$USE_LOCAL_SCRATCH" -eq 1 ]; then
  LOCAL_DATA="${SLURM_TMPDIR:-/tmp/slurm_${SLURM_JOB_ID}}/data"
  mkdir -p "$LOCAL_DATA"
  echo "Staging data to node-local scratch: $LOCAL_DATA"
  rsync -a "$PROJECT_ROOT/data/" "$LOCAL_DATA/"
  DATA_DIR="$LOCAL_DATA"
else
  DATA_DIR="$PROJECT_ROOT/data"
fi

# ── Environment ───────────────────────────────────────────────────
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
cd "$PROJECT_ROOT"

mkdir -p slurm_logs

export WANDB_PROJECT=terrains

TEST_FILE_ABS="$DATA_DIR/generated2/philly_test_1k_sources.npz"
TEST_FILE_REL="generated2/philly_test_1k_sources.npz"

RESOLUTIONS=(20 19 18 17 16 15 14 13 12 11 10 9 8 7 6 5 4 3 2 1)

for RES in "${RESOLUTIONS[@]}"; do
  RES_PADDED=$(printf "%02d" "$RES")
  TRAIN_FILE_ABS="$DATA_DIR/philly_res${RES_PADDED}_hybrid.npz"
  TRAIN_FILE_REL="philly_res${RES_PADDED}_hybrid.npz"
  DATASET_NAME="philadelphia/res${RES_PADDED}"

  echo "=================================================="
  echo "Resolution ${RES_PADDED}"
  echo "Train file: ${TRAIN_FILE_ABS}"
  echo "Test file:  ${TEST_FILE_ABS}"
  echo "=================================================="

  if [ ! -f "$TRAIN_FILE_ABS" ]; then
    echo "ERROR: Missing training data: $TRAIN_FILE_ABS" >&2
    exit 1
  fi

  echo "=================================================="
  echo "Training TAGConv for resolution ${RES}"
  echo "=================================================="

  python train_single_terrain_case.py \
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

  python train_single_terrain_case.py \
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

  python train_single_terrain_case.py \
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
