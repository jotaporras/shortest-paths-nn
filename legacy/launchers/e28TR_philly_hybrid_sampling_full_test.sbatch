#!/bin/bash
#SBATCH --job-name=e28TR_philly_hybrid_sampling_full_test
#SBATCH --output=/vast/projects/aribeiro/alelab/jporras/shortest-paths-nn/slurm-%A_%a.out
#SBATCH --time=55:00:00
#SBATCH --partition=dgx-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=14
#SBATCH --mem=256G
#SBATCH --array=0-59

set -euo pipefail

# ── Configurable defaults ─────────────────────────────────────────
PROJECT=${PROJECT:-shortest-paths-nn}
ENTITY=${ENTITY:-alelab}
EXPERIMENT_TAG=${EXPERIMENT_TAG:-e28TR_philly_hybrid_sampling_full_test}
ENV_NAME=${ENV_NAME:-/vast/projects/aribeiro/alelab/jporras/envs/shortest-paths-nn}
OUTPUT_BASE=${OUTPUT_BASE:-/vast/projects/aribeiro/alelab/jporras/shortest-paths-nn}
PROJECT_ROOT=${PROJECT_ROOT:-$HOME/sourcecode/shortest-paths-nn}

mkdir -p "${OUTPUT_BASE}"

# ── Environment ───────────────────────────────────────────────────
module load anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

cd "$PROJECT_ROOT"
mkdir -p slurm_logs

export WANDB_PROJECT=terrains

# ── Data staging ──────────────────────────────────────────────────
USE_LOCAL_SCRATCH=${USE_LOCAL_SCRATCH:-0}

if [ "$USE_LOCAL_SCRATCH" -eq 1 ]; then
  LOCAL_DATA="${SLURM_TMPDIR:-/tmp/slurm_${SLURM_JOB_ID}}/data"
  mkdir -p "$LOCAL_DATA"
  echo "Staging data to node-local scratch: $LOCAL_DATA"
  rsync -a "$PROJECT_ROOT/data/" "$LOCAL_DATA/"
  DATA_DIR="$LOCAL_DATA"
else
  DATA_DIR="$PROJECT_ROOT/data"
fi

TEST_FILE_ABS="$DATA_DIR/generated2/philly_test_1k_sources.npz"
TEST_FILE_REL="generated2/philly_test_1k_sources.npz"

# ── Array index → (resolution, model) ─────────────────────────────
# array_id = res_idx * 3 + model_idx
#   model_idx: 0 = TAGConv, 1 = SparseGT-Random, 2 = SparseGT-Data
RESOLUTIONS=(20 19 18 17 16 15 14 13 12 11 10 9 8 7 6 5 4 3 2 1)
MODELS=(tagconv sparsegt_random sparsegt_data)

TASK_ID=${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is not set}
RES_IDX=$(( TASK_ID / ${#MODELS[@]} ))
MODEL_IDX=$(( TASK_ID % ${#MODELS[@]} ))

if [ "$RES_IDX" -ge "${#RESOLUTIONS[@]}" ]; then
  echo "ERROR: TASK_ID $TASK_ID out of range (max $(( ${#RESOLUTIONS[@]} * ${#MODELS[@]} - 1 )))" >&2
  exit 1
fi

RES=${RESOLUTIONS[$RES_IDX]}
MODEL=${MODELS[$MODEL_IDX]}
RES_PADDED=$(printf "%02d" "$RES")
TRAIN_FILE_ABS="$DATA_DIR/philly_res${RES_PADDED}_hybrid.npz"
TRAIN_FILE_REL="philly_res${RES_PADDED}_hybrid.npz"
DATASET_NAME="philadelphia/res${RES_PADDED}"

echo "=================================================="
echo "TASK_ID=${TASK_ID}  RES=${RES_PADDED}  MODEL=${MODEL}"
echo "Train file: ${TRAIN_FILE_ABS}"
echo "Test file:  ${TEST_FILE_ABS}"
echo "=================================================="

if [ ! -f "$TRAIN_FILE_ABS" ]; then
  echo "ERROR: Missing training data: $TRAIN_FILE_ABS" >&2
  exit 1
fi

case "$MODEL" in
  tagconv)
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
      --wandb-tag "$EXPERIMENT_TAG"
    ;;
  sparsegt_random)
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
      --wandb-tag "$EXPERIMENT_TAG"
    ;;
  sparsegt_data)
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
      --wandb-tag "$EXPERIMENT_TAG"
    ;;
  *)
    echo "ERROR: Unknown model $MODEL" >&2
    exit 1
    ;;
esac
