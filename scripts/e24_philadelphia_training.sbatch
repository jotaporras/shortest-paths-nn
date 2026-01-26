#!/bin/bash
#SBATCH --job-name=e24-phil-train
#SBATCH --output=/vast/projects/aribeiro/alelab/jporras/shortest-paths-nn/logs/slurm-e24-phil-train-%A_%a.out
#SBATCH --time=48:00:00
#SBATCH --partition=dgx-b200
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --array=0-36%5

# e24: TAGConv Stage 1 training on Philadelphia terrain across res40..res04 (smallest to largest)
# Test set: generated_phil/full_test-004.npz
#
# Time requirements:
#   res40-res09: 48h is sufficient (default)
#   res08-res04: 72h required (use --time=72:00:00)
#
# Usage:
#   sbatch scripts/e24_philadelphia_training.sbatch
#
# Override defaults:
#   TRIAL=2 WANDB_TAG=my_tag sbatch scripts/e24_philadelphia_training.sbatch
#
# Run specific resolutions only:
#   sbatch --array=0-19%5 scripts/e24_philadelphia_training.sbatch  # res40-res21 (48h is enough)
#   sbatch --array=20-31%5 scripts/e24_philadelphia_training.sbatch # res20-res09 (48h is enough)
#   sbatch --array=32-36%2 --time=72:00:00 scripts/e24_philadelphia_training.sbatch # res08-res04 (need 72h)

set -euo pipefail

# -----------------------------------------------------------------------------
# Configuration (override via environment variables)
# -----------------------------------------------------------------------------
PROJECT_DIR=${PROJECT_DIR:-/vast/home/j/jporras/sourcecode/shortest-paths-nn}
OUTPUT_DIR=${OUTPUT_DIR:-/vast/projects/aribeiro/alelab/jporras/shortest-paths-nn}
TRIAL=${TRIAL:-1}
WANDB_TAG=${WANDB_TAG:-e24_philadelphia_transferability}
TEST_DATA=${TEST_DATA:-generated_phil/full_test-004.npz}
ENV_NAME=${ENV_NAME:-shortest-paths-nn}

# -----------------------------------------------------------------------------
# Environment Setup
# -----------------------------------------------------------------------------
cd "$PROJECT_DIR"

# Set output directory for models (used by training scripts via TERRAIN_OUTPUT_DIR)
export TERRAIN_OUTPUT_DIR="$OUTPUT_DIR"

log() {
    echo "[$(date --iso-8601=seconds)] $*"
}

# Conda activation
module load anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

# -----------------------------------------------------------------------------
# Resolution Mapping
# -----------------------------------------------------------------------------
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
TOTAL_TASKS=37

# Resolutions array: res40 to res04 (smallest to largest, i.e., most coarse to finer)
# Index 0 = res40, index 36 = res04
RESOLUTIONS=(40 39 38 37 36 35 34 33 32 31 30 29 28 27 26 25 24 23 22 21 20 19 18 17 16 15 14 13 12 11 10 09 08 07 06 05 04)
RES=${RESOLUTIONS[$TASK_ID]}

# Batch size: smaller for larger graphs (fine resolutions)
if [[ "$RES" =~ ^0[4-5]$ ]]; then
    BATCH_SIZE=512
else
    BATCH_SIZE=1024
fi

log "========================================"
log "=== e24 Philadelphia Training: res${RES} (Stage 1 Only) ==="
log "========================================"
log "  Resolution: res${RES} (task $((TASK_ID + 1))/${TOTAL_TASKS})"
log "  Batch size: ${BATCH_SIZE}"
log "  Test data: ${TEST_DATA}"
log "  Trial: ${TRIAL}"
log "  Wandb tag: ${WANDB_TAG}"
log "  Output dir: ${OUTPUT_DIR}"

# Verify training data exists
TRAIN_DATA="generated_phil/res${RES}_phase1.npz"
TRAIN_DATA_PATH="${OUTPUT_DIR}/data/${TRAIN_DATA}"
if [[ ! -f "$TRAIN_DATA_PATH" ]]; then
    log "ERROR: Training data not found: $TRAIN_DATA_PATH"
    log "  Run e24data_philadelphia_transferability.sbatch first to generate datasets"
    exit 1
fi

# =============================================================================
# TAGConv Training (Phase 1 Only)
# =============================================================================

log ""
log "========================================"
log "=== TAGConv Phase 1: GNN Training ==="
log "========================================"

TAGCONV_PHASE1_ARGS=(
    --train-data "$TRAIN_DATA"
    --test-data "$TEST_DATA"
    --epochs 500
    --device cuda
    --batch-size "$BATCH_SIZE"
    --dataset-name "phil/res${RES}"
    --config configs/tagconv-k5.yml
    --siamese 1
    --vn 0
    --layer-type TAGConv
    --aggr 'sum+diff'
    --p 4
    --loss mse_loss
    --finetune 0
    --include-edge-attr 1
    --lr 0.0001
    --trial "$TRIAL"
    --new
    --wandb-tag "$WANDB_TAG" stage1 TAGConv "train-res${RES}" "test-004" philadelphia
)

log "Command: python train_single_terrain_case.py ${TAGCONV_PHASE1_ARGS[*]}"
python train_single_terrain_case.py "${TAGCONV_PHASE1_ARGS[@]}"

log "TAGConv Phase 1 completed for res${RES}"

log ""
log "========================================"
log "=== Completed res${RES} (TAGConv, Stage 1 only) ==="
log "========================================"
