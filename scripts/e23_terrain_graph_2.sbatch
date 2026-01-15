#!/bin/bash
#SBATCH --job-name=e23-terrain
#SBATCH --output=/vast/projects/aribeiro/alelab/jporras/shortest-paths-nn/slurm-e23-terrain-%A_%a.out
#SBATCH --time=48:00:00
#SBATCH --partition=dgx-b200
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --array=0-38%5

# e23: TAGConv + SparseGT (k=5) Stage 1 ONLY across res40..res02 (smallest to largest)
# Test set: generated2/full_test-003.npz
# Usage: sbatch scripts/e23_terrain_graph_2.sbatch
# Override defaults: TRIAL=2 WANDB_TAG=my_tag sbatch scripts/e23_terrain_graph_2.sbatch

set -euo pipefail

# -----------------------------------------------------------------------------
# Configuration (override via environment variables)
# -----------------------------------------------------------------------------
PROJECT_DIR=${PROJECT_DIR:-/vast/home/j/jporras/sourcecode/shortest-paths-nn}
OUTPUT_DIR=${OUTPUT_DIR:-/vast/projects/aribeiro/alelab/jporras/shortest-paths-nn}
TRIAL=${TRIAL:-1}
WANDB_TAG=${WANDB_TAG:-e23TG_neurogf_terrain_graph_2}
TEST_DATA=${TEST_DATA:-generated2/full_test-001.npz}
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
TOTAL_TASKS=39

# Resolutions array: res40 to res02 (smallest to largest, i.e., most coarse to finer)
# Index 0 = res40, index 38 = res02
RESOLUTIONS=(40 39 38 37 36 35 34 33 32 31 30 29 28 27 26 25 24 23 22 21 20 19 18 17 16 15 14 13 12 11 10 09 08 07 06 05 04 03 02)
RES=${RESOLUTIONS[$TASK_ID]}

# Batch size: 32 for res01-05 (larger graphs), 256 for res06+ (smaller graphs)
if [[ "$RES" =~ ^0[1-5]$ ]]; then
    BATCH_SIZE=32
else
    BATCH_SIZE=256
fi

log "========================================"
log "=== e23 Experiment: res${RES} (Stage 1 Only) ==="
log "========================================"
log "  Resolution: res${RES} (task $((TASK_ID + 1))/${TOTAL_TASKS})"
log "  Batch size: ${BATCH_SIZE}"
log "  Test data: ${TEST_DATA}"
log "  Trial: ${TRIAL}"
log "  Wandb tag: ${WANDB_TAG}"
log "  Output dir: ${OUTPUT_DIR}"

# =============================================================================
# TAGConv Training (Phase 1 Only)
# =============================================================================

log ""
log "========================================"
log "=== TAGConv Phase 1: GNN Training ==="
log "========================================"

TAGCONV_PHASE1_ARGS=(
    --train-data "generated2/res${RES}_phase1.npz"
    --test-data "$TEST_DATA"
    --epochs 500
    --device cuda
    --batch-size "$BATCH_SIZE"
    --dataset-name "norway/res${RES}"
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
    --wandb-tag "$WANDB_TAG" stage1 TAGConv "train-res${RES}" "test-003"
)

log "Command: python train_single_terrain_case.py ${TAGCONV_PHASE1_ARGS[*]}"
python train_single_terrain_case.py "${TAGCONV_PHASE1_ARGS[@]}"

log "TAGConv Phase 1 completed for res${RES}"

# =============================================================================
# SparseGT Training (Phase 1 Only)
# =============================================================================

log ""
log "========================================"
log "=== SparseGT Phase 1: GNN Training ==="
log "========================================"

SPARSEGT_PHASE1_ARGS=(
    --train-data "generated2/res${RES}_phase1.npz"
    --test-data "$TEST_DATA"
    --epochs 250
    --device cuda
    --batch-size "$BATCH_SIZE"
    --dataset-name "norway/res${RES}"
    --config configs/sparse-gt-rpearl-k5.yml
    --siamese 1
    --vn 0
    --layer-type SparseGT
    --aggr 'sum+diff'
    --p 4
    --loss mse_loss
    --finetune 0
    --include-edge-attr 1
    --lr 0.0001
    --trial "$TRIAL"
    --new
    --wandb-tag "$WANDB_TAG" stage1 SparseGT "train-res${RES}" "test-003"
)

log "Command: python train_single_terrain_case.py ${SPARSEGT_PHASE1_ARGS[*]}"
python train_single_terrain_case.py "${SPARSEGT_PHASE1_ARGS[@]}"

log "SparseGT Phase 1 completed for res${RES}"

log ""
log "========================================"
log "=== Completed res${RES} (TAGConv + SparseGT, Stage 1 only) ==="
log "========================================"
