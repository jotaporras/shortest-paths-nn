#!/bin/bash
#SBATCH --job-name=dev-e23-tg-sgt-test
#SBATCH --output=/vast/projects/aribeiro/alelab/jporras/shortest-paths-nn/slurm-dev-e23-tg-sgt-test-%A.out
#SBATCH --time=1:00:00
#SBATCH --partition=dgx-b200
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

# dev-e23-tg-sgt-test: TAGConv + SparseGT (k=5) Stage 1 ONLY across res40..res02 (smallest to largest)
# Test set: generated2/full_test-003.npz
# Usage: sbatch scripts/dev_e23_tg_sgt_test.sbatch
# Override defaults: TRIAL=2 WANDB_TAG=my_tag sbatch scripts/dev_e23_tg_sgt_test.sbatch

set -euo pipefail

# -----------------------------------------------------------------------------
# Configuration (override via environment variables)
# -----------------------------------------------------------------------------
PROJECT_DIR=${PROJECT_DIR:-/vast/home/j/jporras/sourcecode/shortest-paths-nn}
OUTPUT_DIR=${OUTPUT_DIR:-/vast/projects/aribeiro/alelab/jporras/shortest-paths-nn}
TRIAL=${TRIAL:-1}
WANDB_TAG=${WANDB_TAG:-dev_e23TG_tg_sgt_test}
TEST_DATA=${TEST_DATA:-generated2/full_test-004.npz}
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
TOTAL_TASKS=1

RESOLUTIONS=(40)
RES=${RESOLUTIONS[$TASK_ID]}

BATCH_SIZE=256

log "========================================"
log "=== dev e23 Experiment: res${RES} (Stage 1 Only) ==="
log "========================================"
log "  Resolution: res${RES} (task $((TASK_ID + 1))/${TOTAL_TASKS})"
log "  Batch size: ${BATCH_SIZE}"
log "  Test data: ${TEST_DATA}"
log "  Trial: ${TRIAL}"
log "  Wandb tag: ${WANDB_TAG}"
log "  Output dir: ${OUTPUT_DIR}"

# =============================================================================
# SparseGT Training (Phase 1 Only)
# =============================================================================

log ""
log "========================================"
log "=== SparseGT Phase 1: GNN Training ==="
log "================   ========================"

SPARSEGT_PHASE1_ARGS=(
    --train-data "generated2/res${RES}_phase1.npz"
    --test-data "$TEST_DATA"
    --epochs 250
    --device cuda
    --batch-size "$BATCH_SIZE"
    --dataset-name "norway/dev_res${RES}"
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
    --wandb-tag "$WANDB_TAG" stage1 SparseGT "train-dev_res${RES}" "test-004"
)

log "Command: python train_single_terrain_case.py ${SPARSEGT_PHASE1_ARGS[*]}"
python train_single_terrain_case.py "${SPARSEGT_PHASE1_ARGS[@]}"

log "SparseGT Phase 1 completed for res${RES}"

log ""
log "========================================"
log "=== Completed res${RES} (TAGConv + SparseGT, Stage 1 only) ==="
log "========================================"
