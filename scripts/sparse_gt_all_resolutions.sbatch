#!/bin/bash
#SBATCH --job-name=sparse-gt-terrain
#SBATCH --output=/home/jporras/sourcecode/shortest-paths-nn/logs/slurm-%A_%a.out
#SBATCH --time=24:00:00
#SBATCH --partition=dgx-b200
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --array=0-19

# Sparse GT full pipeline (Phase 1 + Phase 2) across all terrain resolutions (res01-res20)
# Usage: sbatch scripts/sparse_gt_all_resolutions.sbatch
# Override defaults: TRIAL=2 WANDB_TAG=my_tag sbatch scripts/sparse_gt_all_resolutions.sbatch

set -euo pipefail

# -----------------------------------------------------------------------------
# Configuration (override via environment variables)
# -----------------------------------------------------------------------------
PROJECT_DIR=${PROJECT_DIR:-/home/jporras/sourcecode/shortest-paths-nn}
TRIAL=${TRIAL:-1}
WANDB_TAG=${WANDB_TAG:-e21_sparse_gt_terrain}
ENV_NAME=${ENV_NAME:-}  # Set to conda env path if needed

# -----------------------------------------------------------------------------
# Environment Setup
# -----------------------------------------------------------------------------
cd "$PROJECT_DIR"

log() {
    echo "[$(date --iso-8601=seconds)] $*"
}

# Conda activation (uncomment and modify if using conda)
# if [ -n "$ENV_NAME" ]; then
#     module load anaconda3
#     source "$(conda info --base)/etc/profile.d/conda.sh"
#     conda activate "$ENV_NAME"
# fi

# -----------------------------------------------------------------------------
# Resolution Mapping
# -----------------------------------------------------------------------------
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
TOTAL_TASKS=20

# Resolutions array (index 0 = res01, index 19 = res20)
RESOLUTIONS=(01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20)
RES=${RESOLUTIONS[$TASK_ID]}

# Batch size: 32 for res01-05 (high node count), 256 for res06-20 (lower node count)
RES_NUM=$((10#$RES))
if [ $RES_NUM -le 5 ]; then
    BATCH_SIZE=32
else
    BATCH_SIZE=256
fi

# -----------------------------------------------------------------------------
# Phase 1: GNN Training
# -----------------------------------------------------------------------------
log "========================================"
log "=== Phase 1: GNN Training ==="
log "========================================"
log "  Resolution: res${RES} (task $((TASK_ID + 1))/${TOTAL_TASKS})"
log "  Batch size: ${BATCH_SIZE}"
log "  Trial: ${TRIAL}"
log "  Wandb tag: ${WANDB_TAG}"

PHASE1_ARGS=(
    --train-data "generated/res${RES}_phase1.npz"
    --test-data generated/full_test-001.npz
    --epochs 500
    --device cuda
    --batch-size "$BATCH_SIZE"
    --dataset-name "norway/res${RES}"
    --config configs/sparse-gt-rpearl.yml
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
    --wandb-tag "$WANDB_TAG"
)

log "Command: python train_single_terrain_case.py ${PHASE1_ARGS[*]}"
python train_single_terrain_case.py "${PHASE1_ARGS[@]}"

log "Phase 1 completed for res${RES}"

# -----------------------------------------------------------------------------
# Phase 2: MLP Finetuning
# -----------------------------------------------------------------------------
log ""
log "========================================"
log "=== Phase 2: MLP Finetuning ==="
log "========================================"

# Phase 1 model path (must match format_log_dir output structure)
PHASE1_MODEL="models/single_dataset/norway/res${RES}/SparseGT/no-vn/siamese/p-4/mse_loss/sparse-gt-rpearl/${TRIAL}/new"

if [ ! -f "${PHASE1_MODEL}/final_model.pt" ]; then
    log "ERROR: Phase 1 model not found at ${PHASE1_MODEL}/final_model.pt"
    log "Phase 1 may have failed - check logs above"
    exit 1
fi

log "  Loading GNN from: ${PHASE1_MODEL}"

PHASE2_ARGS=(
    --finetune-from "$PHASE1_MODEL"
    --train-data "generated/res${RES}_phase2.npz"
    --epochs 1000
    --device cuda
    --batch-size "$BATCH_SIZE"
    --dataset-name "res${RES}_phase2"
    --config configs/sparse-gt-rpearl.yml
    --siamese 1
    --vn 0
    --layer-type SparseGT
    --aggr 'sum+diff'
    --p 4
    --loss mse_loss
    --include-edge-attr 1
    --lr 0.001
    --trial "$TRIAL"
    --new
    --single-terrain-per-model
    --wandb-tag "${WANDB_TAG}_phase2"
)

log "Command: python train-de-coupled.py ${PHASE2_ARGS[*]}"
python train-de-coupled.py "${PHASE2_ARGS[@]}"

log ""
log "========================================"
log "=== Completed res${RES} (Phase 1 + Phase 2) ==="
log "========================================"
