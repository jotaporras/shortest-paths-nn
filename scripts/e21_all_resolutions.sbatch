#!/bin/bash
#SBATCH --job-name=e21-terrain
#SBATCH --output=/vast/projects/aribeiro/alelab/jporras/shortest-paths-nn/slurm-e21-terrain-%A_%a.out
#SBATCH --time=48:00:00
#SBATCH --partition=dgx-b200
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --array=0-19%4

# e21: TAGConv + SparseGT (k=5) full pipeline across smaller terrain fractions (res21-res40)
# Usage: sbatch scripts/e21_all_resolutions.sbatch
# Override defaults: TRIAL=2 WANDB_TAG=my_tag sbatch scripts/e21_all_resolutions.sbatch

set -euo pipefail

# -----------------------------------------------------------------------------
# Configuration (override via environment variables)
# -----------------------------------------------------------------------------
PROJECT_DIR=${PROJECT_DIR:-/vast/home/j/jporras/sourcecode/shortest-paths-nn}
OUTPUT_DIR=${OUTPUT_DIR:-/vast/projects/aribeiro/alelab/jporras/shortest-paths-nn}
TRIAL=${TRIAL:-2}
WANDB_TAG=${WANDB_TAG:-e21TG_smaller_fracions_deeper_k}
ENV_NAME=${ENV_NAME:-shortest-paths-nn}  # Set to conda env path if needed

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
TOTAL_TASKS=20

# Resolutions array (index 0 = res21, index 19 = res40)
RESOLUTIONS=(21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40)
RES=${RESOLUTIONS[$TASK_ID]}

# Batch size: 256 for all resolutions (res21-40 have lower node counts)
BATCH_SIZE=256

log "========================================"
log "=== e21 Experiment: res${RES} ==="
log "========================================"
log "  Resolution: res${RES} (task $((TASK_ID + 1))/${TOTAL_TASKS})"
log "  Batch size: ${BATCH_SIZE}"
log "  Trial: ${TRIAL}"
log "  Wandb tag: ${WANDB_TAG}"
log "  Output dir: ${OUTPUT_DIR}"

# =============================================================================
# TAGConv Training (Phase 1 + Phase 2)
# =============================================================================

log ""
log "========================================"
log "=== TAGConv Phase 1: GNN Training ==="
log "========================================"

TAGCONV_PHASE1_ARGS=(
    --train-data "generated2/res${RES}_phase1.npz"
    --test-data generated2/full_test-001.npz
    --epochs 500
    --device cuda
    --batch-size "$BATCH_SIZE"
    --dataset-name "norway/res${RES}"
    --config configs/tagconv-k3.yml
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
    --wandb-tag "$WANDB_TAG" stage1 TAGConv
)

log "Command: python train_single_terrain_case.py ${TAGCONV_PHASE1_ARGS[*]}"
python train_single_terrain_case.py "${TAGCONV_PHASE1_ARGS[@]}"

log "TAGConv Phase 1 completed for res${RES}"

log ""
log "========================================"
log "=== TAGConv Phase 2: MLP Finetuning ==="
log "========================================"

# TAGConv Phase 1 model path
TAGCONV_PHASE1_MODEL="${OUTPUT_DIR}/models/single_dataset/norway/res${RES}/TAGConv/no-vn/siamese/p-4/mse_loss/tagconv-k3/${TRIAL}/new"

if [ ! -f "${TAGCONV_PHASE1_MODEL}/final_model.pt" ]; then
    log "ERROR: TAGConv Phase 1 model not found at ${TAGCONV_PHASE1_MODEL}/final_model.pt"
    log "TAGConv Phase 1 may have failed - check logs above"
    exit 1
fi

log "  Loading TAGConv GNN from: ${TAGCONV_PHASE1_MODEL}"

TAGCONV_PHASE2_ARGS=(
    --finetune-from "$TAGCONV_PHASE1_MODEL"
    --train-data "generated2/res${RES}_phase2.npz"
    --epochs 250
    --device cuda
    --batch-size "$BATCH_SIZE"
    --dataset-name "res${RES}_phase2"
    --config configs/tagconv-k3.yml
    --siamese 1
    --vn 0
    --layer-type TAGConv
    --aggr 'sum+diff'
    --p 4
    --loss mse_loss
    --include-edge-attr 1
    --lr 0.001
    --trial "$TRIAL"
    --new
    --single-terrain-per-model
    --wandb-tag "$WANDB_TAG" stage2 TAGConv
)

log "Command: python train-de-coupled.py ${TAGCONV_PHASE2_ARGS[*]}"
python train-de-coupled.py "${TAGCONV_PHASE2_ARGS[@]}"

log "TAGConv Phase 2 completed for res${RES}"

# =============================================================================
# SparseGT Training (Phase 1 + Phase 2)
# =============================================================================

log ""
log "========================================"
log "=== SparseGT Phase 1: GNN Training ==="
log "========================================"

SPARSEGT_PHASE1_ARGS=(
    --train-data "generated2/res${RES}_phase1.npz"
    --test-data generated2/full_test-001.npz
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
    --wandb-tag "$WANDB_TAG" stage1 SparseGT
)

log "Command: python train_single_terrain_case.py ${SPARSEGT_PHASE1_ARGS[*]}"
python train_single_terrain_case.py "${SPARSEGT_PHASE1_ARGS[@]}"

log "SparseGT Phase 1 completed for res${RES}"

log ""
log "========================================"
log "=== SparseGT Phase 2: MLP Finetuning ==="
log "========================================"

# SparseGT Phase 1 model path
SPARSEGT_PHASE1_MODEL="${OUTPUT_DIR}/models/single_dataset/norway/res${RES}/SparseGT/no-vn/siamese/p-4/mse_loss/sparse-gt-rpearl/${TRIAL}/new"

if [ ! -f "${SPARSEGT_PHASE1_MODEL}/final_model.pt" ]; then
    log "ERROR: SparseGT Phase 1 model not found at ${SPARSEGT_PHASE1_MODEL}/final_model.pt"
    log "SparseGT Phase 1 may have failed - check logs above"
    exit 1
fi

log "  Loading SparseGT GNN from: ${SPARSEGT_PHASE1_MODEL}"

SPARSEGT_PHASE2_ARGS=(
    --finetune-from "$SPARSEGT_PHASE1_MODEL"
    --train-data "generated2/res${RES}_phase2.npz"
    --epochs 250
    --device cuda
    --batch-size "$BATCH_SIZE"
    --dataset-name "res${RES}_phase2"
    --config configs/sparse-gt-rpearl-k5.yml
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
    --wandb-tag "$WANDB_TAG" stage2 SparseGT
)

log "Command: python train-de-coupled.py ${SPARSEGT_PHASE2_ARGS[*]}"
python train-de-coupled.py "${SPARSEGT_PHASE2_ARGS[@]}"

log ""
log "========================================"
log "=== Completed res${RES} (TAGConv + SparseGT, Phase 1 + Phase 2) ==="
log "========================================"

