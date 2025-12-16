#!/bin/bash
#SBATCH --job-name=sparse-gt-phase2
#SBATCH --output=/home/jporras/sourcecode/shortest-paths-nn/logs/slurm-phase2-%A_%a.out
#SBATCH --time=4:00:00
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --array=0-19

# Sparse GT Phase 2: MLP finetuning across all terrain resolutions (res01-res20)
# Requires Phase 1 models to be trained first!
# Usage: sbatch scripts/sparse_gt_phase2_all_resolutions.sbatch
# Override defaults: TRIAL=2 sbatch scripts/sparse_gt_phase2_all_resolutions.sbatch

set -euo pipefail

# -----------------------------------------------------------------------------
# Configuration (override via environment variables)
# -----------------------------------------------------------------------------
PROJECT_DIR=${PROJECT_DIR:-/home/jporras/sourcecode/shortest-paths-nn}
TRIAL=${TRIAL:-1}
WANDB_TAG=${WANDB_TAG:-e21_sparse_gt_terrain_phase2}
ENV_NAME=${ENV_NAME:-}

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

RESOLUTIONS=(01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20)
RES=${RESOLUTIONS[$TASK_ID]}

# Batch size: 32 for res01-05, 256 for res06-20
RES_NUM=$((10#$RES))
if [ $RES_NUM -le 5 ]; then
    BATCH_SIZE=32
else
    BATCH_SIZE=256
fi

# -----------------------------------------------------------------------------
# Phase 1 Model Path (must match format_log_dir output structure)
# -----------------------------------------------------------------------------
PHASE1_MODEL="models/single_dataset/norway/res${RES}/SparseGT/no-vn/siamese/p-4/mse_loss/sparse-gt-rpearl/${TRIAL}/new"

if [ ! -f "${PHASE1_MODEL}/final_model.pt" ]; then
    log "ERROR: Phase 1 model not found at ${PHASE1_MODEL}/final_model.pt"
    log "Run Phase 1 first: sbatch scripts/sparse_gt_all_resolutions.sbatch"
    exit 1
fi

# -----------------------------------------------------------------------------
# Common Arguments
# -----------------------------------------------------------------------------
COMMON_ARGS=(
    --epochs 1000
    --device cuda
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
    --wandb-tag "$WANDB_TAG"
)

# -----------------------------------------------------------------------------
# Resolution-Specific Arguments
# -----------------------------------------------------------------------------
VARIANT_ARGS=(
    --finetune-from "$PHASE1_MODEL"
    --train-data "generated/res${RES}_phase2.npz"
    --dataset-name "res${RES}_phase2"
    --batch-size "$BATCH_SIZE"
)

# -----------------------------------------------------------------------------
# Launch Training
# -----------------------------------------------------------------------------
log "=== Sparse GT Phase 2: MLP Finetuning ==="
log "  Resolution: res${RES} (task $((TASK_ID + 1))/${TOTAL_TASKS})"
log "  Batch size: ${BATCH_SIZE}"
log "  Trial: ${TRIAL}"
log "  Loading GNN from: ${PHASE1_MODEL}"

CMD=(
    python
    train-de-coupled.py
)
CMD+=("${COMMON_ARGS[@]}")
CMD+=("${VARIANT_ARGS[@]}")

log "Command: ${CMD[*]}"
"${CMD[@]}"

log "Completed Phase 2 for res${RES}"

