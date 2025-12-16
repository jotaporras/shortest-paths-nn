#!/bin/bash
#SBATCH --job-name=sparse-gt-terrain
#SBATCH --output=/home/jporras/sourcecode/shortest-paths-nn/logs/slurm-%A_%a.out
#SBATCH --time=8:00:00
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --array=0-19

# Sparse GT training across all terrain resolutions (res01-res20)
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
# Map array task ID (0-19) to resolution (01-20)
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
# Common Arguments
# -----------------------------------------------------------------------------
COMMON_ARGS=(
    --test-data generated/full_test-001.npz
    --epochs 500
    --device cuda
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

# -----------------------------------------------------------------------------
# Resolution-Specific Arguments
# -----------------------------------------------------------------------------
VARIANT_ARGS=(
    --train-data "generated/res${RES}_phase1.npz"
    --dataset-name "norway/res${RES}"
    --batch-size "$BATCH_SIZE"
)

# -----------------------------------------------------------------------------
# Launch Training
# -----------------------------------------------------------------------------
log "=== Sparse GT Training ==="
log "  Resolution: res${RES} (task $((TASK_ID + 1))/${TOTAL_TASKS})"
log "  Batch size: ${BATCH_SIZE}"
log "  Trial: ${TRIAL}"
log "  Wandb tag: ${WANDB_TAG}"

CMD=(
    python
    train_single_terrain_case.py
)
CMD+=("${COMMON_ARGS[@]}")
CMD+=("${VARIANT_ARGS[@]}")

log "Command: ${CMD[*]}"
"${CMD[@]}"

log "Completed res${RES}"

