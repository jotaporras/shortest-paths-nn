#!/bin/bash
# Jan 15 2026: This OOMs for sparse GT on lb1. Test set: generated2/full_test-003.npz
# e23: TAGConv + SparseGT Stage 1 ONLY across res40..res2 (smallest to largest)
# Test set: generated2/full_test-002.npz
# Usage: ./scripts/e23_terrain_graph_2.sh
# Override defaults: TRIAL=2 CUDA_VISIBLE_DEVICES=0 ./scripts/e23_terrain_graph_2.sh

set -euo pipefail

# -----------------------------------------------------------------------------
# Configuration (override via environment variables)
# -----------------------------------------------------------------------------
PROJECT_DIR=${PROJECT_DIR:-$(pwd)}
OUTPUT_DIR=${OUTPUT_DIR:-.}
TRIAL=${TRIAL:-1}
WANDB_TAG=${WANDB_TAG:-e23TG_neurogf_terrain_graph_2}
TEST_DATA=${TEST_DATA:-generated2/full_test-004.npz}

# -----------------------------------------------------------------------------
# Environment Setup
# -----------------------------------------------------------------------------
cd "$PROJECT_DIR"

# Set output directory for models (used by training scripts via TERRAIN_OUTPUT_DIR)
export TERRAIN_OUTPUT_DIR="$OUTPUT_DIR"

log() {
    echo "[$(date --iso-8601=seconds)] $*"
}

# -----------------------------------------------------------------------------
# Resolution List: res40 to res02 (smallest to largest, i.e., most coarse to finer)
# -----------------------------------------------------------------------------
# Running on left, redundancy in case right one doing biggies crashes.
RESOLUTIONS=(40 39 38 37 36 35 34 33 32 31 30 29 28 27 26 25 24 23 22 21 20 19 18 17 16 15 14 13 12 11 10 09 08 07 06 05 04 03 02)
# Running on right
#RESOLUTIONS=(02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20)
TOTAL_RESOLUTIONS=${#RESOLUTIONS[@]}

log "========================================"
log "=== e23 Experiment: Stage 1 Only ==="
log "========================================"
log "  Resolutions: res40 -> res02 (${TOTAL_RESOLUTIONS} total)"
log "  Test data: ${TEST_DATA}"
log "  Trial: ${TRIAL}"
log "  Wandb tag: ${WANDB_TAG}"
log "  Output dir: ${OUTPUT_DIR}"
log "========================================"

# =============================================================================
# Loop through all resolutions
# =============================================================================
for i in "${!RESOLUTIONS[@]}"; do
    RES=${RESOLUTIONS[$i]}
    
    # Batch size: 32 for res01-05 (larger graphs), 256 for res06+ (smaller graphs)
    if [[ "$RES" =~ ^0[1-5]$ ]]; then
        BATCH_SIZE=32
    else
        BATCH_SIZE=256
    fi
    
    log ""
    log "========================================"
    log "=== Processing res${RES} ($((i + 1))/${TOTAL_RESOLUTIONS}) ==="
    log "========================================"
    log "  Batch size: ${BATCH_SIZE}"

    # -------------------------------------------------------------------------
    # TAGConv Phase 1: GNN Training
    # -------------------------------------------------------------------------
    log ""
    log "--- TAGConv Phase 1: GNN Training (train=res${RES}, test=002) ---"

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
        --wandb-tag "$WANDB_TAG" stage1 TAGConv "train-res${RES}" "test-002"
    )

    log "Command: python train_single_terrain_case.py ${TAGCONV_PHASE1_ARGS[*]}"
    python train_single_terrain_case.py "${TAGCONV_PHASE1_ARGS[@]}"

    log "TAGConv Phase 1 completed for res${RES}"

    # -------------------------------------------------------------------------
    # SparseGT Phase 1: GNN Training
    # -------------------------------------------------------------------------
    log ""
    log "--- SparseGT Phase 1: GNN Training (train=res${RES}, test=002) ---"

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
        --wandb-tag "$WANDB_TAG" stage1 SparseGT "train-res${RES}" "test-002"
    )

    log "Command: python train_single_terrain_case.py ${SPARSEGT_PHASE1_ARGS[*]}"
    python train_single_terrain_case.py "${SPARSEGT_PHASE1_ARGS[@]}"

    log "SparseGT Phase 1 completed for res${RES}"

    log ""
    log "=== Completed res${RES} (TAGConv + SparseGT, Stage 1 only) ==="
done

log ""
log "========================================"
log "=== e23 Experiment Complete ==="
log "========================================"
log "  All ${TOTAL_RESOLUTIONS} resolutions processed (Stage 1 only)"
log "  Models saved to: ${OUTPUT_DIR}/models/single_dataset/norway/"
log "========================================"
