#!/bin/bash
#SBATCH --job-name=e24_phil_datagen
#SBATCH --output=/vast/projects/aribeiro/alelab/jporras/shortest-paths-nn/logs/slurm-e24-phil-%A_%a.out
#SBATCH --time=16:00:00
#SBATCH --partition=genoa-std-mem
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --array=0-39%4

# SLURM Array Job: Generate Philadelphia transferability datasets
# Each array task generates one resolution's Phase 1 dataset.
# Array index 0 = res40 (coarsest/fastest), index 39 = res01 (finest/slowest)
#
# Usage:
#   sbatch scripts/e24data_philadelphia_transferability.sbatch
#
# Run specific resolutions only:
#   sbatch --array=0-19%4 scripts/e24data_philadelphia_transferability.sbatch  # res40-res21 only
#   sbatch --array=35-39%2 scripts/e24data_philadelphia_transferability.sbatch # res05-res01 (slow)
#
# Outputs:
#   Training: "${OUTPUT_DATA_DIR}/res${RES}_phase1.npz"
#   Test:     "${OUTPUT_DATA_DIR}/full_test-${RES}.npz" (for res 3, 4, 8)
#
# Time estimates per task (500 sources):
#   res40-res21 (idx 0-19):   ~5-30 min each
#   res20-res11 (idx 20-29):  ~30-60 min each
#   res10-res06 (idx 30-34):  ~1-2 hours each
#   res05-res03 (idx 35-37):  ~2-4 hours each
#   res02       (idx 38):     ~4-6 hours
#   res01       (idx 39):     ~10-15 hours

set -euo pipefail

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
PROJECT_DIR=${PROJECT_DIR:-/vast/home/j/jporras/sourcecode/shortest-paths-nn}
OUTPUT_DIR=${OUTPUT_DIR:-/vast/projects/aribeiro/alelab/jporras/shortest-paths-nn}
ENV_NAME=${ENV_NAME:-shortest-paths-nn}

# Dataset generation parameters
NUM_SOURCES=${NUM_SOURCES:-500}
TRAIN_DATASET_SIZE=${TRAIN_DATASET_SIZE:-50000}
TEST_DATASET_SIZE=${TEST_DATASET_SIZE:-100000}

# -----------------------------------------------------------------------------
# Compute resolution from array index
# Index 0 = res40 (coarsest), Index 39 = res01 (finest)
# -----------------------------------------------------------------------------
ARRAY_IDX=${SLURM_ARRAY_TASK_ID:-0}
RESOLUTION=$((40 - ARRAY_IDX))
RES_PADDED=$(printf "%02d" "$RESOLUTION")

# -----------------------------------------------------------------------------
# Environment Setup
# -----------------------------------------------------------------------------
cd "$PROJECT_DIR"

log() {
    echo "[$(date --iso-8601=seconds)] [res${RES_PADDED}] $*"
}

# Conda activation
module load anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

# Data paths
RAW_DATA="${OUTPUT_DIR}/data/flat-phil-data-1.txt"
OUTPUT_DATA_DIR="${OUTPUT_DIR}/data/generated_phil"
LOG_DIR="${OUTPUT_DIR}/logs"

# Ensure directories exist
mkdir -p "$OUTPUT_DATA_DIR"
mkdir -p "$LOG_DIR"

log "========================================"
log "=== Philadelphia Dataset Generation ==="
log "========================================"
log "  Array index: ${ARRAY_IDX}"
log "  Resolution: ${RESOLUTION} (res${RES_PADDED})"
log "  Raw data: ${RAW_DATA}"
log "  Output dir: ${OUTPUT_DATA_DIR}"
log "  Num sources: ${NUM_SOURCES}"
log "  Dataset size: ${TRAIN_DATASET_SIZE}"

# Verify raw data exists
if [[ ! -f "$RAW_DATA" ]]; then
    log "ERROR: Raw data file not found: $RAW_DATA"
    exit 1
fi

# -----------------------------------------------------------------------------
# Generate Phase 1 Training Dataset
# -----------------------------------------------------------------------------
PHASE1_OUTPUT="${OUTPUT_DATA_DIR}/res${RES_PADDED}_phase1.npz"

if [[ -f "$PHASE1_OUTPUT" ]]; then
    log "Phase 1 dataset already exists, skipping: $PHASE1_OUTPUT"
else
    log "Generating Phase 1 dataset..."
    log "  Output: ${PHASE1_OUTPUT}"
    log "  Samples per source: $((TRAIN_DATASET_SIZE / NUM_SOURCES))"
    
    python dataset/dataset.py \
        --name phil \
        --raw-data "$RAW_DATA" \
        --filename "$PHASE1_OUTPUT" \
        --graph-resolution "$RESOLUTION" \
        --num-sources "$NUM_SOURCES" \
        --dataset-size "$TRAIN_DATASET_SIZE" \
        --sampling-technique distance-based \
        --triangles
    
    log "Phase 1 dataset complete"
fi

# -----------------------------------------------------------------------------
# Generate Test Dataset (only for resolutions 3, 4, 8)
# -----------------------------------------------------------------------------
if [[ "$RESOLUTION" == 3 ]] || [[ "$RESOLUTION" == 4 ]] || [[ "$RESOLUTION" == 8 ]]; then
    RES_PADDED_3=$(printf "%03d" "$RESOLUTION")
    TEST_OUTPUT="${OUTPUT_DATA_DIR}/full_test-${RES_PADDED_3}.npz"
    
    if [[ -f "$TEST_OUTPUT" ]]; then
        log "Test dataset already exists, skipping: $TEST_OUTPUT"
    else
        log "Generating test dataset..."
        log "  Output: ${TEST_OUTPUT}"
        log "  Dataset size: ${TEST_DATASET_SIZE}"
        
        python dataset/generate-test-dataset.py \
            --name phil \
            --raw-data "$RAW_DATA" \
            --filename "$TEST_OUTPUT" \
            --graph-resolution "$RESOLUTION" \
            --dataset-size "$TEST_DATASET_SIZE" \
            --triangles
        
        log "Test dataset complete"
    fi
fi

log "========================================"
log "=== Task complete for res${RES_PADDED} ==="
log "========================================"
