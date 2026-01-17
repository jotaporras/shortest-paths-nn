#!/bin/bash
#SBATCH --job-name=e23_all_testsets
#SBATCH --output=/vast/projects/aribeiro/alelab/jporras/shortest-paths-nn/slurm-e23-all-testsets-%A.out
#SBATCH --time=24:00:00
#SBATCH --partition=genoa-std-mem
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G

# Generate all test datasets from res002 to res040
# Usage: sbatch scripts/e23data_generate_smaller_data.sbatch
# Override defaults: DATASET_SIZE=50000 sbatch scripts/e23data_generate_smaller_data.sbatch

set -euo pipefail

# -----------------------------------------------------------------------------
# Configuration (override via environment variables)
# -----------------------------------------------------------------------------
PROJECT_DIR=${PROJECT_DIR:-/vast/home/j/jporras/sourcecode/shortest-paths-nn}
OUTPUT_DIR=${OUTPUT_DIR:-/vast/projects/aribeiro/alelab/jporras/shortest-paths-nn}
ENV_NAME=${ENV_NAME:-shortest-paths-nn}
DATASET_SIZE=${DATASET_SIZE:-100000}

# -----------------------------------------------------------------------------
# Environment Setup
# -----------------------------------------------------------------------------
cd "$PROJECT_DIR"

log() {
    echo "[$(date --iso-8601=seconds)] $*"
}

# Conda activation
module load anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

# Data paths - use OUTPUT_DIR for large data files
RAW_DATA="${OUTPUT_DIR}/data/norway-smallest.txt"
OUTPUT_DATA_DIR="${OUTPUT_DIR}/data/generated2"

# Ensure output directory exists
mkdir -p "$OUTPUT_DATA_DIR"

log "========================================"
log "=== Generating Test Datasets res002-res040 ==="
log "========================================"
log "  Raw data: ${RAW_DATA}"
log "  Output dir: ${OUTPUT_DATA_DIR}"
log "  Dataset size: ${DATASET_SIZE}"

# Verify raw data exists
if [[ ! -f "$RAW_DATA" ]]; then
    log "ERROR: Raw data file not found: $RAW_DATA"
    exit 1
fi

# Generate test datasets for resolutions 2 through 40
for res in $(seq 2 40); do
    # Zero-pad resolution to 3 digits (002, 003, ..., 040)
    res_padded=$(printf "%03d" "$res")
    output_file="${OUTPUT_DATA_DIR}/full_test-${res_padded}.npz"
    
    # Skip if file already exists
    if [[ -f "$output_file" ]]; then
        log "Skipping res${res_padded} - file already exists: $output_file"
        continue
    fi
    
    log "Creating test dataset for res${res_padded} (resolution=${res})"
    
    python dataset/generate-test-dataset.py \
        --name norway \
        --raw-data "$RAW_DATA" \
        --filename "$output_file" \
        --graph-resolution "$res" \
        --dataset-size "$DATASET_SIZE" \
        --triangles
    
    log "Completed res${res_padded}"
done

log "========================================"
log "=== All test datasets generated ==="
log "========================================"
