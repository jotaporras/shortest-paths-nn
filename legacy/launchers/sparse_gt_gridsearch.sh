#!/bin/bash
# Sparse GT Grid Search Script
# Runs parallel experiments to find best SparseGT configuration
# Usage: ./scripts/sparse_gt_gridsearch.sh
# Override defaults: PARALLEL=8 CUDA_VISIBLE_DEVICES=0 ./scripts/sparse_gt_gridsearch.sh

set -euo pipefail

# -----------------------------------------------------------------------------
# Configuration (override via environment variables)
# -----------------------------------------------------------------------------
PROJECT_DIR=${PROJECT_DIR:-$(pwd)}
TRIAL=${TRIAL:-1}
WANDB_TAG=${WANDB_TAG:-dev_e23TG_sparse_gt_ft}
TRAIN_DATA=${TRAIN_DATA:-generated2/res40_phase1.npz}
TEST_DATA=${TEST_DATA:-generated2/full_test-008.npz}
EPOCHS=${EPOCHS:-2}
BATCH_SIZE=${BATCH_SIZE:-128}
LR=${LR:-0.0001}
PARALLEL=${PARALLEL:-2}  # Number of parallel runs

# -----------------------------------------------------------------------------
# Environment Setup
# -----------------------------------------------------------------------------
cd "$PROJECT_DIR"

log() {
    echo "[$(date --iso-8601=seconds)] $*"
}

# Create temporary config directory
CONFIG_DIR=$(mktemp -d)
trap "rm -rf $CONFIG_DIR" EXIT

log "========================================"
log "=== Sparse GT Grid Search ==="
log "========================================"
log "  Train data: ${TRAIN_DATA}"
log "  Test data: ${TEST_DATA}"
log "  Epochs: ${EPOCHS}"
log "  Batch size: ${BATCH_SIZE}"
log "  Learning rate: ${LR}"
log "  Parallel runs: ${PARALLEL}"
log "  Wandb tag: ${WANDB_TAG}"
log "  Config dir: ${CONFIG_DIR}"
log "========================================"

# -----------------------------------------------------------------------------
# Grid Search Parameters
# -----------------------------------------------------------------------------
HIDDEN_DIMS=(64 128)
NUM_LAYERS=(2 3 4)
NUM_HEADS=(4 8)
NUM_HOPS=(2 3 5)
RPEARL_SAMPLES=(30)
RPEARL_NUM_LAYERS=(3 5)
DROPOUTS=(0.1 0.3)
ATTN_DROPOUTS=(0.01)

# -----------------------------------------------------------------------------
# Generate all config combinations
# -----------------------------------------------------------------------------
declare -a CONFIGS=()

for hidden_dim in "${HIDDEN_DIMS[@]}"; do
    for num_layers in "${NUM_LAYERS[@]}"; do
        for num_heads in "${NUM_HEADS[@]}"; do
            # num_heads must divide hidden_dim
            if (( hidden_dim % num_heads != 0 )); then
                continue
            fi
            for num_hops in "${NUM_HOPS[@]}"; do
                for rpearl_samples in "${RPEARL_SAMPLES[@]}"; do
                    for rpearl_num_layers in "${RPEARL_NUM_LAYERS[@]}"; do
                        for dropout in "${DROPOUTS[@]}"; do
                            for attn_dropout in "${ATTN_DROPOUTS[@]}"; do
                                CONFIG_NAME="h${hidden_dim}_l${num_layers}_nh${num_heads}_k${num_hops}_rs${rpearl_samples}_rl${rpearl_num_layers}_d${dropout}_ad${attn_dropout}"
                                CONFIGS+=("$CONFIG_NAME|$hidden_dim|$num_layers|$num_heads|$num_hops|$rpearl_samples|$rpearl_num_layers|$dropout|$attn_dropout")
                            done
                        done
                    done
                done
            done
        done
    done
done

TOTAL_CONFIGS=${#CONFIGS[@]}
log "Total configurations to run: ${TOTAL_CONFIGS}"

# -----------------------------------------------------------------------------
# Function to generate YAML config file
# -----------------------------------------------------------------------------
generate_config() {
    local config_name=$1
    local hidden_dim=$2
    local num_layers=$3
    local num_heads=$4
    local num_hops=$5
    local rpearl_samples=$6
    local rpearl_num_layers=$7
    local dropout=$8
    local attn_dropout=$9
    local config_file="${CONFIG_DIR}/${config_name}.yml"
    
    cat > "$config_file" << EOF
# Auto-generated Sparse GT config: ${config_name}
sparse-gt-rpearl:
  gnn:
    constr:
      input: 3
      hidden: ${hidden_dim}
      output: ${hidden_dim}
      layers: ${num_layers}
    layer_norm: false
    dropout: true
    activation: lrelu
    sparse_gt:
      hidden_dim: ${hidden_dim}
      num_layers: ${num_layers}
      num_heads: ${num_heads}
      num_hops: ${num_hops}
      rpearl_samples: ${rpearl_samples}
      rpearl_num_layers: ${rpearl_num_layers}
      dropout: ${dropout}
      attn_dropout: ${attn_dropout}
  mlp:
    constr:
      input: ${hidden_dim}
      hidden: $((hidden_dim * 2))
      output: 1
      layers: 3
    layer_norm: false
    dropout: true
EOF
    echo "$config_file"
}

# -----------------------------------------------------------------------------
# Function to run a single experiment
# -----------------------------------------------------------------------------
run_experiment() {
    local config_str=$1
    local idx=$2
    
    IFS='|' read -r config_name hidden_dim num_layers num_heads num_hops rpearl_samples rpearl_num_layers dropout attn_dropout <<< "$config_str"
    
    local config_file
    config_file=$(generate_config "$config_name" "$hidden_dim" "$num_layers" "$num_heads" "$num_hops" "$rpearl_samples" "$rpearl_num_layers" "$dropout" "$attn_dropout")
    
    log "[$idx/${TOTAL_CONFIGS}] Starting: ${config_name}"
    
    python train_single_terrain_case.py \
        --train-data "$TRAIN_DATA" \
        --test-data "$TEST_DATA" \
        --epochs "$EPOCHS" \
        --device cuda \
        --batch-size "$BATCH_SIZE" \
        --dataset-name "norway/gridsearch/${config_name}" \
        --config "$config_file" \
        --siamese 1 \
        --vn 0 \
        --layer-type SparseGT \
        --aggr 'sum+diff' \
        --p 4 \
        --loss mse_loss \
        --finetune 0 \
        --include-edge-attr 1 \
        --lr "$LR" \
        --trial "$TRIAL" \
        --new \
        --wandb-tag "$WANDB_TAG" gridsearch SparseGT "$config_name" \
        2>&1 | while read -r line; do echo "[$config_name] $line"; done
    
    log "[$idx/${TOTAL_CONFIGS}] Completed: ${config_name}"
}

export -f run_experiment generate_config log
export PROJECT_DIR TRAIN_DATA TEST_DATA EPOCHS BATCH_SIZE LR TRIAL WANDB_TAG CONFIG_DIR TOTAL_CONFIGS

# -----------------------------------------------------------------------------
# Run experiments in parallel
# -----------------------------------------------------------------------------
log ""
log "Starting ${TOTAL_CONFIGS} experiments with ${PARALLEL} parallel workers..."
log ""

# Use GNU parallel if available, otherwise use xargs
if command -v parallel &> /dev/null; then
    printf '%s\n' "${CONFIGS[@]}" | \
        parallel --jobs "$PARALLEL" --keep-order --line-buffer \
        'run_experiment {} {#}'
else
    log "GNU parallel not found, using sequential execution with background jobs"
    
    running=0
    idx=0
    for config in "${CONFIGS[@]}"; do
        idx=$((idx + 1))
        run_experiment "$config" "$idx" &
        running=$((running + 1))
        
        # Wait if we've reached the parallel limit
        if (( running >= PARALLEL )); then
            wait -n
            running=$((running - 1))
        fi
    done
    
    # Wait for remaining jobs
    wait
fi

log ""
log "========================================"
log "=== Grid Search Complete ==="
log "========================================"
log "  Total configurations: ${TOTAL_CONFIGS}"
log "  Results logged to wandb with tag: ${WANDB_TAG}"
log "========================================"
