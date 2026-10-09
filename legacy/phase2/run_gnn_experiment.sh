#!/bin/bash
# Run full GNN experiment across multiple resolutions
# Usage: CUDA_VISIBLE_DEVICES=0 ./run_gnn_experiment.sh --split even [trial]
# Usage: CUDA_VISIBLE_DEVICES=1 ./run_gnn_experiment.sh --split odd [trial]

set -e

if [ "$1" != "--split" ] || [ -z "$2" ]; then
    echo "Usage: CUDA_VISIBLE_DEVICES=X ./run_gnn_experiment.sh --split <even|odd> [trial]"
    echo ""
    echo "Examples:"
    echo "  CUDA_VISIBLE_DEVICES=0 ./run_gnn_experiment.sh --split even 1"
    echo "  CUDA_VISIBLE_DEVICES=1 ./run_gnn_experiment.sh --split odd 1"
    exit 1
fi

SPLIT=$2
TRIAL=${3:-1}

if [ "$SPLIT" = "even" ]; then
    RESOLUTIONS=(02 04 06 08 10 12 14 16 18 20)
elif [ "$SPLIT" = "odd" ]; then
    RESOLUTIONS=(01 03 05 07 09 11 13 15 17 19)
else
    echo "ERROR: Split must be 'even' or 'odd', got: $SPLIT"
    exit 1
fi

echo "Starting GNN experiment"
echo "  Split: $SPLIT"
echo "  Trial: $TRIAL"
echo "  Resolutions: ${RESOLUTIONS[*]} (will run in reverse order)"
echo ""

# Run in reverse order (highest resolution first)
for ((i=${#RESOLUTIONS[@]}-1; i>=0; i--)); do
    RES=${RESOLUTIONS[$i]}
    echo ""
    echo "========================================"
    echo "  Resolution: res${RES}"
    echo "========================================"
    
    echo "--- Phase 1 ---"
    ./run_phase1_gnn.sh $RES $TRIAL
    
    echo "--- Phase 2 ---"
    ./run_phase2_gnn.sh $RES $TRIAL
    
    echo "Completed res${RES}"
done

echo ""
echo "========================================"
echo "  Experiment complete!"
echo "========================================"

