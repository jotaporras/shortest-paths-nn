#!/bin/bash
# Spectral analysis for e34 section 4, on plaza.
# Step 1: test-graph embeddings of the grid s12 models on GPU 0.
# Step 2: dense eigh of the five stride-12 graphs, five processes in parallel, 9 BLAS threads each (~25 GB each).
# Launch: setsid nohup bash scripts/e34_spectral.sh > ~/runs/e34/analysis/spectral.log 2>&1 < /dev/null &
set -u
source ~/miniconda3/etc/profile.d/conda.sh
conda activate spnn-asilomar
cd ~/sourcecode/shortest-paths-nn
export E34_RUNS=~/runs/e34 E34_DATA=~/data/spnn/e31 E34_TEST=~/data/spnn/full_test-004.npz
LOG=~/runs/e34/analysis
mkdir -p "$LOG"
STRIDE=${STRIDE:-12}

echo "$(date) embed start"
CUDA_VISIBLE_DEVICES=0 python notebooks/e34_spectral.py embed --scheme grid --stride "$STRIDE" > "$LOG/spectral_embed.log" 2>&1 \
    || { echo "embed failed, see $LOG/spectral_embed.log"; exit 1; }
echo "$(date) embed done"

for s in grid uniform poisson-disk slope-adaptive quadtree; do
    OMP_NUM_THREADS=9 OPENBLAS_NUM_THREADS=9 MKL_NUM_THREADS=9 \
        python notebooks/e34_spectral.py eig --scheme "$s" --stride "$STRIDE" > "$LOG/spectral_eig_$s.log" 2>&1 &
done
wait
echo "$(date) ALL DONE"
