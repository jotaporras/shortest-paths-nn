#!/bin/bash
# Spectral analysis for e34 section 7 on plaza, GPU version: dense eigh of the five stride-12 graphs on GPU 0, one after the other.
# Launch: setsid nohup bash scripts/e34_spectral_gpu.sh > ~/runs/e34/analysis/spectral_gpu.log 2>&1 < /dev/null &
set -u
source ~/miniconda3/etc/profile.d/conda.sh
conda activate spnn-asilomar
cd ~/sourcecode/shortest-paths-nn
export E34_RUNS=~/runs/e34 E34_DATA=~/data/spnn/e31 E34_TEST=~/data/spnn/full_test-004.npz
STRIDE=${STRIDE:-12}
DTYPE=${DTYPE:-float32}
for s in grid uniform poisson-disk slope-adaptive quadtree; do
    echo "$(date) $s start"
    CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 python notebooks/e34_spectral.py eig --scheme "$s" --stride "$STRIDE" --device cuda --dtype "$DTYPE" \
        || { echo "$s failed"; exit 1; }
done
echo "$(date) ALL DONE"
