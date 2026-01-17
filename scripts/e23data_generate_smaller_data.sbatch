#!/bin/bash
#SBATCH --job-name=e23_smaller_datagen
#SBATCH --output=/vast/projects/aribeiro/alelab/jporras/shortest-paths-nn/slurm-dev-e23-tg-sgt-test-%A.out
#SBATCH --time=00:15:00
#SBATCH --partition=genoa-std-mem
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=8G



set -euo pipefail

# -----------------------------------------------------------------------------
# Configuration (override via environment variables)
# -----------------------------------------------------------------------------
PROJECT_DIR=${PROJECT_DIR:-/vast/home/j/jporras/sourcecode/shortest-paths-nn}
OUTPUT_DIR=${OUTPUT_DIR:-/vast/projects/aribeiro/alelab/jporras/shortest-paths-nn}
TRIAL=${TRIAL:-1}
WANDB_TAG=${WANDB_TAG:-dev_e23TG_tg_sgt_test}
TEST_DATA=${TEST_DATA:-generated2/full_test-004.npz}
ENV_NAME=${ENV_NAME:-shortest-paths-nn}

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

echo "Creating r=003"
python dataset/generate-test-dataset.py \
    --name norway \
    --raw-data data/norway-smallest.txt \
    --filename data/generated2/full_test-003.npz \
    --graph-resolution 3 \
    --dataset-size 100000 \
    --triangles  # if you want diagonal edges


echo "Creating r=004"
python dataset/generate-test-dataset.py \
    --name norway \
    --raw-data data/norway-smallest.txt \
    --filename data/generated2/full_test-004.npz \
    --graph-resolution 4 \
    --dataset-size 100000 \
    --triangles  # if you want diagonal edges


echo "Creating r=008"
python dataset/generate-test-dataset.py \
    --name norway \
    --raw-data data/norway-smallest.txt \
    --filename data/generated2/full_test-008.npz \
    --graph-resolution 8 \
    --dataset-size 100000 \
    --triangles  # if you want diagonal edges
