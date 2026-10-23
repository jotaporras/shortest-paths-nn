#!/bin/bash
#SBATCH --job-name=e35-lappe
#SBATCH --output=/vast/projects/aribeiro/alelab/jporras/shortest-paths-nn/e35/slurm-%x-%A_%a.out
#SBATCH --time=16:00:00
#SBATCH --partition=dgx-b200
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=28
#SBATCH --mem=224G
#SBATCH --array=0-29

# e35: the e34 transferability suite for SparseGT k=2 with Laplacian eigenvector encodings (sgt2lap),
# 5 schemes x strides 32 24 16 12 8 4, e32 recipe, 200 epochs, tested on the stride-4 grid full_test-004.
# Same command as scripts/e34_transfer_suite.sh (model sgt2lap). LAPPE_K comes from the e35 stride-12 tuning
# on plaza (docs/2026-10-e34-transfer-suite.md, section 7).
#
#   LAPPE_K=<k> sbatch scripts/e35_lappe_suite.sbatch          # all 30 runs
#   LAPPE_K=<k> sbatch --array=0-24 scripts/e35_lappe_suite.sbatch   # strides 32-8 only
#
# Task order: stride-major, small graphs first (tasks 0-4 = stride 32, ..., 25-29 = stride 4).
# Eigenvectors are read from $SPNN_LAPPE_CACHE (rsynced from plaza, where the timings were recorded);
# a missing entry is computed on the node and its cost goes to the wandb config.
set -uo pipefail

REPO=${REPO:-$HOME/sourcecode/shortest-paths-nn-asilomar}
ENV_NAME=${ENV_NAME:-/vast/projects/aribeiro/alelab/jporras/envs/shortest-paths-nn}
BASE=/vast/projects/aribeiro/alelab/jporras/shortest-paths-nn
DATA=${DATA:-$BASE/data/e31}
TEST=${TEST:-$BASE/data/full_test-004.npz}
OUT=${OUT:-$BASE/e35}
LAPPE_K=${LAPPE_K:?set LAPPE_K to the tuned number of eigenvectors}
EPOCHS=${EPOCHS:-200}
TAG=${TAG:-e35}
export SPNN_LAPPE_CACHE=${SPNN_LAPPE_CACHE:-$BASE/data/lappe}
export WANDB_PROJECT=${WANDB_PROJECT:-terrains}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TQDM_DISABLE=1 PYTHONUNBUFFERED=1

module load anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"
cd "$REPO"
mkdir -p "$OUT"

STRIDES=(32 24 16 12 8 4)
SCHEMES=(grid uniform poisson-disk slope-adaptive quadtree)
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
stride=${STRIDES[$((TASK_ID / 5))]}
scheme=${SCHEMES[$((TASK_ID % 5))]}
data=$(printf "norway_%s_s%02d" "$scheme" "$stride")
run="${data}_sgt2lap"
if [ -f "$OUT/$run.json" ]; then echo "$run already done"; exit 0; fi
echo "$(date +%F_%T) $(hostname) task $TASK_ID start $run (lappe_k $LAPPE_K)"
start=$SECONDS
TERRAIN_OUTPUT_DIR="$OUT/$run" python tests/smoke/run_with_metrics.py "$OUT/$run.json.tmp" \
  --train-data "$DATA/$data.npz" --test-data "$TEST" \
  --layer-type SparseGT --rpearl-embedding-mode lappe --lappe-k "$LAPPE_K" --config configs/e35-sparse-gt-k2-lappe.yml \
  --epochs "$EPOCHS" --batch-size 8192 --lr 0.0035 \
  --siamese 1 --p 1 --aggr sum+diff --loss mse_loss --include-edge-attr 1 --new --trial 1 --device cuda \
  --val-pairs 25000 --weight-decay 0.1 --warmup-steps 300 --lr-schedule constant --fast-loader \
  --resolution "$stride" --dataset-name "norway/e31/$data" \
  --run-name "$TAG-$run" --wandb-tag "$TAG" "$TAG-$scheme" "$TAG-s$stride" "$TAG-sgt2lap" "$TAG-k$LAPPE_K" \
  > "$OUT/$run.log" 2>&1
status=$?
if [ $status -eq 0 ] && [ -f "$OUT/$run.json.tmp" ]; then
  mv "$OUT/$run.json.tmp" "$OUT/$run.json"
  find "$OUT/$run" -name preds.csv -exec gzip -f {} \;
  echo "$(date +%F_%T) done  $run $((SECONDS - start))s"
else
  echo "$(date +%F_%T) FAILED $run (exit $status) $((SECONDS - start))s, see $OUT/$run.log"
  exit $status
fi
