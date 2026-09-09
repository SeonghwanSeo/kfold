#!/usr/bin/env bash
#SBATCH --job-name=kfold-sequential
#SBATCH --partition=gpu-all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=08:00:00

# Set source, Python, input, config, checkpoint, CCD and output explicitly.
# No fixed checkpoint or benchmark source is embedded in this experimental job.
set -euo pipefail
: "${SEQUENTIAL_SOURCE:?export path to the sequential-complex-prior worktree}"
: "${KFOLD_PYTHON:?export Python executable}"
: "${INPUT:?export one assembly query or a query directory}"
: "${CONFIG:?export checkpoint-compatible model configuration}"
: "${WEIGHT:?export inference weight path}"
: "${CCD:?export CCD dictionary path}"
: "${OUTPUT:?export a fresh shared output directory}"
export PYTHONPATH="$SEQUENTIAL_SOURCE/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export NUMBA_CACHE_DIR="${SLURM_TMPDIR:-/tmp}/kfold-sequential-numba-${SLURM_JOB_ID}"
export PYTORCH_ALLOC_CONF=expandable_segments:True
cd "$SEQUENTIAL_SOURCE"
exec "$KFOLD_PYTHON" scripts/inference_sequential.py \
  --input "$INPUT" --config "$CONFIG" --weight "$WEIGHT" --ccd "$CCD" \
  --out-dir "$OUTPUT" --seed 1 2 3 4 5 --num-samples 5 \
  --num-recycles 10 --num-steps 100 "$@"
