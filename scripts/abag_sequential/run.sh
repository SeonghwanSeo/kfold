#!/usr/bin/env bash
# Activate the recipient's kfold environment before invoking this script.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd -- "$SCRIPT_DIR/../.." && pwd)
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
: "${INPUT:?Set INPUT to the directory containing sequential YAML queries}"
[[ -d "$INPUT" ]] || { echo "Missing input directory: $INPUT" >&2; exit 2; }
OUT=${OUT:-$PWD/predictions_abag_sequential}
mkdir -p "$OUT"
OUT=$(cd -- "$OUT" && pwd)
read -r -a GPU_ARGS <<< "${GPU_IDS:-0}"
read -r -a MODES <<< "${MODES:-prior_and_trunk prior_and_trunk_multichain}"
for mode in "${MODES[@]}"; do
    case "$mode" in
        prior_only|prior_and_trunk|prior_and_trunk_multichain) ;;
        *) echo "Unknown conditioning mode: $mode" >&2; exit 2 ;;
    esac
done
COMMON=(-i "$INPUT" --seeds 1 2 3 4 5 6 7 8 9 10 --num-apos 5
        --num-samples 5 --num-recycles 10 --num-steps 100
        --gpu-ids "${GPU_ARGS[@]}")
case "${CPU_OFFLOAD:-0}" in
    0) ;;
    1) COMMON+=(--cpu-offload) ;;
    *) echo 'CPU_OFFLOAD must be 0 or 1' >&2; exit 2 ;;
esac
if [[ -n ${CACHE_DIR:-} ]]; then COMMON+=(--cache-dir "$CACHE_DIR"); fi
python -m kfold.cli.main "${COMMON[@]}" -o "$OUT/prepared_apo" --dry-run
# Prepare AtlasFold monomers once; repeated runs reuse completed preparation.
python -m kfold.cli.main "${COMMON[@]}" -o "$OUT/prepared_apo" --stage apo \
    2>&1 | tee -a "$OUT/prepare_apo.log"
for mode in "${MODES[@]}"; do
    python "$SCRIPT_DIR/reuse_prepared_apo.py" --input "$INPUT" \
        --source "$OUT/prepared_apo" --target "$OUT/$mode" \
        --seeds 1 2 3 4 5 6 7 8 9 10 \
        2>&1 | tee -a "$OUT/apo_reuse_$mode.log"
    python -m kfold.cli.main "${COMMON[@]}" -o "$OUT/$mode" \
        --stage complex --conditioning "$mode" --save-confidence \
        2>&1 | tee -a "$OUT/$mode.log"
done
