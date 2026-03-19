#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -x "${PROJECT_DIR}/.venv/bin/python" ]]; then
  PYTHON_BIN="${PROJECT_DIR}/.venv/bin/python"
else
  PYTHON_BIN="python"
fi
CONFIG_PATH="${PROJECT_DIR}/configs/train-ecsi-3.yaml"
WEIGHT="/mnt/parallel_storage/wykim_lab/icl_shwan/entire-train-log/20260305/ecsi-phase3/v260305/lpp0asxu/checkpoints/epoch0078_step00079000_wlddt0.6566.ckpt"
INPUT_ROOT="/mnt/parallel_storage/wykim_lab/share/kfold_benchmarks"
OUTPUT_ROOT="${PROJECT_DIR}/tmp/benchmark_outputs"
NAME=""
CCD_PATH=""
SEEDS=("0")
NUM_GPUS=8
NUM_WORKERS=4
NUM_RECYCLES=10
NUM_STEPS=200
NUM_SAMPLES=5
OVERWRITE=0

# BENCHMARKS=(foldbench casp16 pronaset deepternary)
# BENCHMARKS=(foldbench casp16)
BENCHMARKS=(pronaset deepternary)
APOS=(af2)

PHASE_SAMPLER_OVERRIDES=(
  "model.structure_module.sampling_schedule_type=phase_power"
  "model.structure_module.sampling_schedule_start_power=2.0"
  "model.structure_module.sampling_schedule_end_power=3.0"
  "model.structure_module.sampling_schedule_midpoint=0.4"
  "model.structure_module.sampling_schedule_endpoint_trim=0.02"
  "model.structure_module.sampling_schedule_global_u_power=2.0"
  "model.structure_module.sampling_schedule_middle_power=1.0"
  "model.structure_module.sampling_schedule_churn_power=1.2"
  "model.structure_module.sampling_schedule_ode_power=2.6"
  "model.structure_module.ode_time_duration=0.4"
  "model.structure_module.sampling_schedule_ode_fraction=0.25"
  "model.structure_module.use_forward_pinned_churn=true"
  "model.structure_module.churn_until_time=0.7"
  "model.structure_module.sampling_schedule_churn_fraction=0.35"
  "model.structure_module.churn_factor=2.5"
)

usage() {
  cat <<'EOF'
Usage: scripts/run_benchmark.sh --name RUN_NAME [options]

Options:
  --name NAME            Output run name. Required.
  --ccd PATH             Optional CCD path. If omitted, use train.data.ccd_path
                         from the resolved config, like validate.py.
  --config PATH          Config path. Default: configs/train-ecsi-3.yaml
  --checkpoint PATH      Checkpoint path.
  --input-root PATH      Benchmark input root. Default:
                         /mnt/parallel_storage/wykim_lab/share/kfold_benchmarks
  --output-root PATH     Writable benchmark output root. Default:
                         <repo>/tmp/benchmark_outputs
  --num-gpus N           Number of GPUs to use. Default: 8
  --num-workers N        DataLoader workers. Default: 4
  --num-recycles N       Number of recycles. Default: 10
  --num-steps N          Number of diffusion steps. Default: 200
  --num-samples N        Samples per query. Default: 5
  --seed S               Repeatable. Default: 0
  --overwrite            Overwrite existing benchmark outputs.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --name)
      NAME="$2"
      shift 2
      ;;
    --ccd)
      CCD_PATH="$2"
      shift 2
      ;;
    --config)
      CONFIG_PATH="$2"
      shift 2
      ;;
    --checkpoint)
      WEIGHT="$2"
      shift 2
      ;;
    --input-root)
      INPUT_ROOT="$2"
      shift 2
      ;;
    --output-root)
      OUTPUT_ROOT="$2"
      shift 2
      ;;
    --num-gpus)
      NUM_GPUS="$2"
      shift 2
      ;;
    --num-workers)
      NUM_WORKERS="$2"
      shift 2
      ;;
    --num-recycles)
      NUM_RECYCLES="$2"
      shift 2
      ;;
    --num-steps)
      NUM_STEPS="$2"
      shift 2
      ;;
    --num-samples)
      NUM_SAMPLES="$2"
      shift 2
      ;;
    --seed)
      SEEDS+=("$2")
      shift 2
      ;;
    --overwrite)
      OVERWRITE=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -z "${NAME}" ]]; then
  echo "--name is required." >&2
  usage >&2
  exit 1
fi

if [[ "${#SEEDS[@]}" -gt 0 && "${SEEDS[0]}" == "0" && "${#SEEDS[@]}" -gt 1 ]]; then
  SEEDS=("${SEEDS[@]:1}")
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

for apo in "${APOS[@]}"; do
  for benchmark in "${BENCHMARKS[@]}"; do
    INPUT_QUERY="${INPUT_ROOT}/${benchmark}/kfold_inputs/${apo}/queries"
    OUTPUT_DIR="${OUTPUT_ROOT}/${benchmark}/${NAME}/${apo}"

    CMD=(
      "${PYTHON_BIN}"
      "${PROJECT_DIR}/scripts/inference_multigpu.py"
      --config "${CONFIG_PATH}"
      --checkpoint "${WEIGHT}"
      --num_gpus "${NUM_GPUS}"
      --num_workers "${NUM_WORKERS}"
      --num_recycles "${NUM_RECYCLES}"
      --num_steps "${NUM_STEPS}"
      --num_samples "${NUM_SAMPLES}"
      -i "${INPUT_QUERY}"
      -o "${OUTPUT_DIR}"
      --seed "${SEEDS[@]}"
    )

    if [[ -n "${CCD_PATH}" ]]; then
      CMD+=(--ccd "${CCD_PATH}")
    fi
    if [[ "${OVERWRITE}" -eq 1 ]]; then
      CMD+=(--overwrite)
    fi
    for override in "${PHASE_SAMPLER_OVERRIDES[@]}"; do
      CMD+=(--override "${override}")
    done

    echo "[run_benchmark] benchmark=${benchmark} apo=${apo} gpus=${NUM_GPUS}"
    printf '  %q' "${CMD[@]}"
    printf '\n'
    "${CMD[@]}"
  done
done
