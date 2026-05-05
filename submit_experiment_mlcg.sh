#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
PREP_SCRIPT="${SCRIPT_DIR}/prepare_mlcg_runs.py"
WORKER_SCRIPT="${SCRIPT_DIR}/run_experiment_mlcg_array.sh"
BASE="/p/project1/cameo/schmidt36"

if [[ -f "${BASE}/load_modules.sh" ]]; then
  # shellcheck disable=SC1090
  source "${BASE}/load_modules.sh"
fi
if [[ -f "${BASE}/test_env_newsetup/bin/activate" ]]; then
  # shellcheck disable=SC1090
  source "${BASE}/test_env_newsetup/bin/activate"
fi

PYTHON_BIN="${PYTHON_BIN:-$(command -v python3 || command -v python)}"
if [[ -z "${PYTHON_BIN}" ]]; then
  echo "ERROR: no python interpreter available" >&2
  exit 2
fi

usage() {
  cat <<USAGE
Usage:
  bash submit_experiment_mlcg.sh \
    --trained-model <model.mlir> \
    --classical-output <outputs/<experiment_name>> \
    --input-mlcg <input_template.in> \
    --max-concurrent <N> \
    [--dry-run]
USAGE
}

TRAINED_MODEL=""
CLASSICAL_OUTPUT=""
INPUT_MLCG=""
MAX_CONCURRENT=""
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --trained-model) TRAINED_MODEL="${2:-}"; shift 2 ;;
    --classical-output) CLASSICAL_OUTPUT="${2:-}"; shift 2 ;;
    --input-mlcg) INPUT_MLCG="${2:-}"; shift 2 ;;
    --max-concurrent) MAX_CONCURRENT="${2:-}"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 1 ;;
  esac
done

if [[ -z "${TRAINED_MODEL}" || -z "${CLASSICAL_OUTPUT}" || -z "${INPUT_MLCG}" || -z "${MAX_CONCURRENT}" ]]; then
  usage
  exit 1
fi

if [[ ! "${MAX_CONCURRENT}" =~ ^[0-9]+$ ]] || [[ "${MAX_CONCURRENT}" -lt 1 ]]; then
  echo "ERROR: --max-concurrent must be >= 1" >&2
  exit 2
fi

if [[ ! -f "${PREP_SCRIPT}" ]]; then
  echo "ERROR: missing prep script: ${PREP_SCRIPT}" >&2
  exit 2
fi
if [[ ! -f "${WORKER_SCRIPT}" ]]; then
  echo "ERROR: missing worker script: ${WORKER_SCRIPT}" >&2
  exit 2
fi
if [[ ! -f "${INPUT_MLCG}" ]]; then
  echo "ERROR: missing MLCG input template: ${INPUT_MLCG}" >&2
  exit 2
fi
if [[ ! -d "${CLASSICAL_OUTPUT}" ]]; then
  echo "ERROR: classical output directory not found: ${CLASSICAL_OUTPUT}" >&2
  exit 2
fi

PREP_ARGS=(
  --classical-output "${CLASSICAL_OUTPUT}"
  --trained-model "${TRAINED_MODEL}"
  --input-template "${INPUT_MLCG}"
)
if [[ ${DRY_RUN} -eq 1 ]]; then
  PREP_ARGS+=(--dry-run)
fi

"${PYTHON_BIN}" "${PREP_SCRIPT}" "${PREP_ARGS[@]}"

MLCG_READY_MANIFEST="${CLASSICAL_OUTPUT}/mlcg/mlcg_ready_manifest.tsv"
SUMMARY_JSON="${CLASSICAL_OUTPUT}/mlcg/mlcg_submit_summary.json"

N_MLCG=0
[[ -f "${MLCG_READY_MANIFEST}" ]] && N_MLCG=$(awk 'END{print NR}' "${MLCG_READY_MANIFEST}")

if [[ ${DRY_RUN} -eq 1 ]]; then
  cat <<EOF
[dry-run] mlcg preparation complete
python: ${PYTHON_BIN}
classical_output: ${CLASSICAL_OUTPUT}
mlcg_runs_ready: ${N_MLCG}
max_concurrent: ${MAX_CONCURRENT}
EOF
  exit 0
fi

MLCG_JOB_ID=""
if [[ "${N_MLCG}" -gt 0 ]]; then
  MLCG_JOB_LINE=$(sbatch \
    --account=cameo \
    --nodes=1 \
    --ntasks-per-node=1 \
    --partition=booster \
    --gres=gpu:1 \
    --time=02:00:00 \
    --job-name=exp_mlcg \
    --output="${CLASSICAL_OUTPUT}/mlcg/slurm-mlcg-%A_%a.out" \
    --array="0-$((N_MLCG - 1))%${MAX_CONCURRENT}" \
    --export=ALL,EXPERIMENT_MANIFEST="${MLCG_READY_MANIFEST}" \
    "${WORKER_SCRIPT}")
  MLCG_JOB_ID=$(awk '{print $4}' <<< "${MLCG_JOB_LINE}")
fi

cat > "${SUMMARY_JSON}" <<EOF
{
  "python": "${PYTHON_BIN}",
  "classical_output": "${CLASSICAL_OUTPUT}",
  "mlcg_ready_manifest": "${MLCG_READY_MANIFEST}",
  "mlcg_runs_ready": ${N_MLCG},
  "max_concurrent": ${MAX_CONCURRENT},
  "job_id": "${MLCG_JOB_ID}"
}
EOF

echo "Submitted MLCG experiment"
echo "  classical_output: ${CLASSICAL_OUTPUT}"
echo "  mlcg runs:        ${N_MLCG} (job ${MLCG_JOB_ID:-n/a})"
echo "  summary:          ${SUMMARY_JSON}"
