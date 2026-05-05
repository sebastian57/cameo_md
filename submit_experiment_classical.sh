#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
PREP_SCRIPT="${SCRIPT_DIR}/prepare_experiment_runs.py"
WORKER_SCRIPT="${SCRIPT_DIR}/run_experiment_classical_array.sh"
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
  bash submit_experiment_classical.sh \
    --in-scope-dir <dir> \
    --out-scope-dir <dir> \
    --output-name <name> \
    --input-free <free_input.in> \
    --input-fixed <fixed_input.in> \
    [--seed 123] [--samples-per-protein 5] [--max-concurrent 3] [--fixed-eq-steps 50000] [--skip-free] [--dry-run]
USAGE
}

IN_SCOPE_DIR=""
OUT_SCOPE_DIR=""
OUTPUT_NAME=""
INPUT_FREE=""
INPUT_FIXED=""
SEED=123
SAMPLES_PER_PROTEIN=5
MAX_CONCURRENT=3
FIXED_EQ_STEPS=50000
SKIP_FREE=0
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --in-scope-dir) IN_SCOPE_DIR="${2:-}"; shift 2 ;;
    --out-scope-dir) OUT_SCOPE_DIR="${2:-}"; shift 2 ;;
    --output-name) OUTPUT_NAME="${2:-}"; shift 2 ;;
    --input-free) INPUT_FREE="${2:-}"; shift 2 ;;
    --input-fixed) INPUT_FIXED="${2:-}"; shift 2 ;;
    --seed) SEED="${2:-}"; shift 2 ;;
    --samples-per-protein) SAMPLES_PER_PROTEIN="${2:-}"; shift 2 ;;
    --max-concurrent) MAX_CONCURRENT="${2:-}"; shift 2 ;;
    --fixed-eq-steps) FIXED_EQ_STEPS="${2:-}"; shift 2 ;;
    --skip-free) SKIP_FREE=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 1 ;;
  esac
done

if [[ -z "${IN_SCOPE_DIR}" || -z "${OUT_SCOPE_DIR}" || -z "${OUTPUT_NAME}" || -z "${INPUT_FREE}" || -z "${INPUT_FIXED}" ]]; then
  usage
  exit 1
fi

if [[ ! "${MAX_CONCURRENT}" =~ ^[0-9]+$ ]] || [[ "${MAX_CONCURRENT}" -lt 1 ]]; then
  echo "ERROR: --max-concurrent must be >= 1" >&2
  exit 2
fi

if [[ ! "${SAMPLES_PER_PROTEIN}" =~ ^[0-9]+$ ]] || [[ "${SAMPLES_PER_PROTEIN}" -lt 1 ]]; then
  echo "ERROR: --samples-per-protein must be >= 1" >&2
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
if [[ ! -f "${INPUT_FREE}" ]]; then
  echo "ERROR: missing free template: ${INPUT_FREE}" >&2
  exit 2
fi
if [[ ! -f "${INPUT_FIXED}" ]]; then
  echo "ERROR: missing fixed template: ${INPUT_FIXED}" >&2
  exit 2
fi

OUTPUT_ROOT="${SCRIPT_DIR}/outputs/${OUTPUT_NAME}"
mkdir -p "${OUTPUT_ROOT}"

PREP_ARGS=(
  --in-scope-dir "${IN_SCOPE_DIR}"
  --out-scope-dir "${OUT_SCOPE_DIR}"
  --output-root "${OUTPUT_ROOT}"
  --input-free "${INPUT_FREE}"
  --input-fixed "${INPUT_FIXED}"
  --seed "${SEED}"
  --samples-per-protein "${SAMPLES_PER_PROTEIN}"
  --fixed-eq-steps "${FIXED_EQ_STEPS}"
)
if [[ ${SKIP_FREE} -eq 1 ]]; then
  PREP_ARGS+=(--skip-free)
fi
if [[ ${DRY_RUN} -eq 1 ]]; then
  PREP_ARGS+=(--dry-run)
fi

"${PYTHON_BIN}" "${PREP_SCRIPT}" "${PREP_ARGS[@]}"

FREE_MANIFEST="${OUTPUT_ROOT}/classical_free_manifest.tsv"
FIXED_MANIFEST="${OUTPUT_ROOT}/classical_fixed_manifest.tsv"
SUMMARY_JSON="${OUTPUT_ROOT}/classical_submit_summary.json"

N_FREE=0
N_FIXED=0
[[ -f "${FREE_MANIFEST}" ]] && N_FREE=$(awk 'END{print NR}' "${FREE_MANIFEST}")
[[ -f "${FIXED_MANIFEST}" ]] && N_FIXED=$(awk 'END{print NR}' "${FIXED_MANIFEST}")

if [[ ${DRY_RUN} -eq 1 ]]; then
  cat <<EOF
[dry-run] classical preparation complete
python: ${PYTHON_BIN}
output_root: ${OUTPUT_ROOT}
free_runs_ready: ${N_FREE}
fixed_runs_ready: ${N_FIXED}
max_concurrent: ${MAX_CONCURRENT}
EOF
  exit 0
fi

FREE_JOB_ID=""
FIXED_JOB_ID=""

if [[ "${N_FREE}" -gt 0 ]]; then
  FREE_JOB_LINE=$(sbatch \
    --account=cameo \
    --nodes=1 \
    --ntasks-per-node=1 \
    --partition=booster \
    --gres=gpu:1 \
    --time=10:00:00 \
    --job-name=exp_classical_free \
    --output="${OUTPUT_ROOT}/slurm-classical-free-%A_%a.out" \
    --array="0-$((N_FREE - 1))%${MAX_CONCURRENT}" \
    --export=ALL,EXPERIMENT_MANIFEST="${FREE_MANIFEST}" \
    "${WORKER_SCRIPT}")
  FREE_JOB_ID=$(awk '{print $4}' <<< "${FREE_JOB_LINE}")
fi

if [[ "${N_FIXED}" -gt 0 ]]; then
  FIXED_JOB_LINE=$(sbatch \
    --account=cameo \
    --nodes=1 \
    --ntasks-per-node=1 \
    --partition=booster \
    --gres=gpu:1 \
    --time=10:00:00 \
    --job-name=exp_classical_fixed \
    --output="${OUTPUT_ROOT}/slurm-classical-fixed-%A_%a.out" \
    --array="0-$((N_FIXED - 1))%${MAX_CONCURRENT}" \
    --export=ALL,EXPERIMENT_MANIFEST="${FIXED_MANIFEST}" \
    "${WORKER_SCRIPT}")
  FIXED_JOB_ID=$(awk '{print $4}' <<< "${FIXED_JOB_LINE}")
fi

cat > "${SUMMARY_JSON}" <<EOF
{
  "python": "${PYTHON_BIN}",
  "output_root": "${OUTPUT_ROOT}",
  "free_manifest": "${FREE_MANIFEST}",
  "fixed_manifest": "${FIXED_MANIFEST}",
  "free_runs_ready": ${N_FREE},
  "fixed_runs_ready": ${N_FIXED},
  "max_concurrent": ${MAX_CONCURRENT},
  "jobs": {
    "free_job_id": "${FREE_JOB_ID}",
    "fixed_job_id": "${FIXED_JOB_ID}"
  }
}
EOF

echo "Submitted classical experiment"
echo "  output_root: ${OUTPUT_ROOT}"
echo "  free runs:   ${N_FREE} (job ${FREE_JOB_ID:-n/a})"
echo "  fixed runs:  ${N_FIXED} (job ${FIXED_JOB_ID:-n/a})"
echo "  summary:     ${SUMMARY_JSON}"
