#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE="/p/project1/cameo/schmidt36"

VENV_DEFAULT="/p/project1/cameo/schmidt36/test_env_newsetup"
VENV_PATH="${VENV_DEFAULT}"
EXPERIMENT_ROOT=""
MODEL_MLIR=""
CG_NPZ_MAP=""
ANALYSIS_ROOT=""
MLCG_SET="both"
MATCH_LENGTH="min"
MAX_FRAMES="0"
LAGTIME="10"
BINS="80"
PAIR_MODE="random"
N_PAIRS="200"
PAIR_SEED="42"
FRAME_STRIDE="1"
INCLUDE_INCOMPLETE="0"

usage() {
  cat <<EOF_USAGE
Usage:
  $(basename "$0") \
    --experiment-root <outputs/<experiment_name>> \
    --model-mlir <model.mlir> \
    --cg-npz-map <protein->cg map csv/json> \
    --analysis-root <output dir> \
    [--venv <venv path>] \
    [--mlcg-set both|t0|eq] \
    [--match-length min|classical] \
    [--max-frames N] [--lagtime N] [--bins N] \
    [--pair-mode random|sequential] [--n-pairs N] [--pair-seed N] [--frame-stride N] \
    [--include-incomplete]
EOF_USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --experiment-root) EXPERIMENT_ROOT="${2:-}"; shift 2 ;;
    --model-mlir) MODEL_MLIR="${2:-}"; shift 2 ;;
    --cg-npz-map) CG_NPZ_MAP="${2:-}"; shift 2 ;;
    --analysis-root) ANALYSIS_ROOT="${2:-}"; shift 2 ;;
    --venv) VENV_PATH="${2:-}"; shift 2 ;;
    --mlcg-set) MLCG_SET="${2:-}"; shift 2 ;;
    --match-length) MATCH_LENGTH="${2:-}"; shift 2 ;;
    --max-frames) MAX_FRAMES="${2:-}"; shift 2 ;;
    --lagtime) LAGTIME="${2:-}"; shift 2 ;;
    --bins) BINS="${2:-}"; shift 2 ;;
    --pair-mode) PAIR_MODE="${2:-}"; shift 2 ;;
    --n-pairs) N_PAIRS="${2:-}"; shift 2 ;;
    --pair-seed) PAIR_SEED="${2:-}"; shift 2 ;;
    --frame-stride) FRAME_STRIDE="${2:-}"; shift 2 ;;
    --include-incomplete) INCLUDE_INCOMPLETE="1"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ -z "${EXPERIMENT_ROOT}" || -z "${MODEL_MLIR}" || -z "${CG_NPZ_MAP}" || -z "${ANALYSIS_ROOT}" ]]; then
  echo "ERROR: --experiment-root, --model-mlir, --cg-npz-map, --analysis-root are required" >&2
  usage
  exit 2
fi

if [[ ! -f "${VENV_PATH}/bin/activate" ]]; then
  echo "ERROR: venv activate script not found: ${VENV_PATH}/bin/activate" >&2
  exit 2
fi

mkdir -p "${ANALYSIS_ROOT}/option3" "${ANALYSIS_ROOT}/tica"

# Ensure the same runtime stack as MD/MLCG jobs.
if [[ -f "${BASE}/load_modules.sh" ]]; then
  # shellcheck disable=SC1090
  source "${BASE}/load_modules.sh"
fi
if [[ -f "${BASE}/set_lammps_paths.sh" ]]; then
  # shellcheck disable=SC1090
  source "${BASE}/set_lammps_paths.sh"
fi
source "${VENV_PATH}/bin/activate"

OPTION3_ARGS=(
  --experiment-root "${EXPERIMENT_ROOT}"
  --model-mlir "${MODEL_MLIR}"
  --cg-npz-map "${CG_NPZ_MAP}"
  --out-dir "${ANALYSIS_ROOT}/option3"
  --max-frames "${MAX_FRAMES}"
)
TICA_ARGS=(
  --experiment-root "${EXPERIMENT_ROOT}"
  --outdir "${ANALYSIS_ROOT}/tica"
  --mlcg-set "${MLCG_SET}"
  --match-length "${MATCH_LENGTH}"
  --lagtime "${LAGTIME}"
  --bins "${BINS}"
  --pair-mode "${PAIR_MODE}"
  --n-pairs "${N_PAIRS}"
  --pair-seed "${PAIR_SEED}"
  --frame-stride "${FRAME_STRIDE}"
  --max-frames "${MAX_FRAMES}"
)

if [[ "${INCLUDE_INCOMPLETE}" == "1" ]]; then
  OPTION3_ARGS+=(--include-incomplete)
  TICA_ARGS+=(--include-incomplete)
fi

python "${SCRIPT_DIR}/option3_fixed_ca_analysis.py" "${OPTION3_ARGS[@]}"
python "${SCRIPT_DIR}/tica_from_lammps_dump.py" "${TICA_ARGS[@]}"

echo "Analysis complete"
echo "Option3 summary: ${ANALYSIS_ROOT}/option3/option3_summary.csv"
echo "TICA summary: ${ANALYSIS_ROOT}/tica/tica_summary.csv"
