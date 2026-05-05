#!/usr/bin/env bash
#SBATCH --account=cameo
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=01:00:00
#SBATCH --partition=booster
#SBATCH --job-name=tica_h5
#SBATCH --output=/p/project1/cameo/schmidt36/cameo_md/slurm-tica-h5-%j.out

# =============================================================================
# Submit the h5-based TICA analysis (tica_from_h5.py, experiment mode).
#
# Direct invocation (auto-submits via sbatch):
#   bash submit_tica_h5.sh \
#       --experiment-root outputs/exp_v2 \
#       --h5-dir          structures/larger_dataset \
#       --outdir          outputs/exp_v2/tica_h5
#
# Or submit directly:
#   sbatch submit_tica_h5.sh \
#       --experiment-root outputs/exp_v2 \
#       --h5-dir          structures/larger_dataset \
#       --outdir          outputs/exp_v2/tica_h5
# =============================================================================

# Auto-submit when called with bash instead of sbatch
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    if [[ $# -lt 1 ]]; then
        echo "Usage: bash submit_tica_h5.sh --experiment-root <dir> --h5-dir <dir> --outdir <dir> [opts]"
        exit 1
    fi
    SCRIPT_SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)/$(basename "${BASH_SOURCE[0]}")"
    OUTDIR_ARG=""
    _args=("$@")
    for (( _i=0; _i<${#_args[@]}; _i++ )); do
        if [[ "${_args[$_i]}" == "--outdir" ]]; then
            OUTDIR_ARG="${_args[$(( _i + 1 ))]:-}"
            break
        fi
    done
    # derive a log location from the outdir or fall back to cameo_md
    if [[ -n "${OUTDIR_ARG}" ]]; then
        mkdir -p "${OUTDIR_ARG}"
        LOG="${OUTDIR_ARG}/slurm-tica-h5-%j.out"
    else
        LOG="/p/project1/cameo/schmidt36/cameo_md/slurm-tica-h5-%j.out"
    fi
    echo "Auto-submitting via sbatch (log: ${LOG})"
    sbatch --output="${LOG}" "${SCRIPT_SELF}" "$@"
    exit 0
fi

set -Eeuo pipefail

BASE="/p/project1/cameo/schmidt36"
SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)}"
VENV_DEFAULT="${BASE}/test_env_newsetup"

EXPERIMENT_ROOT=""
H5_DIR=""
OUTDIR=""
TEMP_GROUP="320"
LAGTIME="10"
BINS="80"
TEMPERATURE="320.0"
N_PAIRS="200"
PAIR_SEED="42"
MLCG_SET="eq"
MAX_MLCG_FRAMES="0"
SHARED_BINS="1"
VENV="${VENV_DEFAULT}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --experiment-root) EXPERIMENT_ROOT="${2:-}"; shift 2 ;;
    --h5-dir)          H5_DIR="${2:-}";          shift 2 ;;
    --outdir)          OUTDIR="${2:-}";           shift 2 ;;
    --temp-group)      TEMP_GROUP="${2:-}";       shift 2 ;;
    --lagtime)         LAGTIME="${2:-}";          shift 2 ;;
    --bins)            BINS="${2:-}";             shift 2 ;;
    --temperature)     TEMPERATURE="${2:-}";      shift 2 ;;
    --n-pairs)         N_PAIRS="${2:-}";          shift 2 ;;
    --pair-seed)       PAIR_SEED="${2:-}";        shift 2 ;;
    --mlcg-set)        MLCG_SET="${2:-}";         shift 2 ;;
    --max-mlcg-frames) MAX_MLCG_FRAMES="${2:-}";  shift 2 ;;
    --no-shared-bins)  SHARED_BINS="0";           shift ;;
    --venv)            VENV="${2:-}";             shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

if [[ -z "${EXPERIMENT_ROOT}" || -z "${H5_DIR}" || -z "${OUTDIR}" ]]; then
  echo "ERROR: --experiment-root, --h5-dir, and --outdir are required" >&2
  exit 2
fi

# Resolve relative paths from the submit dir
SUBMIT_WORKDIR="${SLURM_SUBMIT_DIR:-$(pwd -P)}"
[[ "${EXPERIMENT_ROOT}" != /* ]] && EXPERIMENT_ROOT="${SUBMIT_WORKDIR}/${EXPERIMENT_ROOT}"
[[ "${H5_DIR}"          != /* ]] && H5_DIR="${SUBMIT_WORKDIR}/${H5_DIR}"
[[ "${OUTDIR}"          != /* ]] && OUTDIR="${SUBMIT_WORKDIR}/${OUTDIR}"

echo "[tica_h5] job=${SLURM_JOB_ID} host=$(hostname) start=$(date -Is)"
echo "[tica_h5] experiment_root=${EXPERIMENT_ROOT}"
echo "[tica_h5] h5_dir=${H5_DIR}"
echo "[tica_h5] outdir=${OUTDIR}"

if [[ -f "${BASE}/load_modules.sh" ]]; then
  source "${BASE}/load_modules.sh"
fi

if [[ ! -f "${VENV}/bin/activate" ]]; then
  echo "ERROR: venv not found: ${VENV}/bin/activate" >&2
  exit 2
fi
source "${VENV}/bin/activate"

mkdir -p "${OUTDIR}"

ARGS=(
  --experiment-root "${EXPERIMENT_ROOT}"
  --h5-dir          "${H5_DIR}"
  --outdir          "${OUTDIR}"
  --temp-group      "${TEMP_GROUP}"
  --lagtime         "${LAGTIME}"
  --bins            "${BINS}"
  --temperature     "${TEMPERATURE}"
  --n-pairs         "${N_PAIRS}"
  --pair-seed       "${PAIR_SEED}"
  --mlcg-set        "${MLCG_SET}"
  --max-mlcg-frames "${MAX_MLCG_FRAMES}"
)
[[ "${SHARED_BINS}" == "1" ]] && ARGS+=(--shared-bins)

python "${SCRIPT_DIR}/tica_from_h5.py" "${ARGS[@]}"

echo "[tica_h5] done $(date -Is)"
echo "[tica_h5] summary: ${OUTDIR}/tica_h5_summary.csv"
