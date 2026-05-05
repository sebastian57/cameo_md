#!/usr/bin/env bash
#SBATCH --account=cameo
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=04:00:00
#SBATCH --partition=booster
#SBATCH --job-name=exp_analysis
#SBATCH --output=/p/project1/cameo/schmidt36/cameo_md/slurm-exp-analysis-%j.out

set -euo pipefail

BASE="/p/project1/cameo/schmidt36"
DEFAULT_SCRIPT_DIR="${BASE}/cameo_md"

if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/run_experiment_analysis.sh" ]]; then
  SCRIPT_DIR="${SLURM_SUBMIT_DIR}"
else
  SCRIPT_DIR="${DEFAULT_SCRIPT_DIR}"
fi

RUNNER="${SCRIPT_DIR}/run_experiment_analysis.sh"

if [[ ! -f "${RUNNER}" ]]; then
  echo "ERROR: missing analysis runner: ${RUNNER}" >&2
  exit 2
fi

if [[ -f "${BASE}/load_modules.sh" ]]; then
  # shellcheck disable=SC1090
  source "${BASE}/load_modules.sh"
fi

if [[ -f "${BASE}/test_env_newsetup/bin/activate" ]]; then
  # shellcheck disable=SC1090
  source "${BASE}/test_env_newsetup/bin/activate"
fi

echo "[submit_experiment_analysis] host=$(hostname) job=${SLURM_JOB_ID:-n/a} start=$(date -Is)"
echo "[submit_experiment_analysis] runner=${RUNNER}"
echo "[submit_experiment_analysis] args: $*"

bash "${RUNNER}" "$@"

echo "[submit_experiment_analysis] done $(date -Is)"
