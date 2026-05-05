#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${EXPERIMENT_MANIFEST:-}" ]]; then
  echo "ERROR: EXPERIMENT_MANIFEST is not set" >&2
  exit 2
fi
if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "ERROR: SLURM_ARRAY_TASK_ID is not set" >&2
  exit 3
fi
if [[ ! -f "${EXPERIMENT_MANIFEST}" ]]; then
  echo "ERROR: manifest not found: ${EXPERIMENT_MANIFEST}" >&2
  exit 4
fi

line=$(awk -F '\t' -v n="$((SLURM_ARRAY_TASK_ID + 1))" 'NR==n {print; exit}' "${EXPERIMENT_MANIFEST}")
if [[ -z "${line}" ]]; then
  echo "ERROR: no manifest row for task ${SLURM_ARRAY_TASK_ID}" >&2
  exit 5
fi

IFS=$'\t' read -r RUN_ID PROTEIN SEED_MODE RUN_DIR INPUT_FILE <<< "${line}"

if [[ -z "${RUN_DIR}" || -z "${INPUT_FILE}" ]]; then
  echo "ERROR: malformed manifest row: ${line}" >&2
  exit 6
fi

mkdir -p "${RUN_DIR}"

BASE=/p/project1/cameo/schmidt36
LAMMPS_BIN=${LAMMPS_BIN:-${BASE}/lammps/build/lmp}

if [[ -f "${BASE}/load_modules.sh" ]]; then
  # shellcheck disable=SC1090
  source "${BASE}/load_modules.sh"
fi

# Prefer the original MLCG runtime env; fall back to test_env_newsetup.
if [[ -f "${BASE}/env_cueq_allegro_opt/bin/activate" ]]; then
  # shellcheck disable=SC1090
  source "${BASE}/env_cueq_allegro_opt/bin/activate"
elif [[ -f "${BASE}/test_env_newsetup/bin/activate" ]]; then
  # shellcheck disable=SC1090
  source "${BASE}/test_env_newsetup/bin/activate"
fi

# Ensure stale shell exports do not override canonical connector paths.
unset LAMMPS_PLUGIN_PATH
unset JCN_LIB_PATH
unset JCN_PJRT_PATH
unset JCN_PJRT_PLUGIN

# Canonical connector/PJRT runtime (as requested).
# shellcheck disable=SC1090
source "${BASE}/set_lammps_paths.sh"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
export TF_CPP_MIN_LOG_LEVEL=2
export CUDA_HOME=/p/software/juwelsbooster/stages/2025/software/CUDA/12
export XLA_FLAGS="--xla_gpu_autotune_level=0 --xla_gpu_cuda_data_dir=$CUDA_HOME"

echo "[mlcg-array] task=${SLURM_ARRAY_TASK_ID} run_id=${RUN_ID} protein=${PROTEIN} seed=${SEED_MODE}"
echo "[mlcg-array] input=${INPUT_FILE}"
echo "[mlcg-array] run_dir=${RUN_DIR}"
echo "[mlcg-array] python=$(command -v python || true)"
echo "[mlcg-array] LAMMPS_PLUGIN_PATH=${LAMMPS_PLUGIN_PATH:-<unset>}"
echo "[mlcg-array] JCN_LIB_PATH=${JCN_LIB_PATH:-<unset>}"
echo "[mlcg-array] JCN_PJRT_PATH=${JCN_PJRT_PATH:-<unset>}"
echo "[mlcg-array] JCN_PJRT_PLUGIN=${JCN_PJRT_PLUGIN:-<unset>}"

srun "${LAMMPS_BIN}" -in "${INPUT_FILE}" -log "${RUN_DIR}/lammps.log"
