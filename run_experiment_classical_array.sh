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

IFS=$'\t' read -r RUN_ID PROTEIN MODE RUN_DIR INPUT_FILE <<< "${line}"

if [[ -z "${RUN_DIR}" || -z "${INPUT_FILE}" ]]; then
  echo "ERROR: malformed manifest row: ${line}" >&2
  exit 6
fi

mkdir -p "${RUN_DIR}"

BASE=/p/project1/cameo/schmidt36
LAMMPS_BIN=${LAMMPS_BIN:-${BASE}/lammps/build/lmp}
source ${BASE}/load_modules.sh
source ${BASE}/set_lammps_paths.sh

# Stable defaults used in current classical scripts.
export MPICH_GPU_SUPPORT_ENABLED=0
export MPIR_CVAR_CH4_OFI_ENABLE_GPU=0
export PSP_CUDA=0
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}

echo "[classical-array] task=${SLURM_ARRAY_TASK_ID} run_id=${RUN_ID} protein=${PROTEIN} mode=${MODE}"
echo "[classical-array] input=${INPUT_FILE}"
echo "[classical-array] run_dir=${RUN_DIR}"

srun "${LAMMPS_BIN}" -in "${INPUT_FILE}" -log "${RUN_DIR}/lammps.log"
