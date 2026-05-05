#!/bin/bash -x

#SBATCH --account=cameo
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=01:00:00
#SBATCH --partition=booster
#SBATCH --gres=gpu:4

source /p/project1/cameo/schmidt36/load_modules.sh
#source /p/project1/cameo/schmidt36/clean_booster_env/bin/activate
source /p/project1/cameo/schmidt36/env_cueq_allegro_opt/bin/activate
source /p/project1/cameo/schmidt36/set_lammps_paths.sh

# Defaults for standalone usage.
PROJECT_ROOT=${PROJECT_ROOT:-/p/project1/cameo/schmidt36/cameo_md}
INPUT_FILE=${INPUT_FILE:-$PROJECT_ROOT/inp_lammps_trained_forcevarlike.in}

# Optional suite mode: map each array task to a run directory and input file.
# SUITE_MANIFEST format (tab-separated, one line per task):
#   <project_root>\t<input_file>\t<frame_index>
if [[ -n "${SUITE_MANIFEST:-}" && -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    if [[ ! -f "$SUITE_MANIFEST" ]]; then
        echo "ERROR: SUITE_MANIFEST not found: $SUITE_MANIFEST" >&2
        exit 2
    fi

    line=$(awk -F '\t' -v n="$((SLURM_ARRAY_TASK_ID + 1))" 'NR==n {print; exit}' "$SUITE_MANIFEST")
    if [[ -z "$line" ]]; then
        echo "ERROR: no manifest entry for SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID" >&2
        exit 3
    fi

    IFS=$'\t' read -r PROJECT_ROOT INPUT_FILE FRAME_INDEX <<< "$line"
    export FRAME_INDEX
    echo "[suite] task=$SLURM_ARRAY_TASK_ID frame=${FRAME_INDEX:-NA} project_root=$PROJECT_ROOT input_file=$INPUT_FILE"
fi

cd "$PROJECT_ROOT"

echo "[MLIR preflight] disabled for this test run"

export CUDA_VISIBLE_DEVICES=0,1,2,3

export XLA_PYTHON_CLIENT_PREALLOCATE=true
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90

export TF_CPP_MIN_LOG_LEVEL=2
export CUDA_HOME=/p/software/juwelsbooster/stages/2025/software/CUDA/12
export XLA_FLAGS="--xla_gpu_autotune_level=0 --xla_gpu_cuda_data_dir=$CUDA_HOME"

srun /p/project1/cameo/schmidt36/lammps/build/lmp -in "$INPUT_FILE"
