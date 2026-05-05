#!/bin/bash -x

# Single-test launcher for chemtrain_deploy LAMMPS inputs.
# Uses model/output variables defined inside the input file and prepares paths.
# Usage:
#   sbatch /p/project1/cameo/schmidt36/cameo_md/submit_one_lammps_test.sh inp_lammps_trained_forcevarlike.in
#   sbatch /p/project1/cameo/schmidt36/cameo_md/submit_one_lammps_test.sh /abs/path/to/input.in

#SBATCH --account=cameo
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=04:00:00
#SBATCH --partition=booster
#SBATCH --gres=gpu:1
#SBATCH --job-name=md_single_test
#SBATCH --output=/p/project1/cameo/schmidt36/cameo_md/slurm-%j.out

set -euo pipefail

PROJECT_ROOT=/p/project1/cameo/schmidt36/cameo_md
OUTPUT_ROOT=$PROJECT_ROOT/outputs
LAMMPS_BIN=/p/project1/cameo/schmidt36/lammps/build/lmp

if [[ $# -ne 1 ]]; then
  echo "Usage: sbatch $0 <input.in>" >&2
  exit 2
fi

INPUT_FILE="$1"
if [[ "$INPUT_FILE" != /* ]]; then
  INPUT_FILE="$PROJECT_ROOT/$INPUT_FILE"
fi
if [[ ! -f "$INPUT_FILE" ]]; then
  echo "ERROR: input file not found: $INPUT_FILE" >&2
  exit 3
fi

# Parse: variable <name> string "..."
parse_lammps_string_var() {
  local key="$1"
  awk -v key="$key" '
    $1=="variable" && $2==key && $3=="string" {
      $1=""; $2=""; $3=""
      sub(/^[[:space:]]+/, "", $0)
      sub(/[[:space:]]*#.*/, "", $0)
      gsub(/^"|"$/, "", $0)
      print $0
      exit
    }
  ' "$INPUT_FILE"
}

MODEL_FILE="$(parse_lammps_string_var model_file || true)"
DUMP_DIR="$(parse_lammps_string_var dump_dir || true)"

if [[ -n "$MODEL_FILE" ]]; then
  if [[ "$MODEL_FILE" != /* ]]; then
    MODEL_FILE="$PROJECT_ROOT/$MODEL_FILE"
  fi
  if [[ ! -f "$MODEL_FILE" ]]; then
    echo "ERROR: model_file in input does not exist: $MODEL_FILE" >&2
    exit 4
  fi
fi

source /p/project1/cameo/schmidt36/load_modules.sh
source /p/project1/cameo/schmidt36/set_lammps_paths.sh

export MPICH_GPU_SUPPORT_ENABLED=0
export MPIR_CVAR_CH4_OFI_ENABLE_GPU=0
export PSP_CUDA=0
export TF_CPP_MIN_LOG_LEVEL=2
export CUDA_HOME=/p/software/juwelsbooster/stages/2025/software/CUDA/12
export XLA_FLAGS="--xla_gpu_autotune_level=0 --xla_gpu_cuda_data_dir=$CUDA_HOME"

mkdir -p "$OUTPUT_ROOT"
input_base=$(basename "${INPUT_FILE%.in}")
RUN_OUTDIR=$OUTPUT_ROOT/single_runs/${input_base}/job${SLURM_JOB_ID}
RUN_DUMPDIR=$RUN_OUTDIR/dumps
mkdir -p "$RUN_OUTDIR"
mkdir -p "$RUN_DUMPDIR"

RUN_INPUT="$RUN_OUTDIR/$(basename "$INPUT_FILE")"
cp "$INPUT_FILE" "$RUN_INPUT"

if grep -Eq '^variable[[:space:]]+dump_dir[[:space:]]+string' "$RUN_INPUT"; then
  sed -i "s|^variable[[:space:]]\\+dump_dir[[:space:]]\\+string[[:space:]]\\+.*$|variable dump_dir          string \"$RUN_DUMPDIR\"|" "$RUN_INPUT"
else
  printf '\nvariable dump_dir          string "%s"\n' "$RUN_DUMPDIR" >> "$RUN_INPUT"
fi

cd "$PROJECT_ROOT"

echo "Input      : $INPUT_FILE"
[[ -n "$MODEL_FILE" ]] && echo "Model      : $MODEL_FILE"
echo "Run input  : $RUN_INPUT"
echo "Dump dir   : $RUN_DUMPDIR"
echo "Log dir    : $RUN_OUTDIR"

srun "$LAMMPS_BIN" \
  -in "$RUN_INPUT" \
  -log "$RUN_OUTDIR/lammps.log"
