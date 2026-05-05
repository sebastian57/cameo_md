#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GEN_SCRIPT="$SCRIPT_DIR/lmp_input_gen.py"
SUBMIT_SCRIPT="$SCRIPT_DIR/submit_lammps_chemtrain.sh"
LOAD_MODULES="/p/project1/cameo/schmidt36/load_modules.sh"
CLEAN_ENV="/p/project1/cameo/schmidt36/clean_booster_env/bin/activate"

DATASET_DEFAULT="$SCRIPT_DIR/structures/datasets/2g4q4z5k_320K_kcalmol_1bead_notnorm_aggforce.npz"

usage() {
    cat <<USAGE
Usage:
  bash submit_md_suite.sh --structures 5 --input_file /abs/path/to/in.lmp
  bash submit_md_suite.sh --structures [0,1000,535,9,87] --input_file /abs/path/to/in.lmp

Required:
  --structures       Either an integer count (random frames) or an index list.
  --input_file       LAMMPS input template used for all runs.

Optional:
  --max_concurrent   Max concurrent array tasks (default: 4)
  --dataset          NPZ dataset path (default: $DATASET_DEFAULT)
  --suite_dir        Output suite directory (default: cameo_md/outputs/md_suites/<input>_<timestamp>)
  --seed             RNG seed for random selection (default: 12345)
  --safety_factor    Structure box safety factor passed to lmp_input_gen.py (default: 1.20)
  --min_half_width   Minimum half-width for auto box mode (default: 20.0)
USAGE
}

STRUCTURES_SPEC=""
INPUT_FILE=""
MAX_CONCURRENT=4
DATASET="$DATASET_DEFAULT"
SEED=12345
SUITE_DIR=""
SAFETY_FACTOR=1.20
MIN_HALF_WIDTH=20.0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --structures)
            STRUCTURES_SPEC="${2:-}"
            shift 2
            ;;
        --input_file)
            INPUT_FILE="${2:-}"
            shift 2
            ;;
        --max_concurrent)
            MAX_CONCURRENT="${2:-}"
            shift 2
            ;;
        --dataset)
            DATASET="${2:-}"
            shift 2
            ;;
        --suite_dir)
            SUITE_DIR="${2:-}"
            shift 2
            ;;
        --seed)
            SEED="${2:-}"
            shift 2
            ;;
        --safety_factor)
            SAFETY_FACTOR="${2:-}"
            shift 2
            ;;
        --min_half_width)
            MIN_HALF_WIDTH="${2:-}"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage
            exit 1
            ;;
    esac
done

if [[ -z "$STRUCTURES_SPEC" || -z "$INPUT_FILE" ]]; then
    usage
    exit 1
fi

if [[ ! "$MAX_CONCURRENT" =~ ^[0-9]+$ ]] || [[ "$MAX_CONCURRENT" -lt 1 ]]; then
    echo "ERROR: --max_concurrent must be a positive integer" >&2
    exit 1
fi

if [[ ! "$SEED" =~ ^[0-9]+$ ]]; then
    echo "ERROR: --seed must be an integer" >&2
    exit 1
fi

if [[ ! -f "$DATASET" ]]; then
    echo "ERROR: dataset not found: $DATASET" >&2
    exit 1
fi

if [[ ! -f "$INPUT_FILE" ]]; then
    echo "ERROR: input_file not found: $INPUT_FILE" >&2
    exit 1
fi

if [[ ! -f "$GEN_SCRIPT" ]]; then
    echo "ERROR: generator not found: $GEN_SCRIPT" >&2
    exit 1
fi

if [[ ! -f "$SUBMIT_SCRIPT" ]]; then
    echo "ERROR: submit script not found: $SUBMIT_SCRIPT" >&2
    exit 1
fi

INPUT_FILE="$(realpath "$INPUT_FILE")"
DATASET="$(realpath "$DATASET")"

if [[ -z "$SUITE_DIR" ]]; then
    INPUT_BASE="$(basename "${INPUT_FILE%.in}")"
    SUITE_DIR="$SCRIPT_DIR/outputs/md_suites/${INPUT_BASE}_$(date +%Y%m%d_%H%M%S)"
fi
SUITE_DIR="$(realpath -m "$SUITE_DIR")"
mkdir -p "$SUITE_DIR"

N_FRAMES=$(bash -lc "source '$LOAD_MODULES' && source '$CLEAN_ENV' && python -c \"import numpy as np; d=np.load(r'$DATASET', allow_pickle=True); print(d['R'].shape[0])\"")
if [[ -z "$N_FRAMES" ]] || [[ ! "$N_FRAMES" =~ ^[0-9]+$ ]]; then
    echo "ERROR: failed to determine number of frames from dataset" >&2
    exit 1
fi

FRAME_LIST=""
if [[ "$STRUCTURES_SPEC" =~ ^[0-9]+$ ]]; then
    COUNT="$STRUCTURES_SPEC"
    if [[ "$COUNT" -lt 1 ]]; then
        echo "ERROR: --structures count must be >= 1" >&2
        exit 1
    fi
    if [[ "$COUNT" -gt "$N_FRAMES" ]]; then
        echo "ERROR: requested $COUNT random structures, but dataset has only $N_FRAMES frames" >&2
        exit 1
    fi

    FRAME_LIST=$(bash -lc "source '$LOAD_MODULES' && source '$CLEAN_ENV' && python -c \"import numpy as np; rng=np.random.default_rng($SEED); idx=rng.choice($N_FRAMES, size=$COUNT, replace=False); print(','.join(str(int(i)) for i in idx.tolist()))\"")
else
    CLEAN_SPEC="$(echo "$STRUCTURES_SPEC" | tr -d '[][:space:]')"
    if [[ -z "$CLEAN_SPEC" ]]; then
        echo "ERROR: --structures list is empty" >&2
        exit 1
    fi
    IFS=',' read -r -a IDS <<< "$CLEAN_SPEC"
    for id in "${IDS[@]}"; do
        if [[ ! "$id" =~ ^[0-9]+$ ]]; then
            echo "ERROR: invalid structure index in list: $id" >&2
            exit 1
        fi
        if [[ "$id" -lt 0 || "$id" -ge "$N_FRAMES" ]]; then
            echo "ERROR: structure index out of range: $id (dataset frames: 0..$((N_FRAMES - 1)))" >&2
            exit 1
        fi
    done
    FRAME_LIST="$CLEAN_SPEC"
fi

IFS=',' read -r -a FRAMES <<< "$FRAME_LIST"
N_RUNS="${#FRAMES[@]}"
if [[ "$N_RUNS" -lt 1 ]]; then
    echo "ERROR: no structures selected" >&2
    exit 1
fi

MANIFEST="$SUITE_DIR/manifest.tsv"
: > "$MANIFEST"

echo "Preparing $N_RUNS run directories under $SUITE_DIR"

for idx in "${FRAMES[@]}"; do
    RUN_DIR="$SUITE_DIR/frame_${idx}"
    mkdir -p "$RUN_DIR"

    STRUCT_FILE="$RUN_DIR/structure.data"
    RUN_INPUT="$RUN_DIR/$(basename "$INPUT_FILE")"
    RUN_DUMPDIR="$RUN_DIR/dumps"
    mkdir -p "$RUN_DUMPDIR"

    bash -lc "source '$LOAD_MODULES' && source '$CLEAN_ENV' && \
      python '$GEN_SCRIPT' \
        --dataset '$DATASET' \
        --frame '$idx' \
        --out '$STRUCT_FILE' \
        --box-mode auto \
        --safety-factor '$SAFETY_FACTOR' \
        --min-half-width '$MIN_HALF_WIDTH'"

    cp "$INPUT_FILE" "$RUN_INPUT"

    if ! grep -Eq '^variable[[:space:]]+data_file[[:space:]]+string' "$RUN_INPUT"; then
        echo "ERROR: input template has no 'variable data_file string ...' line: $RUN_INPUT" >&2
        exit 1
    fi

    sed -i "s|^variable[[:space:]]\+data_file[[:space:]]\+string[[:space:]]\+.*$|variable data_file         string \"$STRUCT_FILE\"|" "$RUN_INPUT"
    if grep -Eq '^variable[[:space:]]+dump_dir[[:space:]]+string' "$RUN_INPUT"; then
        sed -i "s|^variable[[:space:]]\\+dump_dir[[:space:]]\\+string[[:space:]]\\+.*$|variable dump_dir          string \"$RUN_DUMPDIR\"|" "$RUN_INPUT"
    else
        printf '\nvariable dump_dir          string "%s"\n' "$RUN_DUMPDIR" >> "$RUN_INPUT"
    fi
    printf "%s\t%s\t%s\n" "$RUN_DIR" "$RUN_INPUT" "$idx" >> "$MANIFEST"
done

ARRAY_SPEC="0-$((N_RUNS - 1))%$MAX_CONCURRENT"
echo "Submitting array job: $ARRAY_SPEC"

JOB_LINE=$(sbatch --array="$ARRAY_SPEC" --export=ALL,SUITE_MANIFEST="$MANIFEST" "$SUBMIT_SCRIPT")
JOB_ID=$(echo "$JOB_LINE" | awk '{print $4}')

echo "Submitted suite job: $JOB_ID"
echo "Manifest: $MANIFEST"
echo "Runs: $N_RUNS"
