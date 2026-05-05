#!/usr/bin/env bash
# =============================================================================
# submit_exp_1pro_tica.sh
#
# Run three MLCG variants of 4zohB01 starting directly from a random
# mdCATH h5 frame (no classical CHARMM22 jobs).  Compare their TICA/FES
# to the mdCATH h5 reference trajectory via tica_from_h5.py.
#
# Variants:
#   ml_only     - 1pro_alltemp_cueq_fast_tiled.mlir, NVE + Langevin
#   with_priors - 1pro_alltemp_with_rep.mlir,         NVE + Langevin
#   brownian    - 1pro_alltemp_cueq_fast_tiled.mlir,  fix brownian
#
# Usage:
#   bash submit_exp_1pro_tica.sh [OPTIONS]
#
# Options:
#   --experiment-name NAME     Output name under outputs/  [exp_1pro_tica]
#   --n-seeds N                Number of random h5 frames to use as starts [1]
#   --temp-group T             Temperature group in h5 for seed frames [320]
#   --run-idx R                h5 run index within temp group [0]
#   --seed S                   RNG seed for frame selection [42]
#   --max-concurrent N         Max simultaneous array tasks per variant [3]
#   --tica-temp-group T        Temperature group for h5 TICA reference [320]
#   --skip-tica                Print TICA commands but do not submit them
#   --dry-run                  Prepare manifests but do not submit any jobs
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
BASE="/p/project1/cameo/schmidt36"

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
EXPERIMENT_NAME="exp_1pro_tica"
N_SEEDS=1
TEMP_GROUP="320"
RUN_IDX=0
FRAME_SEED=42
MAX_CONCURRENT=3
TICA_TEMP_GROUP="320"
SKIP_TICA=0
DRY_RUN=0

# ---------------------------------------------------------------------------
# Fixed paths for 1pro_alltemp experiment
# ---------------------------------------------------------------------------
PROTEIN="4zohB01"
H5="${SCRIPT_DIR}/structures/mdcath_dataset_${PROTEIN}.h5"
SPECIES_NPZ="${BASE}/cameo_cg/data_prep/datasets/larger_dataset_320_348_allframes_aggforce_padded/02_cg_npz/${PROTEIN}_cg.npz"

EXPORTS="${BASE}/cameo_cg/local_work/outputs/1pro_alltemp/cueq_fast_tiled/exports"
MODEL_ML_ONLY="${EXPORTS}/1pro_alltemp_symbolic.mlir"
MODEL_WITH_PRIORS="${EXPORTS}/1pro_alltemp_with_rep_symbolic.mlir"

TMPL_ML_ONLY="${SCRIPT_DIR}/inp_lammps_mlcg_1pro_ml_only.in"
TMPL_WITH_PRIORS="${SCRIPT_DIR}/inp_lammps_mlcg_1pro_with_priors.in"
TMPL_BROWNIAN="${SCRIPT_DIR}/inp_lammps_mlcg_1pro_brownian.in"

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --experiment-name)  EXPERIMENT_NAME="${2:-}";  shift 2 ;;
    --n-seeds)          N_SEEDS="${2:-}";           shift 2 ;;
    --temp-group)       TEMP_GROUP="${2:-}";        shift 2 ;;
    --run-idx)          RUN_IDX="${2:-}";           shift 2 ;;
    --seed)             FRAME_SEED="${2:-}";        shift 2 ;;
    --max-concurrent)   MAX_CONCURRENT="${2:-}";    shift 2 ;;
    --tica-temp-group)  TICA_TEMP_GROUP="${2:-}";   shift 2 ;;
    --skip-tica)        SKIP_TICA=1;                shift   ;;
    --dry-run)          DRY_RUN=1;                  shift   ;;
    -h|--help) grep '^#' "$0" | grep -v '^#!/' | sed 's/^# \?//'; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
if [[ -f "${BASE}/load_modules.sh" ]]; then
  source "${BASE}/load_modules.sh"
fi
if [[ -f "${BASE}/test_env_newsetup/bin/activate" ]]; then
  source "${BASE}/test_env_newsetup/bin/activate"
fi
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3 || command -v python)}"
[[ -z "${PYTHON_BIN}" ]] && { echo "ERROR: no python interpreter found" >&2; exit 2; }

# ---------------------------------------------------------------------------
# Validate
# ---------------------------------------------------------------------------
for f in "${H5}" "${SPECIES_NPZ}" "${MODEL_ML_ONLY}" "${MODEL_WITH_PRIORS}" \
          "${TMPL_ML_ONLY}" "${TMPL_WITH_PRIORS}" "${TMPL_BROWNIAN}"; do
  [[ -f "$f" ]] || { echo "ERROR: required file not found: $f" >&2; exit 2; }
done

EXP_ROOT="${SCRIPT_DIR}/outputs/${EXPERIMENT_NAME}"
mkdir -p "${EXP_ROOT}"

echo "============================================================"
echo "[exp_1pro_tica] root        : ${EXP_ROOT}"
echo "[exp_1pro_tica] protein     : ${PROTEIN}"
echo "[exp_1pro_tica] h5          : ${H5}"
echo "[exp_1pro_tica] temp_group  : ${TEMP_GROUP}  run_idx=${RUN_IDX}  n_seeds=${N_SEEDS}"
echo "[exp_1pro_tica] frame seed  : ${FRAME_SEED}"
echo "[exp_1pro_tica] dry_run     : ${DRY_RUN}"
echo "============================================================"

# ---------------------------------------------------------------------------
# Variant definitions: parallel arrays
# ---------------------------------------------------------------------------
VARIANTS=("ml_only" "with_priors" "brownian")
declare -A VARIANT_MODEL=(
  [ml_only]="${MODEL_ML_ONLY}"
  [with_priors]="${MODEL_WITH_PRIORS}"
  [brownian]="${MODEL_ML_ONLY}"
)
declare -A VARIANT_TMPL=(
  [ml_only]="${TMPL_ML_ONLY}"
  [with_priors]="${TMPL_WITH_PRIORS}"
  [brownian]="${TMPL_BROWNIAN}"
)

declare -A VARIANT_JOB_ID=()

# ---------------------------------------------------------------------------
# Step 1: Prepare and submit MLCG runs for each variant
# ---------------------------------------------------------------------------
for VARIANT in "${VARIANTS[@]}"; do
  MLCG_OUT="${EXP_ROOT}/mlcg_${VARIANT}"
  READY_MANIFEST="${MLCG_OUT}/mlcg_ready_manifest.tsv"

  echo ""
  echo "[mlcg] variant=${VARIANT}"

  PREP_ARGS=(
    --h5               "${H5}"
    --protein          "${PROTEIN}"
    --species-npz      "${SPECIES_NPZ}"
    --trained-model    "${VARIANT_MODEL[$VARIANT]}"
    --input-template   "${VARIANT_TMPL[$VARIANT]}"
    --mlcg-out-dir     "${MLCG_OUT}"
    --temp-group       "${TEMP_GROUP}"
    --run-idx          "${RUN_IDX}"
    --n-seeds          "${N_SEEDS}"
    --seed             "${FRAME_SEED}"
    --run-id-suffix    "__${VARIANT}"
  )
  [[ "${DRY_RUN}" -eq 1 ]] && PREP_ARGS+=(--dry-run)

  "${PYTHON_BIN}" "${SCRIPT_DIR}/prepare_mlcg_from_h5.py" "${PREP_ARGS[@]}"

  N_READY=0
  [[ -f "${READY_MANIFEST}" ]] && N_READY=$(awk 'END{print NR}' "${READY_MANIFEST}")
  echo "         ready runs: ${N_READY}"

  if [[ "${N_READY}" -eq 0 || "${DRY_RUN}" -eq 1 ]]; then
    echo "         [skipped submission]"
    VARIANT_JOB_ID[$VARIANT]=""
    continue
  fi

  JOB_LINE=$(sbatch \
    --account=cameo \
    --nodes=1 \
    --ntasks-per-node=1 \
    --partition=booster \
    --gres=gpu:1 \
    --time=02:00:00 \
    --job-name="mlcg_${VARIANT}" \
    --output="${MLCG_OUT}/slurm-mlcg-${VARIANT}-%A_%a.out" \
    --array="0-$((N_READY - 1))%${MAX_CONCURRENT}" \
    --export=ALL,EXPERIMENT_MANIFEST="${READY_MANIFEST}" \
    "${SCRIPT_DIR}/run_experiment_mlcg_array.sh")

  JOB_ID=$(awk '{print $4}' <<< "${JOB_LINE}")
  VARIANT_JOB_ID[$VARIANT]="${JOB_ID}"
  echo "         submitted job: ${JOB_ID}"
done

# ---------------------------------------------------------------------------
# Step 2: TICA analysis (one job per variant, depends on MLCG)
# ---------------------------------------------------------------------------
# tica_from_h5.py experiment mode searches for:
#   <experiment-root>/mlcg/mlcg_manifest.tsv   (primary)
#   <experiment-root>/mlcg_manifest.tsv         (fallback)
# Passing --experiment-root = mlcg_${VARIANT}/ hits the fallback directly.
#
# The manifest uses seed_mode="h5"; tica_from_h5.py filters by seed_mode via
# --mlcg-set.  We pass --mlcg-set h5 so only h5-seeded runs are processed.
# ---------------------------------------------------------------------------
echo ""
echo "[tica] submitting TICA analysis jobs"

for VARIANT in "${VARIANTS[@]}"; do
  MLCG_OUT="${EXP_ROOT}/mlcg_${VARIANT}"
  TICA_OUT="${EXP_ROOT}/tica_${VARIANT}"

  TICA_ARGS=(
    --experiment-root "${MLCG_OUT}"
    --h5-dir          "${SCRIPT_DIR}/structures"
    --outdir          "${TICA_OUT}"
    --temp-group      "${TICA_TEMP_GROUP}"
    --lagtime         10
    --bins            80
    --temperature     320.0
    --n-pairs         200
    --pair-seed       42
    --mlcg-set        h5
  )

  echo "         variant=${VARIANT}  outdir=${TICA_OUT}"

  if [[ "${SKIP_TICA}" -eq 1 || "${DRY_RUN}" -eq 1 ]]; then
    echo "         [skipped] would run: bash submit_tica_h5.sh ${TICA_ARGS[*]}"
    continue
  fi

  DEP_ARGS=()
  JOB="${VARIANT_JOB_ID[$VARIANT]:-}"
  [[ -n "${JOB}" ]] && DEP_ARGS+=(--dependency="afterok:${JOB}")

  TICA_JOB_LINE=$(sbatch \
    "${DEP_ARGS[@]}" \
    --account=cameo \
    --nodes=1 \
    --ntasks-per-node=1 \
    --partition=booster \
    --time=01:00:00 \
    --job-name="tica_${VARIANT}" \
    --output="${TICA_OUT}/slurm-tica-${VARIANT}-%j.out" \
    "${SCRIPT_DIR}/submit_tica_h5.sh" "${TICA_ARGS[@]}")

  echo "         submitted job: $(awk '{print $4}' <<< "${TICA_JOB_LINE}")"
done

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "Experiment: ${EXP_ROOT}"
for VARIANT in "${VARIANTS[@]}"; do
  echo "  mlcg_${VARIANT}/  job=${VARIANT_JOB_ID[$VARIANT]:-dry-run}"
done
echo ""
echo "Force noise estimation (run independently, no GPU needed):"
echo "  python ${SCRIPT_DIR}/estimate_force_noise.py \\"
echo "    --h5 ${H5} --protein ${PROTEIN} \\"
echo "    --outdir ${EXP_ROOT}/force_noise \\"
echo "    --all-temp-groups"
echo "============================================================"
