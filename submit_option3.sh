#!/bin/bash -x
#SBATCH --account=cameo
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=02:00:00
#SBATCH --partition=booster
#SBATCH --gres=gpu:1
#SBATCH --job-name=opt3_noisefloor
#SBATCH --output=/p/project1/cameo/schmidt36/cameo_md/runs/forcevar_320K/option3_slurm-%j.out

# =============================================================================
# Option 3 — Mean-force decomposition at fixed CA positions
#
# Runs option3_fixed_ca_analysis.py for the 3 completed proteins.
# Three models are evaluated:
#   agg1  — aggforce, no priors  (training_md_testing / tiled_cueq_fast_config)
#   agg2  — aggforce, priors cfg (training_md_testing / tiled_cueq_fast_priors_config)
#   noagg — no aggforce          (noagg_test         / tiled_cueq_fast_config_noagg)
#
# The script is run twice per protein (agg1+noagg, agg2+noagg) writing to
# separate output directories so results are not overwritten.
#
# Usage:
#   sbatch cameo_md/submit_option3.sh
#   # or interactively (no GPU, stats only — remove --mlir-* args):
#   bash cameo_md/submit_option3.sh
# =============================================================================

source /p/project1/cameo/schmidt36/load_modules.sh
source /p/project1/cameo/schmidt36/set_lammps_paths.sh

export CUDA_VISIBLE_DEVICES=0
export MPICH_GPU_SUPPORT_ENABLED=0
export MPIR_CVAR_CH4_OFI_ENABLE_GPU=0
export PSP_CUDA=0
export TF_CPP_MIN_LOG_LEVEL=2
export CUDA_HOME=/p/software/juwelsbooster/stages/2025/software/CUDA/12
export XLA_FLAGS="--xla_gpu_autotune_level=0 --xla_gpu_cuda_data_dir=$CUDA_HOME"

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE=/p/project1/cameo/schmidt36
PY=${BASE}/test_env_newsetup/bin/python
SCRIPT=${BASE}/cameo_md/option3_fixed_ca_analysis.py
MD_RUNS=${BASE}/cameo_md/runs/forcevar_320K
CG_NPZ_DIR=${BASE}/cameo_cg/data_prep/datasets/4pro_320K_2500_notnorm/02_cg_npz
MODELS=${BASE}/cameo_md/models
LAMMPS_BIN=${BASE}/lammps/build/lmp

# ---------------------------------------------------------------------------
# Model .mlir files (symbolic reexports — used by LAMMPS chemtrain_deploy)
# ---------------------------------------------------------------------------

# Aggforce model 1 — no priors
MLIR_AGG1=${MODELS}/training_md_testing_cueq_fast_tiled_no_priors_symbolic_reexport.mlir

# Aggforce model 2 — priors cfg
MLIR_AGG2=${MODELS}/training_md_testing_cueq_fast_tiled_priorscfg_symbolic_reexport.mlir

# Non-aggforce model
MLIR_NOAGG=${MODELS}/noagg_test_cueq_fast_tiled_no_agg_symbolic_reexport.mlir

# ---------------------------------------------------------------------------
# Per-protein parameters
# ---------------------------------------------------------------------------
declare -A N_PROT=(
    [2gy5A01]=1613
    [4q5WA02]=1390
    [4zohB01]=868
)
declare -A CG_NPZ=(
    [2gy5A01]=${CG_NPZ_DIR}/2gy5A01_cg.npz
    [4q5WA02]=${CG_NPZ_DIR}/4q5wA02_cg.npz
    [4zohB01]=${CG_NPZ_DIR}/4zohB01_cg.npz
)

# ---------------------------------------------------------------------------
# Run — two passes: agg1+noagg, then agg2+noagg
# ---------------------------------------------------------------------------
for PASS in agg1 agg2; do
    if [[ "$PASS" == "agg1" ]]; then
        MLIR_AGG=${MLIR_AGG1}
        OUT_DIR=${MD_RUNS}/option3_results_agg1
    else
        MLIR_AGG=${MLIR_AGG2}
        OUT_DIR=${MD_RUNS}/option3_results_agg2
    fi

    mkdir -p ${OUT_DIR}

    for PROT in 2gy5A01 4q5WA02 4zohB01; do
        echo "========================================================"
        echo "PASS=${PASS}  Protein=${PROT}"
        echo "========================================================"

        DUMP=${MD_RUNS}/${PROT}/protein_forces.dump

        if [[ ! -f "${DUMP}" ]]; then
            echo "  WARNING: dump not found: ${DUMP} — skipping"
            continue
        fi
        if [[ ! -f "${CG_NPZ[$PROT]}" ]]; then
            echo "  WARNING: CG NPZ not found: ${CG_NPZ[$PROT]} — skipping"
            continue
        fi

        srun ${PY} ${SCRIPT} \
            --protein    ${PROT} \
            --dump       ${DUMP} \
            --cg-npz     ${CG_NPZ[$PROT]} \
            --n-protein  ${N_PROT[$PROT]} \
            --mlir-agg   ${MLIR_AGG} \
            --mlir-noagg ${MLIR_NOAGG} \
            --lammps-bin ${LAMMPS_BIN} \
            --out-dir    ${OUT_DIR}

        echo "  Done: ${PROT}"
    done
done

echo "========================================================"
echo "Option 3 complete."
echo "  agg1 results : ${MD_RUNS}/option3_results_agg1/"
echo "  agg2 results : ${MD_RUNS}/option3_results_agg2/"
echo "========================================================"
