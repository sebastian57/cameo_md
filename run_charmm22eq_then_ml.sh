#!/bin/bash -x
#SBATCH --account=cameo
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=02:00:00
#SBATCH --partition=booster
#SBATCH --gres=gpu:1
#SBATCH --job-name=ch22eq_ml
#SBATCH --output=/p/project1/cameo/schmidt36/cameo_md/slurm-ch22eq-ml-%j.out

# Run CHARMM22+CMAP equilibration, build CA-only ML start data,
# and optionally submit ML run.
#
# Usage:
#   bash cameo_md/run_charmm22eq_then_ml.sh
#   bash cameo_md/run_charmm22eq_then_ml.sh --submit-ml
#   sbatch cameo_md/run_charmm22eq_then_ml.sh

set -euo pipefail

BASE=/p/project1/cameo/schmidt36
MD_DIR=$BASE/cameo_md
LAMMPS_BIN=$BASE/lammps/build/lmp
PY=$BASE/test_env_newsetup/bin/python

CHARMM_IN=$MD_DIR/inp_lammps_charmm22_eq_for_ml.in
CHARMM_OUT=$MD_DIR/outputs/charmm22_eq_for_ml
EQ_LOG=$CHARMM_OUT/lammps.log

EQ_DUMP=$CHARMM_OUT/ca_eq_final.dump
TEMPLATE=$MD_DIR/structures/config_44.data
ML_DATA=$MD_DIR/structures/config_44_from_charmm22eq.data
CONVERTER=$MD_DIR/build_ml_data_from_charmm_eq_dump.py

ML_INPUT=$MD_DIR/inp_lammps_trained_forcevarlike_charmm22eq.in
SUBMIT_ONE=$MD_DIR/submit_one_lammps_test.sh

SUBMIT_ML=0
if [ "${1:-}" = "--submit-ml" ]; then
  SUBMIT_ML=1
fi

source $BASE/load_modules.sh
source $BASE/set_lammps_paths.sh

# Important: avoid GPU-aware MPI CUDA init in CHARMM22 (CPU pair styles)
export MPICH_GPU_SUPPORT_ENABLED=0
export MPIR_CVAR_CH4_OFI_ENABLE_GPU=0
export PSP_CUDA=0

mkdir -p $CHARMM_OUT

echo "[1/3] Running CHARMM22+CMAP equilibration"
$LAMMPS_BIN -in $CHARMM_IN -log $EQ_LOG

echo "[2/3] Converting equilibrated CA dump to ML data file"
$PY $CONVERTER --dump $EQ_DUMP --template $TEMPLATE --out $ML_DATA

if [ "$SUBMIT_ML" -eq 1 ]; then
  echo "[3/3] Submitting ML 200k run via submit_one_lammps_test.sh"
  sbatch $SUBMIT_ONE $ML_INPUT
else
  echo "[3/3] Skipping ML submission (pass --submit-ml to submit)"
  echo "Submit manually with:"
  echo "  sbatch $SUBMIT_ONE $ML_INPUT"
fi
