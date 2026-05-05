#!/bin/bash -x
#SBATCH --account=cameo
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=03:00:00
#SBATCH --partition=booster
#SBATCH --gres=gpu:1
#SBATCH --job-name=cmp_ch22_mlcg
#SBATCH --output=/p/project1/cameo/schmidt36/cameo_md/slurm-cmp-ch22-mlcg-%j.out

# Chained run:
# 1) CHARMM22+CMAP run (writes CA equilibration snapshot + production CA trajectory)
# 2) Convert CA equilibration snapshot into ML-CG LAMMPS data file
# 3) ML-CG run seeded from that converted structure

set -euo pipefail

BASE=/p/project1/cameo/schmidt36
MD_DIR=$BASE/cameo_md
LAMMPS_BIN=$BASE/lammps/build/lmp
PY=$BASE/test_env_newsetup/bin/python

CHARMM_IN=$MD_DIR/inp_lammps_charmm22.in
ML_IN=$MD_DIR/inp_lammps_mlcg.in
CONVERTER=$MD_DIR/build_ml_data_from_charmm_eq_dump.py

CHARMM_OUT=$MD_DIR/outputs/compare_clean/charmm22
ML_OUT=$MD_DIR/outputs/compare_clean/mlcg
CHARMM_LOG=$CHARMM_OUT/lammps.log
ML_LOG=$ML_OUT/lammps.log

EQ_DUMP=$CHARMM_OUT/ca_eq_final.dump
TEMPLATE=$MD_DIR/structures/config_44.data
ML_DATA=$MD_DIR/structures/config_44_from_charmm22eq_compare.data

source $BASE/load_modules.sh
source $BASE/set_lammps_paths.sh

# Avoid GPU-aware MPI CUDA init in CHARMM22 CPU force field path
export MPICH_GPU_SUPPORT_ENABLED=0
export MPIR_CVAR_CH4_OFI_ENABLE_GPU=0
export PSP_CUDA=0

# Keep ML/XLA runtime deterministic and quiet
export TF_CPP_MIN_LOG_LEVEL=2
export CUDA_HOME=/p/software/juwelsbooster/stages/2025/software/CUDA/12
export XLA_FLAGS="--xla_gpu_autotune_level=0 --xla_gpu_cuda_data_dir=$CUDA_HOME"

mkdir -p $CHARMM_OUT
mkdir -p $ML_OUT

echo "[1/3] Running CHARMM22+CMAP comparable run"
$LAMMPS_BIN -in $CHARMM_IN -log $CHARMM_LOG

echo "[2/3] Building ML-CG seed data from CHARMM equilibration snapshot"
$PY $CONVERTER --dump $EQ_DUMP --template $TEMPLATE --out $ML_DATA

echo "[3/3] Running ML-CG comparable run"
$LAMMPS_BIN -in $ML_IN -log $ML_LOG

echo "Completed chained CHARMM22 -> ML-CG workflow"
echo "CHARMM outputs: $CHARMM_OUT"
echo "ML outputs:     $ML_OUT"
