#!/bin/bash -x
#SBATCH --account=cameo
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --time=06:00:00
#SBATCH --partition=booster
#SBATCH --job-name=charmm22_tica
#SBATCH --output=/p/project1/cameo/schmidt36/cameo_md/slurm-charmm22_tica-%j.out

# Classical CHARMM22+CMAP MD for TICA comparison.
# Uses 8 MPI tasks for ~4x speedup vs single-task.
# No GPU needed — PPPM runs on CPU.

set -euo pipefail

BASE=/p/project1/cameo/schmidt36
INPUT=${BASE}/cameo_md/inp_lammps_charmm22_tica.in
LAMMPS_BIN=${BASE}/lammps/build/lmp
OUT_DIR=${BASE}/cameo_md/outputs/charmm22_tica

source ${BASE}/load_modules.sh

export MPICH_GPU_SUPPORT_ENABLED=0
export MPIR_CVAR_CH4_OFI_ENABLE_GPU=0
export PSP_CUDA=0

mkdir -p ${OUT_DIR}

echo "Input   : ${INPUT}"
echo "Output  : ${OUT_DIR}"
echo "MPI tasks: ${SLURM_NTASKS}"

srun ${LAMMPS_BIN} \
    -in  ${INPUT} \
    -log ${OUT_DIR}/lammps.log
