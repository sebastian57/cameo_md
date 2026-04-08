#!/bin/bash -x

#SBATCH --account=cameo
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=04:00:00
#SBATCH --partition=booster
#SBATCH --gres=gpu:1
#SBATCH --job-name=mdcath_ca_forcevar
#SBATCH --output=slurm-%j.out

source /p/project1/cameo/schmidt36/load_modules.sh
source /p/project1/cameo/schmidt36/set_lammps_paths.sh

PROJECT_ROOT=/p/project1/cameo/schmidt36/cameo_md/runs/4q5wA02_charmm22_cmap_fixed
INPUT_FILE="$PROJECT_ROOT/in.ca_force_variance.lmp"
cd "$PROJECT_ROOT"

export MPICH_GPU_SUPPORT_ENABLED=0
export MPIR_CVAR_CH4_OFI_ENABLE_GPU=0
export PSP_CUDA=0
unset LAMMPS_PLUGIN_PATH

srun /p/project1/cameo/schmidt36/lammps/build/lmp -in "$INPUT_FILE" -log forcevar.log
