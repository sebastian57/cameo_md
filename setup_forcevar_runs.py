#!/usr/bin/env python3
"""Set up fixed-CA force-variance LAMMPS runs for selected mdCATH proteins.

For each protein:
  1. Pick a random frame from the 320 K trajectory.
  2. Inject those protein-atom coordinates into the solvated PDB from the h5 file.
  3. Write protein.pdb + protein.psf to a new run directory.
  4. Copy shared force-field files and run charmm2lammps.pl.
  5. Extract CA atom IDs from the PSF and write ca_group.lmp / ca_atom_ids.txt.
  6. Write a protein-specific LAMMPS input and SLURM submission script.

Usage (after loading modules):
    source /p/project1/cameo/schmidt36/load_modules.sh
    python setup_forcevar_runs.py [--seed 42] [--temp 320]
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

import h5py
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path("/p/project1/cameo/schmidt36")
H5_DIR = ROOT / "cameo_md/structures"
RUNS_BASE = ROOT / "cameo_md/runs/forcevar_320K"
FF_SRC = ROOT / "cameo_md/runs/4q5wA02_charmm22_cmap_fixed"  # force-field files live here
LAMMPS_BIN = ROOT / "lammps/build/lmp"
C2L_PL = ROOT / "lammps/tools/ch2lmp/charmm2lammps.pl"
PY_ENV = ROOT / "clean_booster_env/bin/python"
ANALYSIS_PY = ROOT / "cameo_md/compute_aggforce_variance.py"

FF_FILES = [
    "top_all22_prot_mdcath.rtf",
    "par_all22_prot_mdcath.prm",
    "charmm22.cmap",
]

# Requested set from user
PROTEINS = ["2gy5A01", "4q5WA02", "4zohB01", "5k39B02"]

# MD parameters
TEMP_K = 320
EQ_STEPS = 50_000  # 50 ps equilibration
SAMPLE_STEPS = 20_000  # 20 ps sampling
SAMPLE_STRIDE = 10  # dump every 10 steps -> 2 000 frames


def _available_protein_ids() -> list[str]:
    ids: list[str] = []
    for h5 in sorted(H5_DIR.glob("mdcath_dataset_*.h5")):
        name = h5.stem.replace("mdcath_dataset_", "", 1)
        if name:
            ids.append(name)
    return ids


def _resolve_protein_id(requested_id: str) -> str:
    """Resolve requested protein id against available H5 IDs (case-insensitive)."""
    avail = _available_protein_ids()
    lookup = {pid.lower(): pid for pid in avail}
    key = requested_id.lower()
    if key not in lookup:
        raise FileNotFoundError(
            f"No H5 dataset found for '{requested_id}'. Available: {', '.join(avail)}"
        )
    return lookup[key]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seed", type=int, default=42, help="NumPy random seed for frame selection (default 42)")
    p.add_argument("--temp-group", default="320", help="Temperature group to sample from (default '320')")
    p.add_argument("--run-group", default="0", help="Run sub-group within the temperature group (default '0')")
    p.add_argument("--dry-run", action="store_true", help="Write files but do not call charmm2lammps.pl")
    return p.parse_args()


# ---------------------------------------------------------------------------
# PDB helpers
# ---------------------------------------------------------------------------

def replace_protein_coords(pdb_str: str, new_coords: np.ndarray) -> str:
    """Return the solvated PDB string with protein (P0) atom coords replaced."""
    out_lines = []
    prot_idx = 0
    for line in pdb_str.splitlines():
        if line.startswith(("ATOM", "HETATM")):
            seg = line[72:76].strip()  # segment ID in cols 73-76
            if seg == "P0":
                x, y, z = new_coords[prot_idx]
                line = line[:30] + f"{x:8.3f}{y:8.3f}{z:8.3f}" + line[54:]
                prot_idx += 1
        out_lines.append(line)
    return "\n".join(out_lines) + "\n"


def detect_water_angle_type(data_protein_path: Path) -> int:
    """Return LAMMPS angle type number for TIP3P H-O-H angle (HT OT HT)."""
    try:
        in_coeffs = False
        with data_protein_path.open() as fh:
            for line in fh:
                if line.strip() == "Angle Coeffs":
                    in_coeffs = True
                    continue
                if in_coeffs:
                    if not line.strip() or line.strip().startswith("#"):
                        continue
                    if line.strip() and not line.strip()[0].isdigit():
                        break
                    if "HT" in line and "OT" in line:
                        return int(line.split()[0])
    except Exception:
        pass
    return 110


def extract_ca_ids_from_pdb(pdb_protein_atoms_str: str) -> list[int]:
    """Return 1-indexed LAMMPS CA atom IDs from pdbProteinAtoms."""
    ids = []
    atom_idx = 0
    for line in pdb_protein_atoms_str.splitlines():
        if line.startswith("ATOM"):
            atom_name = line[12:16].strip()
            if atom_name == "CA":
                ids.append(atom_idx + 1)
            atom_idx += 1
    return ids


# ---------------------------------------------------------------------------
# LAMMPS input generator
# ---------------------------------------------------------------------------

def write_lammps_input(path: Path, protein_id: str, n_prot: int, water_angle_type: int = 110) -> None:
    content = f"""\
# Fixed-CA force-variance run — {protein_id}
# Phase 1: equilibration with CA positions frozen
# Phase 2: force sampling at fixed CA configuration

variable temp          equal {TEMP_K}.0
variable dt_fs         equal 1.0
variable eq_steps      equal {EQ_STEPS}
variable sample_steps  equal {SAMPLE_STEPS}
variable sample_stride equal {SAMPLE_STRIDE}

units           real
atom_style      full
boundary        p p p

bond_style      harmonic
angle_style     charmm
dihedral_style  charmm
improper_style  harmonic

pair_style      lj/charmm/coul/long 8.0 12.0
pair_modify     mix arithmetic
kspace_style    pppm 1.0e-6
special_bonds   charmm

fix             cmap all cmap charmm22.cmap
fix_modify      cmap energy yes
read_data       data.protein fix cmap crossterm CMAP

include         ca_group.lmp
# Protein-segment atoms only (IDs 1..{n_prot}); excludes water and ions
group           protein id 1:{n_prot}

neighbor        2.0 bin
neigh_modify    every 1 delay 0 check yes

timestep        ${{dt_fs}}

# --- CG minimization with CA fixed ---
fix             freeze_min ca setforce 0.0 0.0 0.0
min_style       cg
minimize        1.0e-4 1.0e-6 200 2000
unfix           freeze_min

# --- NVT-Langevin, mobile atoms only ---
velocity        all create ${{temp}} 12345 mom yes rot yes dist gaussian
velocity        ca set 0.0 0.0 0.0

compute         tmobile mobile temp

fix             int  mobile nve
fix             bath mobile langevin ${{temp}} ${{temp}} 100.0 24680 zero yes
fix_modify      bath temp tmobile
fix             sh   all shake 1.0e-6 500 0 m 1.0 a {water_angle_type}

thermo          5000
thermo_style    custom step temp c_tmobile press pe ke etotal ebond eangle edihed eimp evdwl ecoul f_cmap

# --- Equilibration ---
run             ${{eq_steps}}
write_restart   restart.eq_fixed_ca

# --- Sampling ---
reset_timestep  0

# CA forces (lightweight, for quick check)
dump            dca   ca      custom ${{sample_stride}} ca_forces.dump      id type x y z fx fy fz
dump_modify     dca   sort id format line "%d %d %.8f %.8f %.8f %.10f %.10f %.10f"

# All protein-atom positions + forces (required for aggforce projection)
dump            dprot protein custom ${{sample_stride}} protein_forces.dump id type x y z fx fy fz
dump_modify     dprot sort id format line "%d %d %.8f %.8f %.8f %.10f %.10f %.10f"

run             ${{sample_steps}}

write_restart   restart.forcevar_fixed_ca
"""
    path.write_text(content)


# ---------------------------------------------------------------------------
# SLURM script generator
# ---------------------------------------------------------------------------

def write_slurm_script_with_nprot(path: Path, protein_id: str, run_dir: Path, n_prot: int) -> None:
    content = f"""\
#!/bin/bash -x
#SBATCH --account=cameo
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=18:00:00
#SBATCH --partition=booster
#SBATCH --gres=gpu:1
#SBATCH --job-name=fv_{protein_id}
#SBATCH --output={run_dir}/slurm-%j.out

source {ROOT}/load_modules.sh
export PATH={ROOT}/lammps/build:$PATH
export MPICH_GPU_SUPPORT_ENABLED=0
export MPIR_CVAR_CH4_OFI_ENABLE_GPU=0
export PSP_CUDA=0
unset LAMMPS_PLUGIN_PATH

cd {run_dir}

srun {LAMMPS_BIN} -in in.forcevar.lmp -log forcevar.log

{PY_ENV} {ANALYSIS_PY} \\
    --dump protein_forces.dump \\
    --ca-ids ca_atom_ids.txt \\
    --n-protein {n_prot} \\
    --constraint-mode data_prep \\
    --summary-out aggforce_ca_variance_summary.json \\
    --out aggforce_ca_variance.csv \\
    --block-out aggforce_ca_variance_blocks.csv
"""
    path.write_text(content)
    path.chmod(0o755)


# ---------------------------------------------------------------------------
# Per-protein setup
# ---------------------------------------------------------------------------

def setup_protein(requested_id: str, rng: np.random.Generator,
                  temp_group: str, run_group: str, dry_run: bool) -> str | None:
    print(f"\n{'='*60}")
    print(f"Setting up {requested_id}")
    print(f"{'='*60}")

    try:
        protein_id = _resolve_protein_id(requested_id)
    except FileNotFoundError as exc:
        print(f"SKIP {requested_id}: {exc}")
        return None

    h5_path = H5_DIR / f"mdcath_dataset_{protein_id}.h5"
    run_dir = RUNS_BASE / requested_id
    run_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(h5_path, "r") as f:
        if protein_id in f:
            prot = f[protein_id]
        else:
            first_key = next(iter(f.keys()))
            prot = f[first_key]
            print(f"  WARNING: expected group {protein_id}, using {first_key}")

        pdb_str = prot["pdb"][()].decode()
        psf_str = prot["psf"][()].decode()
        pdb_prot_atoms_str = prot["pdbProteinAtoms"][()].decode()
        coords = prot[temp_group][run_group]["coords"][()]

    n_frames, n_prot, _ = coords.shape
    frame_idx = int(rng.integers(0, n_frames))
    print(f"  Frames available : {n_frames}  -> selected frame {frame_idx}")
    print(f"  Protein atoms    : {n_prot}")

    new_coords = coords[frame_idx].astype(float)
    updated_pdb = replace_protein_coords(pdb_str, new_coords)

    (run_dir / "protein.pdb").write_text(updated_pdb)
    (run_dir / "protein.psf").write_text(psf_str)
    print(f"  Written protein.pdb ({n_prot} protein + water atoms)")

    for ff_file in FF_FILES:
        shutil.copy(FF_SRC / ff_file, run_dir / ff_file)
    print(f"  Copied FF files: {FF_FILES}")

    if not dry_run:
        perl = shutil.which("perl") or "perl"
        cmd = [perl, str(C2L_PL), "-cmap=22", "all22_prot_mdcath", "protein"]
        print("  Running charmm2lammps.pl ...")
        result = subprocess.run(cmd, cwd=run_dir, capture_output=True, text=True)
        (run_dir / "conversion.log").write_text(result.stdout + result.stderr)
        if result.returncode != 0:
            print(f"  ERROR in charmm2lammps.pl (see {run_dir}/conversion.log):")
            print(result.stderr[-500:])
            return None
        src = run_dir / "protein.data"
        dst = run_dir / "data.protein"
        if src.exists():
            shutil.copy(src, dst)
            print("  Conversion OK -> data.protein")
        else:
            print("  WARNING: protein.data not found after conversion")
    else:
        print("  [dry-run] skipping charmm2lammps.pl")

    ca_ids = extract_ca_ids_from_pdb(pdb_prot_atoms_str)
    if not ca_ids:
        print("  ERROR: no CA atoms found in pdbProteinAtoms")
        return None
    print(f"  CA atoms: {len(ca_ids)}  (IDs {ca_ids[0]}..{ca_ids[-1]})")

    with (run_dir / "ca_atom_ids.txt").open("w") as g:
        for ca_id in ca_ids:
            g.write(f"{ca_id}\n")

    id_list = " ".join(str(i) for i in ca_ids)
    ca_group_content = (
        f"# Auto-generated for {requested_id}\n"
        f"group ca id {id_list}\n"
        f"group mobile subtract all ca\n"
    )
    (run_dir / "ca_group.lmp").write_text(ca_group_content)

    water_angle_type = detect_water_angle_type(run_dir / "data.protein")
    print(f"  Water (HOH) angle type : {water_angle_type}")

    write_lammps_input(run_dir / "in.forcevar.lmp", requested_id, n_prot, water_angle_type)
    print(f"  Written in.forcevar.lmp  (T={TEMP_K}K, eq={EQ_STEPS}, sample={SAMPLE_STEPS}, stride={SAMPLE_STRIDE})")

    write_slurm_script_with_nprot(run_dir / "submit.sh", requested_id, run_dir, n_prot)
    print("  Written submit.sh")

    return requested_id


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    RUNS_BASE.mkdir(parents=True, exist_ok=True)

    submitted_ids: list[str] = []
    for requested_id in PROTEINS:
        out_id = setup_protein(
            requested_id,
            rng,
            temp_group=args.temp_group,
            run_group=args.run_group,
            dry_run=args.dry_run,
        )
        if out_id is not None:
            submitted_ids.append(out_id)

    submit_all = RUNS_BASE / "submit_all.sh"
    lines = ["#!/bin/bash", f"# Submit all {len(submitted_ids)} force-variance jobs", ""]
    for pid in submitted_ids:
        lines.append(f"sbatch {RUNS_BASE}/{pid}/submit.sh")
    lines.append("")
    submit_all.write_text("\n".join(lines))
    submit_all.chmod(0o755)

    print(f"\nWrote {submit_all}")
    print("Run with:  bash cameo_md/runs/forcevar_320K/submit_all.sh")


if __name__ == "__main__":
    main()
