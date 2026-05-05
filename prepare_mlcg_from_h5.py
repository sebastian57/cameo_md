#!/usr/bin/env python3
"""Prepare MLCG runs directly from an mdCATH h5 trajectory — no classical CHARMM22 needed.

Extracts N random CA-coordinate frames from the h5 file at a chosen temperature
group, writes a LAMMPS atomic data file for each, patches the MLCG input template,
and produces the manifest files that run_experiment_mlcg_array.sh expects.

Each starting frame becomes one MLCG run.  The run does its own CG-space
equilibration (the eq_steps in the LAMMPS template) before collecting TICA frames.

Outputs in --mlcg-out-dir:
    mlcg_ready_manifest.tsv     5-col, consumed by run_experiment_mlcg_array.sh
    mlcg_manifest.tsv           9-col, consumed by tica_from_h5.py experiment mode
    mlcg_prepare_summary.json
    h5/<protein>/<run_id>/
        seed_h5.data            LAMMPS atomic data file (CA positions + types)
        in.mlcg.lmp             Patched LAMMPS input
        run_meta.json

Usage (one call per model variant):
    python prepare_mlcg_from_h5.py \\
        --h5          structures/mdcath_dataset_4zohB01.h5 \\
        --protein     4zohB01 \\
        --species-npz /path/to/4zohB01_cg.npz \\
        --trained-model   /path/to/model.mlir \\
        --input-template  inp_lammps_mlcg_1pro_ml_only.in \\
        --mlcg-out-dir    outputs/exp_1pro_tica/mlcg_ml_only \\
        [--temp-group 320] [--run-idx 0] [--n-seeds 3] [--seed 42] \\
        [--run-id-suffix __ml_only] [--box-pad 15.0] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import h5py
import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def extract_ca_indices(pdb_protein_atoms: str) -> list[int]:
    indices: list[int] = []
    atom_idx = 0
    for line in pdb_protein_atoms.splitlines():
        if line.startswith("ATOM"):
            if line[12:16].strip() == "CA":
                indices.append(atom_idx)
            atom_idx += 1
    return indices


def parse_species_types(npz_path: Path) -> list[int]:
    """Return 1-indexed LAMMPS atom types from species npz."""
    with np.load(npz_path, allow_pickle=True) as data:
        if "species" not in data:
            raise ValueError(f"NPZ missing 'species' array: {npz_path}")
        species = np.asarray(data["species"])
    if species.ndim == 2:
        species = species[0]
    species = np.asarray(species, dtype=int)
    if species.min() < 0:
        raise ValueError(f"Negative species indices in {npz_path}")
    return [int(v) + 1 for v in species.tolist()]


def write_data_file(out: Path, atom_types: list[int], xyz: np.ndarray, box_pad: float) -> None:
    """Write a LAMMPS atomic-style data file for a CA-bead system.

    atom_types: list of 1-indexed LAMMPS types, len == n_atoms
    xyz:        (n_atoms, 3) CA positions in Angstrom
    """
    xlo, xhi = float(xyz[:, 0].min()) - box_pad, float(xyz[:, 0].max()) + box_pad
    ylo, yhi = float(xyz[:, 1].min()) - box_pad, float(xyz[:, 1].max()) + box_pad
    zlo, zhi = float(xyz[:, 2].min()) - box_pad, float(xyz[:, 2].max()) + box_pad
    n_atoms = len(atom_types)
    n_types = max(atom_types)

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as fh:
        fh.write("LAMMPS data file from mdCATH h5 CA snapshot\n\n")
        fh.write(f"{n_atoms} atoms\n")
        fh.write(f"{n_types} atom types\n\n")
        fh.write(f"{xlo:.6f} {xhi:.6f} xlo xhi\n")
        fh.write(f"{ylo:.6f} {yhi:.6f} ylo yhi\n")
        fh.write(f"{zlo:.6f} {zhi:.6f} zlo zhi\n\n")
        fh.write("Masses\n\n")
        for t in range(1, n_types + 1):
            fh.write(f"  {t} 12.01\n")
        fh.write("\nAtoms\n\n")
        for i, (t, (x, y, z)) in enumerate(zip(atom_types, xyz.tolist()), start=1):
            fh.write(f"{i} {t} {x:.8f} {y:.8f} {z:.8f}\n")


def patch_input(template_text: str, *, data_file: Path, model_file: Path, dump_dir: Path) -> str:
    def set_string_var(src: str, key: str, value: str) -> str:
        repl = f'variable {key:<14} string "{value}"'
        pat = re.compile(rf"^\s*variable\s+{re.escape(key)}\s+string\s+.*$", re.MULTILINE)
        if pat.search(src):
            return pat.sub(repl, src)
        return repl + "\n" + src

    text = set_string_var(template_text, "data_file", str(data_file))
    text = set_string_var(text, "model_file", str(model_file))
    text = set_string_var(text, "dump_dir", str(dump_dir))
    return text


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--h5",            type=Path, required=True)
    p.add_argument("--protein",       type=str,  required=True)
    p.add_argument("--species-npz",   type=Path, required=True,
                   help="CG npz file with 'species' array for the protein")
    p.add_argument("--trained-model", type=Path, required=True,
                   help="MLIR model file to use for this variant")
    p.add_argument("--input-template", type=Path, required=True,
                   help="LAMMPS input template (e.g. inp_lammps_mlcg_1pro_ml_only.in)")
    p.add_argument("--mlcg-out-dir",  type=Path, required=True,
                   help="Output directory for this variant's manifests and run dirs")
    p.add_argument("--temp-group",    type=str,  default="320",
                   help="Temperature group to draw frames from (default: 320)")
    p.add_argument("--run-idx",       type=int,  default=0,
                   help="h5 run index within the temperature group (default: 0)")
    p.add_argument("--n-seeds",       type=int,  default=1,
                   help="Number of random frames to extract as starting seeds")
    p.add_argument("--seed",          type=int,  default=42,
                   help="RNG seed for frame selection")
    p.add_argument("--run-id-suffix", type=str,  default="",
                   help="Suffix appended to run IDs (e.g. '__ml_only')")
    p.add_argument("--box-pad",       type=float, default=15.0,
                   help="Box padding around CA extent in Angstrom (default: 15.0)")
    p.add_argument("--dry-run",       action="store_true")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # Validate inputs
    for attr, path in [("h5", args.h5), ("species_npz", args.species_npz),
                       ("trained_model", args.trained_model),
                       ("input_template", args.input_template)]:
        if not path.exists() and not args.dry_run:
            raise FileNotFoundError(f"--{attr.replace('_','-')} not found: {path}")

    mlcg_out = args.mlcg_out_dir.resolve()
    model_file = args.trained_model.resolve()
    template_text = args.input_template.read_text()

    # -----------------------------------------------------------------------
    # Load h5: CA indices, frame count, coordinates
    # -----------------------------------------------------------------------
    with h5py.File(args.h5, "r") as hf:
        prot = hf[args.protein] if args.protein in hf else hf[next(iter(hf.keys()))]
        pdb_str: str = prot["pdbProteinAtoms"][()].decode()
        ca_idx = extract_ca_indices(pdb_str)
        if not ca_idx:
            raise ValueError(f"No CA atoms found in pdbProteinAtoms for {args.protein}")

        tg = prot[args.temp_group]
        run_key = str(args.run_idx)
        if run_key not in tg:
            available = sorted(tg.keys(), key=int)
            raise ValueError(f"run-idx {args.run_idx} not in h5 temp_group {args.temp_group}; "
                             f"available: {available}")
        coords_all = tg[run_key]["coords"][()]  # (n_frames, n_atoms, 3)
        n_frames_total = coords_all.shape[0]
        ca_coords_all = coords_all[:, ca_idx, :].astype(float)  # (n_frames, n_ca, 3)

    n_ca = len(ca_idx)
    print(f"[h5-prep] protein={args.protein}  CA atoms={n_ca}  "
          f"temp_group={args.temp_group}  run={args.run_idx}  "
          f"total frames={n_frames_total}")

    # -----------------------------------------------------------------------
    # Select starting frames
    # -----------------------------------------------------------------------
    rng = np.random.default_rng(args.seed)
    n_seeds = min(args.n_seeds, n_frames_total)
    frame_indices = sorted(rng.choice(n_frames_total, size=n_seeds, replace=False).tolist())
    print(f"[h5-prep] selected {n_seeds} frame(s): {frame_indices}")

    # -----------------------------------------------------------------------
    # Species types
    # -----------------------------------------------------------------------
    species_types = parse_species_types(args.species_npz)
    if len(species_types) != n_ca:
        raise ValueError(f"Species npz has {len(species_types)} entries but protein has {n_ca} CA atoms")

    # -----------------------------------------------------------------------
    # Build runs
    # -----------------------------------------------------------------------
    mlcg_out.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []
    ready_rows: list[dict] = []

    for fi, frame_idx in enumerate(frame_indices):
        source_run_id = f"{args.protein}__h5_{args.temp_group}_{args.run_idx}__f{frame_idx:05d}"
        run_id = f"{source_run_id}__mlcg_h5{args.run_id_suffix}"
        run_dir = mlcg_out / "h5" / args.protein / source_run_id

        xyz = ca_coords_all[frame_idx]  # (n_ca, 3)
        data_out = run_dir / "seed_h5.data"
        input_out = run_dir / "in.mlcg.lmp"

        if not args.dry_run:
            write_data_file(data_out, species_types, xyz, args.box_pad)
            patched = patch_input(template_text,
                                  data_file=data_out,
                                  model_file=model_file,
                                  dump_dir=run_dir)
            run_dir.mkdir(parents=True, exist_ok=True)
            input_out.write_text(patched)

            meta = {
                "run_id": run_id,
                "protein": args.protein,
                "seed_mode": "h5",
                "source_run_id": source_run_id,
                "source_h5": str(args.h5.resolve()),
                "temp_group": args.temp_group,
                "h5_run_idx": args.run_idx,
                "frame_idx": int(frame_idx),
                "n_ca": n_ca,
                "cg_data": str(data_out),
                "input_file": str(input_out),
                "model_file": str(model_file),
                "species_npz": str(args.species_npz.resolve()),
            }
            (run_dir / "run_meta.json").write_text(json.dumps(meta, indent=2))

        row = {
            "run_id": run_id,
            "protein": args.protein,
            "seed_mode": "h5",
            "source_run_id": source_run_id,
            "source_run_dir": str(run_dir.parent),
            "run_dir": str(run_dir),
            "input_file": str(input_out),
            "status": "ready" if not args.dry_run else "dry-run",
            "reason": "",
        }
        all_rows.append(row)
        if not args.dry_run:
            ready_rows.append(row)

    # -----------------------------------------------------------------------
    # Write manifests
    # -----------------------------------------------------------------------
    all_manifest  = mlcg_out / "mlcg_manifest.tsv"
    ready_manifest = mlcg_out / "mlcg_ready_manifest.tsv"

    if not args.dry_run:
        with all_manifest.open("w") as f:
            for r in all_rows:
                f.write("\t".join([
                    r["run_id"], r["protein"], r["seed_mode"],
                    r["source_run_id"], r["source_run_dir"],
                    r["run_dir"], r["input_file"],
                    r["status"], r["reason"],
                ]) + "\n")

        with ready_manifest.open("w") as f:
            for r in ready_rows:
                f.write("\t".join([
                    r["run_id"], r["protein"], r["seed_mode"],
                    r["run_dir"], r["input_file"],
                ]) + "\n")

    summary = {
        "protein": args.protein,
        "h5": str(args.h5.resolve()),
        "temp_group": args.temp_group,
        "h5_run_idx": args.run_idx,
        "frame_indices": frame_indices,
        "n_seeds": n_seeds,
        "n_ca": n_ca,
        "mlcg_out_dir": str(mlcg_out),
        "trained_model": str(model_file),
        "input_template": str(args.input_template.resolve()),
        "run_id_suffix": args.run_id_suffix,
        "counts": {"ready": len(ready_rows), "total": len(all_rows)},
        "manifests": {
            "all": str(all_manifest),
            "ready": str(ready_manifest),
        },
        "dry_run": args.dry_run,
    }

    if not args.dry_run:
        (mlcg_out / "mlcg_prepare_summary.json").write_text(json.dumps(summary, indent=2))

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
