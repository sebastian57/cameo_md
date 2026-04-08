#!/usr/bin/env python3
"""Compute per-CA aggforce-projected force variance from a LAMMPS protein-atom dump.

Workflow:
  1. Read a LAMMPS custom dump of all protein atoms (IDs 1..N_prot) that includes
     fx, fy, fz columns.
  2. Apply aggforce force projection onto the CA beads.
  3. Compute per-CA variance across the trajectory and write CSV output.

Notes on metric comparison:
  - `RMS per-comp` is sqrt(mean(var_fx,var_fy,var_fz)) and is the apples-to-apples
    quantity to compare against component-wise force RMSE values from cameo_cg.
  - `RMS |F| noise` is sqrt(mean(var|F|^2)) and is larger by construction.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterator, List, Tuple

import numpy as np
from aggforce import LinearMap, guess_pairwise_constraints, project_forces


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dump", type=Path, required=True, help="LAMMPS custom dump with protein-atom positions and forces")
    p.add_argument("--ca-ids", type=Path, required=True, help="File with LAMMPS CA atom IDs (one per line)")
    p.add_argument("--out", type=Path, default=Path("aggforce_ca_variance.csv"), help="Per-CA variance output CSV")
    p.add_argument("--block-out", type=Path, default=Path("aggforce_ca_variance_blocks.csv"), help="Block-convergence summary CSV")
    p.add_argument("--summary-out", type=Path, default=Path("aggforce_ca_variance_summary.json"), help="Summary JSON with comparable RMS metrics")
    p.add_argument("--n-protein", type=int, required=True, help="Number of protein atoms (LAMMPS IDs 1..N)")
    p.add_argument("--blocks", type=int, default=4, help="Number of time blocks for convergence summary; default 4")
    p.add_argument("--max-frames", type=int, default=0, help="If >0, subsample this many frames uniformly from the dump")
    p.add_argument(
        "--constraint-mode",
        type=str,
        default="data_prep",
        choices=("data_prep", "auto", "none"),
        help="Constraint mode: data_prep (matches cg_1bead.py), auto, or none",
    )
    p.add_argument("--constraint-threshold", type=float, default=1.0e-3, help="Threshold for data_prep constraint guessing")
    p.add_argument("--constraint-frames", type=int, default=10, help="Frames used for data_prep constraint guessing")
    p.add_argument(
        "--allow-fallback-none",
        action="store_true",
        help="If projection fails, retry unconstrained mapping instead of failing",
    )
    return p.parse_args()


def parse_dump_frames(path: Path, n_atoms: int) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """Yield (R, F) per frame as float64 arrays of shape (n_atoms, 3).

    Expects columns: id type x y z fx fy fz, sorted by id (IDs 1..n_atoms).
    Skips truncated or malformed frames silently.
    """
    with path.open() as fh:
        while True:
            line = fh.readline()
            if not line:
                return
            if not line.startswith("ITEM: TIMESTEP"):
                continue

            if not fh.readline():
                return
            if not fh.readline().startswith("ITEM: NUMBER OF ATOMS"):
                continue
            try:
                n = int(fh.readline().strip())
            except ValueError:
                continue
            if n != n_atoms:
                for _ in range(n + 4):
                    fh.readline()
                continue

            if not fh.readline().startswith("ITEM: BOX BOUNDS"):
                continue
            fh.readline()
            fh.readline()
            fh.readline()

            if not fh.readline().startswith("ITEM: ATOMS"):
                continue

            R = np.empty((n_atoms, 3), dtype=np.float64)
            F = np.empty((n_atoms, 3), dtype=np.float64)
            ok = True
            for _ in range(n_atoms):
                raw = fh.readline()
                if not raw:
                    ok = False
                    break
                cols = raw.split()
                if len(cols) < 8:
                    ok = False
                    break
                try:
                    idx = int(cols[0]) - 1
                    R[idx, 0] = float(cols[2])
                    R[idx, 1] = float(cols[3])
                    R[idx, 2] = float(cols[4])
                    F[idx, 0] = float(cols[5])
                    F[idx, 1] = float(cols[6])
                    F[idx, 2] = float(cols[7])
                except (ValueError, IndexError):
                    ok = False
                    break
            if ok:
                yield R, F


def load_ca_ids(path: Path) -> List[int]:
    ids: List[int] = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            try:
                ids.append(int(parts[-1]))
            except ValueError:
                continue
    if not ids:
        raise ValueError(f"No CA atom IDs found in {path}")
    return ids


def ca_indices_in_protein(ca_lammps_ids: List[int]) -> List[int]:
    return [i - 1 for i in ca_lammps_ids]


def _constraints_from_mode(mode: str, coords: np.ndarray, threshold: float, n_frames: int):
    if mode == "auto":
        return "auto", "auto"
    if mode == "none":
        return set(), "none"

    # Match cameo_cg/data_prep/cg_1bead.py behavior
    n_probe = max(1, min(int(n_frames), int(coords.shape[0])))
    constraints = guess_pairwise_constraints(coords[:n_probe], threshold=float(threshold))
    return constraints, "data_prep"


def main() -> None:
    args = parse_args()

    if not args.dump.exists():
        sys.exit(f"ERROR: dump file not found: {args.dump}")
    if not args.ca_ids.exists():
        sys.exit(f"ERROR: CA IDs file not found: {args.ca_ids}")
    if args.blocks < 1:
        sys.exit("ERROR: --blocks must be >= 1")
    if args.n_protein < 1:
        sys.exit("ERROR: --n-protein must be >= 1")

    ca_lammps_ids = load_ca_ids(args.ca_ids)
    ca_idx = ca_indices_in_protein(ca_lammps_ids)
    n_ca = len(ca_idx)
    n_prot = int(args.n_protein)

    print(f"Protein atoms : {n_prot}")
    print(f"CA atoms      : {n_ca}")
    print(f"Dump file     : {args.dump}")

    R_list, F_list = [], []
    for R, F in parse_dump_frames(args.dump, n_prot):
        R_list.append(R)
        F_list.append(F)

    if not R_list:
        sys.exit("ERROR: no complete frames parsed from dump.")

    n_frames_total = len(R_list)
    if args.max_frames > 0 and n_frames_total > args.max_frames:
        indices = np.round(np.linspace(0, n_frames_total - 1, args.max_frames)).astype(int)
        R_list = [R_list[i] for i in indices]
        F_list = [F_list[i] for i in indices]
        print(f"Subsampled    : {n_frames_total} -> {len(R_list)} frames (--max-frames)")

    n_frames = len(R_list)
    coords = np.stack(R_list, axis=0)
    forces = np.stack(F_list, axis=0)
    del R_list, F_list
    print(f"Loaded frames : {n_frames}  ->  coords {coords.shape}, forces {forces.shape}")

    cmap = LinearMap([[i] for i in ca_idx], n_fg_sites=n_prot)
    constrained_inds, mode_used = _constraints_from_mode(
        args.constraint_mode,
        coords,
        threshold=args.constraint_threshold,
        n_frames=args.constraint_frames,
    )
    print(f"Running aggforce project_forces (constraint_mode={args.constraint_mode}, mode_used={mode_used})...")

    try:
        result = project_forces(
            coords=coords,
            forces=forces,
            coord_map=cmap,
            constrained_inds=constrained_inds,
        )
        constraint_info = result.get("constraints", None)
        n_constraints = len(constraint_info) if constraint_info is not None else 0
    except Exception as e:
        if not args.allow_fallback_none:
            raise
        print(f"  WARNING: projection failed ({e}); retrying with no constraints.")
        result = project_forces(
            coords=coords,
            forces=forces,
            coord_map=cmap,
            constrained_inds=set(),
        )
        mode_used = f"{mode_used}_fallback_none"
        constraint_info = result.get("constraints", None)
        n_constraints = len(constraint_info) if constraint_info is not None else 0

    mapped_forces = result["mapped_forces"]
    residual = float(result.get("residual", np.nan))
    print(f"  Detected constraints : {n_constraints}")
    print(f"  Projection residual  : {residual:.6e}  (mean squared mapped force)")

    var = np.var(mapped_forces, axis=0, ddof=1)
    var_fmag2 = var.sum(axis=1)

    with args.out.open("w") as g:
        print("id,n_samples,var_fx,var_fy,var_fz,var_fmag2", file=g)
        for k, lammps_id in enumerate(ca_lammps_ids):
            print(
                f"{lammps_id},{n_frames},"
                f"{var[k,0]:.12e},{var[k,1]:.12e},{var[k,2]:.12e},{var_fmag2[k]:.12e}",
                file=g,
            )
    print(f"Wrote per-CA CSV : {args.out}")

    block_rows = []
    for b in range(args.blocks):
        i0 = (b * n_frames) // args.blocks
        i1 = ((b + 1) * n_frames) // args.blocks
        blk = mapped_forces[i0:i1]
        blk_n = blk.shape[0]
        if blk_n < 2:
            continue
        blk_var = np.var(blk, axis=0, ddof=1)
        block_rows.append((
            b + 1,
            blk_n,
            blk_var[:, 0].mean(),
            blk_var[:, 1].mean(),
            blk_var[:, 2].mean(),
            blk_var.sum(axis=1).mean(),
        ))

    with args.block_out.open("w") as g:
        print("block,n_frames,n_atoms,mean_var_fx,mean_var_fy,mean_var_fz,mean_var_fmag2", file=g)
        for row in block_rows:
            print(
                f"{row[0]},{row[1]},{n_ca},"
                f"{row[2]:.12e},{row[3]:.12e},{row[4]:.12e},{row[5]:.12e}",
                file=g,
            )
    print(f"Wrote block CSV  : {args.block_out}")

    global_var = np.var(mapped_forces, axis=0, ddof=1)
    mfx = float(global_var[:, 0].mean())
    mfy = float(global_var[:, 1].mean())
    mfz = float(global_var[:, 2].mean())
    mfmag2 = float(global_var.sum(axis=1).mean())

    rms_per_comp = float(np.sqrt((mfx + mfy + mfz) / 3.0))
    rms_force_magnitude = float(np.sqrt(mfmag2))

    summary = {
        "n_frames": int(n_frames),
        "n_ca": int(n_ca),
        "n_protein": int(n_prot),
        "constraint_mode_requested": str(args.constraint_mode),
        "constraint_mode_used": str(mode_used),
        "n_constraints": int(n_constraints),
        "projection_residual": residual,
        "mean_var_fx": mfx,
        "mean_var_fy": mfy,
        "mean_var_fz": mfz,
        "mean_var_fmag2": mfmag2,
        "rms_per_component": rms_per_comp,
        "rms_force_magnitude": rms_force_magnitude,
    }
    args.summary_out.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Wrote summary JSON: {args.summary_out}")

    print(
        f"\nSummary ({n_ca} CA atoms, {n_frames} frames):\n"
        f"  mean var_fx   = {mfx:.6e}  (kcal/mol/A)^2\n"
        f"  mean var_fy   = {mfy:.6e}  (kcal/mol/A)^2\n"
        f"  mean var_fz   = {mfz:.6e}  (kcal/mol/A)^2\n"
        f"  mean var|F|^2 = {mfmag2:.6e}  (kcal/mol/A)^2\n"
        f"  RMS per-comp  = {rms_per_comp:.4f}  kcal/mol/A  [compare to analysis force_rmse]\n"
        f"  RMS |F| noise = {rms_force_magnitude:.4f}  kcal/mol/A"
    )


if __name__ == "__main__":
    main()
