#!/usr/bin/env python3
"""Compute per-CA aggforce-projected force variance from a LAMMPS protein-atom dump.

Workflow:
  1. Read a LAMMPS custom dump of all protein atoms (IDs 1..N_prot) that includes
     fx, fy, fz columns.
  2. Apply aggforce optimal linear force projection onto the CA beads.
  3. Compute per-CA variance across the trajectory and write CSV output.

The dump must be produced with:
    dump  dprot protein custom <stride> protein_forces.dump id type x y z fx fy fz
    dump_modify dprot sort id format line "%d %d %.8f %.8f %.8f %.10f %.10f %.10f"

where the 'protein' group contains exactly the protein segment atoms
(LAMMPS IDs 1..N_PROTEIN, default 1390).

Usage:
    python compute_aggforce_variance.py \\
        --dump protein_forces.dump \\
        --ca-ids ca_atom_ids.txt \\
        --out aggforce_ca_variance.csv \\
        [--n-protein 1390] [--blocks 4] [--block-out aggforce_ca_variance_blocks.csv]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterator, List, Tuple

import numpy as np
from aggforce import LinearMap, project_forces


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dump", type=Path, required=True,
                   help="LAMMPS custom dump with protein-atom positions and forces")
    p.add_argument("--ca-ids", type=Path, required=True,
                   help="File with LAMMPS CA atom IDs (one per line)")
    p.add_argument("--out", type=Path, default=Path("aggforce_ca_variance.csv"),
                   help="Per-CA variance output CSV")
    p.add_argument("--block-out", type=Path, default=Path("aggforce_ca_variance_blocks.csv"),
                   help="Block-convergence summary CSV")
    p.add_argument("--n-protein", type=int, default=1390,
                   help="Number of protein atoms (LAMMPS IDs 1..N); default 1390")
    p.add_argument("--blocks", type=int, default=4,
                   help="Number of time blocks for convergence summary; default 4")
    p.add_argument("--max-frames", type=int, default=0,
                   help="If >0, subsample this many frames uniformly from the dump (reduces memory)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Dump parser
# ---------------------------------------------------------------------------

def parse_dump_frames(
    path: Path,
    n_atoms: int,
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
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

            if not fh.readline():           # timestep value
                return
            if not fh.readline().startswith("ITEM: NUMBER OF ATOMS"):
                continue
            try:
                n = int(fh.readline().strip())
            except ValueError:
                continue
            if n != n_atoms:
                # skip frames with unexpected atom count
                for _ in range(n + 4):
                    fh.readline()
                continue

            if not fh.readline().startswith("ITEM: BOX BOUNDS"):
                continue
            fh.readline(); fh.readline(); fh.readline()  # 3 box lines

            if not fh.readline().startswith("ITEM: ATOMS"):
                continue

            R = np.empty((n_atoms, 3), dtype=np.float64)
            F = np.empty((n_atoms, 3), dtype=np.float64)
            ok = True
            for i in range(n_atoms):
                raw = fh.readline()
                if not raw:
                    ok = False
                    break
                cols = raw.split()
                if len(cols) < 8:
                    ok = False
                    break
                try:
                    idx = int(cols[0]) - 1          # LAMMPS 1-indexed → 0-indexed
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_ca_ids(path: Path) -> List[int]:
    """Return list of LAMMPS CA atom IDs (1-indexed) from a text file."""
    ids = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            # accept "index  id" (two columns) or just "id" (one column)
            parts = line.split()
            try:
                ids.append(int(parts[-1]))
            except ValueError:
                continue
    if not ids:
        raise ValueError(f"No CA atom IDs found in {path}")
    return ids


def ca_indices_in_protein(ca_lammps_ids: List[int]) -> List[int]:
    """Convert 1-indexed LAMMPS IDs to 0-indexed positions in the protein array.

    Valid because protein atoms have LAMMPS IDs 1..N_prot stored contiguously,
    so index = lammps_id - 1.
    """
    return [i - 1 for i in ca_lammps_ids]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    if not args.dump.exists():
        sys.exit(f"ERROR: dump file not found: {args.dump}")
    if not args.ca_ids.exists():
        sys.exit(f"ERROR: CA IDs file not found: {args.ca_ids}")
    if args.blocks < 1:
        sys.exit("ERROR: --blocks must be >= 1")

    ca_lammps_ids = load_ca_ids(args.ca_ids)
    ca_idx = ca_indices_in_protein(ca_lammps_ids)
    n_ca = len(ca_idx)
    n_prot = args.n_protein

    print(f"Protein atoms : {n_prot}")
    print(f"CA atoms      : {n_ca}")
    print(f"Dump file     : {args.dump}")

    # --- Load all frames into memory ---
    R_list, F_list = [], []
    for R, F in parse_dump_frames(args.dump, n_prot):
        R_list.append(R)
        F_list.append(F)

    if not R_list:
        sys.exit("ERROR: no complete frames parsed from dump.")

    n_frames_total = len(R_list)

    # Optional uniform subsampling to cap memory usage
    if args.max_frames > 0 and n_frames_total > args.max_frames:
        indices = np.round(np.linspace(0, n_frames_total - 1, args.max_frames)).astype(int)
        R_list = [R_list[i] for i in indices]
        F_list = [F_list[i] for i in indices]
        print(f"Subsampled    : {n_frames_total} → {len(R_list)} frames (--max-frames)")

    n_frames = len(R_list)
    coords = np.stack(R_list, axis=0)   # (n_frames, n_prot, 3)
    forces = np.stack(F_list, axis=0)   # (n_frames, n_prot, 3)
    del R_list, F_list
    print(f"Loaded frames : {n_frames}  →  coords {coords.shape}, forces {forces.shape}")

    # --- Build aggforce LinearMap ---
    # Each CG bead maps to exactly one CA atom; indices are into protein-only array.
    cmap = LinearMap([[i] for i in ca_idx], n_fg_sites=n_prot)

    # --- Project forces ---
    # Try auto constraint detection first; fall back to no constraints if it fails
    # (can fail with a matmul dimension error on some proteins due to degenerate constraints).
    print("Running aggforce project_forces (auto constraint detection)...")
    try:
        result = project_forces(
            coords=coords,
            forces=forces,
            coord_map=cmap,
            constrained_inds="auto",
        )
        n_constraints = len(result["constraints"])
        print(f"  Detected constraints : {n_constraints}")
    except (ValueError, np.linalg.LinAlgError) as e:
        print(f"  WARNING: auto-constraint projection failed ({e}); retrying with no constraints.")
        result = project_forces(
            coords=coords,
            forces=forces,
            coord_map=cmap,
            constrained_inds=set(),
        )
        n_constraints = 0
        print(f"  Detected constraints : 0 (fallback)")
    mapped_forces = result["mapped_forces"]  # (n_frames, n_ca, 3)
    residual = result["residual"]
    print(f"  Projection residual  : {residual:.6e}  (mean squared mapped force)")

    # --- Compute per-CA variance across frames ---
    # var shape: (n_ca, 3)
    var = np.var(mapped_forces, axis=0, ddof=1)   # unbiased, (n_ca, 3)
    var_fmag2 = var.sum(axis=1)                    # (n_ca,)

    # --- Write per-atom CSV ---
    with args.out.open("w") as g:
        print("id,n_samples,var_fx,var_fy,var_fz,var_fmag2", file=g)
        for k, lammps_id in enumerate(ca_lammps_ids):
            print(
                f"{lammps_id},{n_frames},"
                f"{var[k,0]:.12e},{var[k,1]:.12e},{var[k,2]:.12e},{var_fmag2[k]:.12e}",
                file=g,
            )
    print(f"Wrote per-CA CSV : {args.out}")

    # --- Block convergence summary ---
    block_rows = []
    for b in range(args.blocks):
        i0 = (b * n_frames) // args.blocks
        i1 = ((b + 1) * n_frames) // args.blocks
        blk = mapped_forces[i0:i1]                        # (blk_frames, n_ca, 3)
        blk_n = blk.shape[0]
        if blk_n < 2:
            continue
        blk_var = np.var(blk, axis=0, ddof=1)            # (n_ca, 3)
        block_rows.append((b + 1, blk_n,
                           blk_var[:, 0].mean(),
                           blk_var[:, 1].mean(),
                           blk_var[:, 2].mean(),
                           blk_var.sum(axis=1).mean()))

    with args.block_out.open("w") as g:
        print("block,n_frames,n_atoms,mean_var_fx,mean_var_fy,mean_var_fz,mean_var_fmag2", file=g)
        for row in block_rows:
            print(
                f"{row[0]},{row[1]},{n_ca},"
                f"{row[2]:.12e},{row[3]:.12e},{row[4]:.12e},{row[5]:.12e}",
                file=g,
            )
    print(f"Wrote block CSV  : {args.block_out}")

    # --- Summary ---
    global_var = np.var(mapped_forces, axis=0, ddof=1)
    mfx = global_var[:, 0].mean()
    mfy = global_var[:, 1].mean()
    mfz = global_var[:, 2].mean()
    mfmag2 = global_var.sum(axis=1).mean()
    print(
        f"\nSummary ({n_ca} CA atoms, {n_frames} frames):\n"
        f"  mean var_fx   = {mfx:.6e}  (kcal/mol/A)^2\n"
        f"  mean var_fy   = {mfy:.6e}  (kcal/mol/A)^2\n"
        f"  mean var_fz   = {mfz:.6e}  (kcal/mol/A)^2\n"
        f"  mean var|F|^2 = {mfmag2:.6e}  (kcal/mol/A)^2\n"
        f"  RMS per-comp  = {np.sqrt((mfx+mfy+mfz)/3):.4f}  kcal/mol/A\n"
        f"  RMS |F| noise = {np.sqrt(mfmag2):.4f}  kcal/mol/A"
    )


if __name__ == "__main__":
    main()
