#!/usr/bin/env python3
"""Apply a saved aggforce weight matrix to fixed-CA MD protein forces.

This gives the thermal noise floor in exactly the same metric as the aggforce
training targets — enabling direct comparison with model force RMSE.

Workflow:
  1. Load the aggforce weight matrix W (shape n_cg × n_prot) from a per-protein
     CG NPZ produced by cg_1bead.py with --use_aggforce (after re-running the
     pipeline with the weight-matrix extraction patch).
  2. Parse the LAMMPS custom dump of all protein atoms (IDs 1..N_prot).
  3. For each frame: F_agg[k, :] = W[k, :] @ F_prot[:, :]  →  shape (n_cg, 3)
  4. Compute per-CA variance across frames and output CSV + JSON.

The RMS per-component metric in the JSON is directly comparable to cameo_cg
model force RMSE values (component-wise).

Usage:
    python compute_aggforce_projected_variance.py \\
        --cg-npz path/to/protein_cg.npz \\
        --dump   path/to/protein_forces.dump \\
        --n-protein 1613
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterator, List, Tuple

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--cg-npz", type=Path, required=True,
                   help="Per-protein CG NPZ containing 'aggforce_weight_matrix'")
    p.add_argument("--dump", type=Path, required=True,
                   help="LAMMPS custom dump with all protein-atom positions and forces")
    p.add_argument("--n-protein", type=int, required=True,
                   help="Number of protein atoms in the dump (LAMMPS IDs 1..N)")
    p.add_argument("--out", type=Path, default=Path("aggforce_projected_variance.csv"),
                   help="Per-CA variance output CSV")
    p.add_argument("--block-out", type=Path, default=Path("aggforce_projected_variance_blocks.csv"),
                   help="Block-convergence summary CSV")
    p.add_argument("--summary-out", type=Path, default=Path("aggforce_projected_variance_summary.json"),
                   help="Summary JSON with comparable RMS metrics")
    p.add_argument("--blocks", type=int, default=4,
                   help="Number of time blocks for convergence check (default 4)")
    p.add_argument("--max-frames", type=int, default=0,
                   help="If >0, subsample this many frames uniformly")
    return p.parse_args()


def load_weight_matrix(npz_path: Path) -> Tuple[np.ndarray, List[int]]:
    """Return (W, ca_indices) from the CG NPZ.

    W has shape (n_cg, n_prot), float64.
    ca_indices are 0-based protein-atom indices for labelling output.
    """
    data = np.load(npz_path, allow_pickle=True)
    if "aggforce_weight_matrix" not in data:
        raise KeyError(
            f"'aggforce_weight_matrix' not found in {npz_path}. "
            "Re-run cg_1bead.py with the weight-matrix extraction patch."
        )
    W = np.asarray(data["aggforce_weight_matrix"], dtype=np.float64)
    ca_indices = list(np.asarray(data["ca_indices"], dtype=int))
    return W, ca_indices


def parse_dump_frames(
    path: Path, n_atoms: int
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """Yield (R, F) per frame, each shape (n_atoms, 3), float64.

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


def main() -> None:
    args = parse_args()

    for path, label in [(args.cg_npz, "CG NPZ"), (args.dump, "dump file")]:
        if not path.exists():
            sys.exit(f"ERROR: {label} not found: {path}")
    if args.blocks < 1:
        sys.exit("ERROR: --blocks must be >= 1")
    if args.n_protein < 1:
        sys.exit("ERROR: --n-protein must be >= 1")

    print(f"Loading weight matrix from : {args.cg_npz}")
    W, ca_indices = load_weight_matrix(args.cg_npz)
    n_cg, n_prot_W = W.shape
    print(f"  Weight matrix shape : {W.shape}  (n_cg={n_cg}, n_prot_W={n_prot_W})")

    if n_prot_W != args.n_protein:
        sys.exit(
            f"ERROR: weight matrix has {n_prot_W} protein atoms but "
            f"--n-protein={args.n_protein}. Check that the CG NPZ matches the dump."
        )

    n_prot = args.n_protein
    print(f"Parsing dump               : {args.dump}")

    F_list: list[np.ndarray] = []
    for _R, F_prot in parse_dump_frames(args.dump, n_prot):
        # F_agg[k, :] = W[k, :] @ F_prot  →  (n_cg, 3)
        F_agg = W @ F_prot          # (n_cg, 3)
        F_list.append(F_agg)

    if not F_list:
        sys.exit("ERROR: no complete frames parsed from dump.")

    n_frames_total = len(F_list)
    if args.max_frames > 0 and n_frames_total > args.max_frames:
        indices = np.round(np.linspace(0, n_frames_total - 1, args.max_frames)).astype(int)
        F_list = [F_list[i] for i in indices]
        print(f"Subsampled : {n_frames_total} -> {len(F_list)} frames (--max-frames)")

    n_frames = len(F_list)
    mapped_forces = np.stack(F_list, axis=0)   # (n_frames, n_cg, 3)
    del F_list
    print(f"Projected forces shape     : {mapped_forces.shape}")

    # --- per-CA variance ---
    var = np.var(mapped_forces, axis=0, ddof=1)   # (n_cg, 3)
    var_fmag2 = var.sum(axis=1)                   # (n_cg,)

    with args.out.open("w") as g:
        print("ca_index_0based,n_samples,var_fx,var_fy,var_fz,var_fmag2", file=g)
        for k, ca0 in enumerate(ca_indices):
            print(
                f"{ca0},{n_frames},"
                f"{var[k,0]:.12e},{var[k,1]:.12e},{var[k,2]:.12e},{var_fmag2[k]:.12e}",
                file=g,
            )
    print(f"Wrote per-CA CSV  : {args.out}")

    # --- block convergence ---
    block_rows = []
    for b in range(args.blocks):
        i0 = (b * n_frames) // args.blocks
        i1 = ((b + 1) * n_frames) // args.blocks
        blk = mapped_forces[i0:i1]
        if blk.shape[0] < 2:
            continue
        blk_var = np.var(blk, axis=0, ddof=1)   # (n_cg, 3)
        block_rows.append((
            b + 1,
            blk.shape[0],
            float(blk_var[:, 0].mean()),
            float(blk_var[:, 1].mean()),
            float(blk_var[:, 2].mean()),
            float(blk_var.sum(axis=1).mean()),
        ))

    with args.block_out.open("w") as g:
        print("block,n_frames,n_ca,mean_var_fx,mean_var_fy,mean_var_fz,mean_var_fmag2", file=g)
        for row in block_rows:
            print(
                f"{row[0]},{row[1]},{n_cg},"
                f"{row[2]:.12e},{row[3]:.12e},{row[4]:.12e},{row[5]:.12e}",
                file=g,
            )
    print(f"Wrote block CSV   : {args.block_out}")

    # --- global summary ---
    global_var = np.var(mapped_forces, axis=0, ddof=1)
    mfx = float(global_var[:, 0].mean())
    mfy = float(global_var[:, 1].mean())
    mfz = float(global_var[:, 2].mean())
    mfmag2 = float(global_var.sum(axis=1).mean())
    rms_per_comp = float(np.sqrt((mfx + mfy + mfz) / 3.0))
    rms_force_magnitude = float(np.sqrt(mfmag2))

    summary = {
        "n_frames": int(n_frames),
        "n_ca": int(n_cg),
        "n_protein": int(n_prot),
        "weight_matrix_shape": list(W.shape),
        "mean_var_fx": mfx,
        "mean_var_fy": mfy,
        "mean_var_fz": mfz,
        "mean_var_fmag2": mfmag2,
        "rms_per_component": rms_per_comp,
        "rms_force_magnitude": rms_force_magnitude,
    }
    args.summary_out.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Wrote summary JSON : {args.summary_out}")

    print(
        f"\nNoise floor in aggforce metric ({n_cg} CA beads, {n_frames} frames):\n"
        f"  mean var_fx     = {mfx:.6e}  (kcal/mol/A)^2\n"
        f"  mean var_fy     = {mfy:.6e}  (kcal/mol/A)^2\n"
        f"  mean var_fz     = {mfz:.6e}  (kcal/mol/A)^2\n"
        f"  mean var|F|^2   = {mfmag2:.6e}  (kcal/mol/A)^2\n"
        f"  RMS per-comp    = {rms_per_comp:.4f}  kcal/mol/A  [compare to model force_rmse]\n"
        f"  RMS |F| noise   = {rms_force_magnitude:.4f}  kcal/mol/A\n"
        f"\n  Expected: RMS per-comp < 8-9 kcal/mol/A (model RMSE) for consistent results."
    )


if __name__ == "__main__":
    main()
