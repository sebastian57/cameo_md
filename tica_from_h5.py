#!/usr/bin/env python3
"""TICA from mdCATH h5 trajectory with MLCG trajectory overlay.

Fits a TICA model on coordinates loaded directly from an mdCATH h5 file
(all runs at a given temperature group, concatenated), then projects an MLCG
LAMMPS dump trajectory onto the same TICA coordinate space for comparison.
By default this uses CA atoms; pass --atom-selection all for all-atom analysis.

Usage (single protein):
    python tica_from_h5.py \
        --h5    /path/to/mdcath_dataset_2gy5A01.h5 \
        --protein 2gy5A01 \
        --mlcg-dump /path/to/ca_tica.dump \
        --outdir /path/to/output/ \
        --prefix 2gy5A01_h5_vs_mlcg

Usage (experiment mode, reads mlcg_manifest.tsv from experiment root):
    python tica_from_h5.py \
        --experiment-root /path/to/outputs/my_experiment \
        --h5-dir /path/to/structures/larger_dataset \
        --outdir /path/to/outputs/my_experiment/tica_h5

The h5-based reference trajectory is always fitted on the full h5 data (all runs).
The MLCG trajectory is projected onto the fitted TICA model and compared on a
frame count matched to the ML trajectory length (controlled by --max-mlcg-frames).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
import re
import sys
from pathlib import Path
from typing import Optional

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Shared helpers (duplicated from tica_from_lammps_dump.py to keep this script
# self-contained; keep the implementations in sync if either changes)
# ---------------------------------------------------------------------------

def choose_pairs(n_atoms: int, n_pairs: int, mode: str, seed: int) -> np.ndarray:
    if n_atoms < 2:
        raise ValueError("Need at least 2 atoms for pair-distance features")
    if mode == "sequential":
        pairs = np.array([(i, i + 1) for i in range(n_atoms - 1)], dtype=np.int64)
        return pairs[: min(n_pairs, len(pairs))] if n_pairs > 0 else pairs
    rng = np.random.default_rng(seed)
    max_pairs = n_atoms * (n_atoms - 1) // 2
    target = min(max(n_pairs, 1), max_pairs)
    selected: set[tuple[int, int]] = set()
    while len(selected) < target:
        i = int(rng.integers(0, n_atoms - 1))
        j = int(rng.integers(i + 1, n_atoms))
        selected.add((i, j))
    return np.asarray(sorted(selected), dtype=np.int64)


def load_pairs_csv(path: Path) -> np.ndarray:
    pairs: list[tuple[int, int]] = []
    with path.open("r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            i = int(row["atom_i_1based"]) - 1
            j = int(row["atom_j_1based"]) - 1
            if i > j:
                i, j = j, i
            pairs.append((i, j))
    if not pairs:
        raise ValueError(f"No pairs found in {path}")
    return np.asarray(pairs, dtype=np.int64)


def build_pair_distance_features(coords: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    diff = coords[:, pairs[:, 0], :] - coords[:, pairs[:, 1], :]
    return np.linalg.norm(diff, axis=-1).astype(np.float64)


def fit_tica(X: np.ndarray, lagtime: int) -> tuple[object, np.ndarray]:
    try:
        from deeptime.decomposition import TICA
    except Exception as exc:
        raise RuntimeError("deeptime is required. Activate the test_env_newsetup venv.") from exc
    if X.shape[0] <= lagtime:
        raise ValueError(f"Need n_frames > lagtime, got n_frames={X.shape[0]}, lagtime={lagtime}")
    model = TICA(lagtime=lagtime, dim=2).fit(X).fetch_model()
    Y = np.asarray(model.transform(X), dtype=np.float64)
    return model, Y


def project_tica(X: np.ndarray, model_pkl: Path) -> np.ndarray:
    with model_pkl.open("rb") as f:
        model = pickle.load(f)
    return np.asarray(model.transform(X), dtype=np.float64)


def compute_fes_2d(
    Y: np.ndarray,
    bins: int,
    temperature: float,
    xedges: Optional[np.ndarray] = None,
    yedges: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x, y = Y[:, 0], Y[:, 1]
    if xedges is not None and yedges is not None:
        H, xedges, yedges = np.histogram2d(x, y, bins=[xedges, yedges])
    else:
        H, xedges, yedges = np.histogram2d(x, y, bins=bins)
    occupied = H > 0
    P = np.zeros_like(H, dtype=np.float64)
    P[occupied] = H[occupied] / H.sum()
    kB = 0.00831446261815324  # kJ/mol/K
    F = np.full_like(P, np.nan, dtype=np.float64)
    F[occupied] = -kB * temperature * np.log(P[occupied])
    if not np.any(occupied):
        return F, xedges, yedges
    F[occupied] -= np.nanmin(F[occupied])
    return F, xedges, yedges


def plot_fes(
    F: np.ndarray,
    xedges: np.ndarray,
    yedges: np.ndarray,
    out_png: Path,
    title: str = "TICA FES",
    vmax: Optional[float] = None,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    cmap = plt.colormaps["turbo"].copy()
    cmap.set_bad("white")
    m = ax.pcolormesh(xedges, yedges, np.ma.masked_invalid(F).T, cmap=cmap,
                      shading="flat", vmin=0, vmax=vmax)
    ax.set_xlabel("TIC 1")
    ax.set_ylabel("TIC 2")
    ax.set_title(title)
    cbar = fig.colorbar(m, ax=ax)
    cbar.set_label("F [kJ/mol]")
    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_projection_csv(path: Path, Y: np.ndarray, frame_indices: Optional[np.ndarray] = None) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame_index", "tic1", "tic2"])
        for i in range(Y.shape[0]):
            idx = int(frame_indices[i]) if frame_indices is not None else i
            w.writerow([idx, float(Y[i, 0]), float(Y[i, 1])])


def write_pairs_csv(path: Path, pairs: np.ndarray) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pair_index", "atom_i_1based", "atom_j_1based"])
        for k, (i, j) in enumerate(pairs):
            w.writerow([k, int(i + 1), int(j + 1)])


def stable_hash_pairs(pairs: np.ndarray) -> str:
    payload = ";".join(f"{int(i)}-{int(j)}" for i, j in pairs).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


# ---------------------------------------------------------------------------
# h5-specific helpers
# ---------------------------------------------------------------------------

def extract_ca_indices_from_pdb(pdb_protein_atoms: str) -> list[int]:
    """Return 0-based indices of CA atoms in the all-protein-atom list."""
    indices: list[int] = []
    atom_idx = 0
    for line in pdb_protein_atoms.splitlines():
        if line.startswith("ATOM"):
            if line[12:16].strip() == "CA":
                indices.append(atom_idx)
            atom_idx += 1
    return indices


def load_h5_coords(
    h5_path: Path,
    protein: str,
    temp_group: str = "320",
    atom_selection: str = "ca",
) -> tuple[np.ndarray, list[int]]:
    """Load selected coordinates from an mdCATH h5 file across all runs.

    Returns:
        coords: (n_total_frames, n_selected_atoms, 3) in Angstrom
        run_boundaries: list of cumulative frame counts per run (for metadata)
    """
    if atom_selection not in {"ca", "all"}:
        raise ValueError(f"Unsupported atom_selection={atom_selection!r}; expected 'ca' or 'all'")

    with h5py.File(h5_path, "r") as h5f:
        prot = h5f[protein] if protein in h5f else h5f[next(iter(h5f.keys()))]
        pdb_prot_atoms: str = prot["pdbProteinAtoms"][()].decode()
        if atom_selection == "ca":
            atom_indices = extract_ca_indices_from_pdb(pdb_prot_atoms)
            if not atom_indices:
                raise ValueError(f"No CA atoms found in pdbProteinAtoms for {protein}")
        else:
            n_atoms = sum(1 for line in pdb_prot_atoms.splitlines() if line.startswith("ATOM"))
            if n_atoms <= 0:
                raise ValueError(f"No ATOM records found in pdbProteinAtoms for {protein}")
            atom_indices = list(range(n_atoms))

        temp_grp = prot[temp_group]
        run_keys = sorted(temp_grp.keys(), key=lambda k: int(k))

        frames_list: list[np.ndarray] = []
        run_boundaries: list[int] = []
        for rk in run_keys:
            coords_all = temp_grp[rk]["coords"][()]  # (n_frames, n_atoms, 3)
            selected_coords = coords_all[:, atom_indices, :]
            frames_list.append(selected_coords.astype(np.float64))
            run_boundaries.append(len(selected_coords))

    all_coords = np.concatenate(frames_list, axis=0)
    return all_coords, run_boundaries


def load_h5_ca_coords(
    h5_path: Path,
    protein: str,
    temp_group: str = "320",
) -> tuple[np.ndarray, list[int]]:
    """Backward-compatible wrapper for CA-only h5 loading."""
    return load_h5_coords(h5_path, protein, temp_group, atom_selection="ca")


def parse_lammps_dump_coords(
    path: Path,
    max_frames: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Parse CA coordinates from a LAMMPS custom dump file.

    Returns:
        coords: (n_frames, n_atoms, 3)
        timesteps: (n_frames,)
    """
    coords_frames: list[np.ndarray] = []
    timesteps: list[int] = []

    with path.open("r") as fh:
        while True:
            line = fh.readline()
            if not line:
                break
            if not line.startswith("ITEM: TIMESTEP"):
                continue
            ts_line = fh.readline()
            if not ts_line:
                break
            timestep = int(ts_line.strip())

            if not fh.readline().startswith("ITEM: NUMBER OF ATOMS"):
                continue
            n_atoms = int(fh.readline().strip())

            if not fh.readline().startswith("ITEM: BOX BOUNDS"):
                continue
            fh.readline(); fh.readline(); fh.readline()

            atom_line = fh.readline().strip()
            if not atom_line.startswith("ITEM: ATOMS"):
                continue
            atom_header = atom_line.split()[2:]

            if "id" not in atom_header:
                raise ValueError("Dump must include atom id column")
            id_idx = atom_header.index("id")
            x_col = "xu" if "xu" in atom_header else "x"
            y_col = "yu" if "yu" in atom_header else "y"
            z_col = "zu" if "zu" in atom_header else "z"
            x_idx = atom_header.index(x_col)
            y_idx = atom_header.index(y_col)
            z_idx = atom_header.index(z_col)

            rows: list[tuple[int, float, float, float]] = []
            ok = True
            for _ in range(n_atoms):
                row = fh.readline()
                if not row:
                    ok = False
                    break
                cols = row.split()
                try:
                    rows.append((int(cols[id_idx]), float(cols[x_idx]),
                                 float(cols[y_idx]), float(cols[z_idx])))
                except Exception:
                    ok = False
                    break
            if not ok:
                continue

            rows.sort(key=lambda r: r[0])
            frame = np.array([[r[1], r[2], r[3]] for r in rows], dtype=np.float64)
            coords_frames.append(frame)
            timesteps.append(timestep)

            if max_frames > 0 and len(coords_frames) >= max_frames:
                break

    if not coords_frames:
        raise ValueError(f"No complete frames parsed from {path}")
    return np.stack(coords_frames, axis=0), np.asarray(timesteps, dtype=np.int64)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)

    # Single-protein mode
    p.add_argument("--h5", type=Path, default=None,
                   help="mdCATH h5 file for a single protein (single mode)")
    p.add_argument("--protein", type=str, default=None,
                   help="Protein label inside h5 file (e.g. 2gy5A01; single mode)")
    p.add_argument("--mlcg-dump", type=Path, default=None,
                   help="LAMMPS ca_tica.dump from ML CG run (single mode)")
    p.add_argument("--prefix", type=str, default="tica",
                   help="Filename prefix for outputs (single mode)")

    # Experiment mode
    p.add_argument("--experiment-root", type=Path, default=None,
                   help="Experiment root dir (e.g. outputs/exp_v2); reads mlcg_manifest.tsv")
    p.add_argument("--h5-dir", type=Path, default=None,
                   help="Directory with mdcath_dataset_<protein>.h5 files (experiment mode)")
    p.add_argument("--mlcg-set", choices=("both", "t0", "eq", "h5"), default="both",
                   help="Which seed modes to include (experiment mode). "
                        "Use 'h5' for runs prepared by prepare_mlcg_from_h5.py.")

    # Shared
    p.add_argument("--outdir", type=Path, required=True,
                   help="Output directory")
    p.add_argument("--temp-group", type=str, default="320",
                   help="Temperature group in h5 file (default: 320)")
    p.add_argument("--lagtime", type=int, default=10,
                   help="TICA lagtime in frames (applied to h5 frame stride)")
    p.add_argument("--bins", type=int, default=80,
                   help="Histogram bins per axis for FES")
    p.add_argument("--temperature", type=float, default=320.0,
                   help="Temperature for Boltzmann FES weights in K")
    p.add_argument("--n-pairs", type=int, default=200,
                   help="Number of pair-distance features for TICA")
    p.add_argument("--pair-seed", type=int, default=42,
                   help="RNG seed for random pair selection")
    p.add_argument("--max-mlcg-frames", type=int, default=0,
                   help="Max frames to use from ML trajectory (0 = all)")
    p.add_argument("--shared-bins", action="store_true",
                   help="Use the same histogram bin edges for h5 and ML FES plots")
    p.add_argument("--atom-selection", choices=("ca", "all"), default="ca",
                   help="Atoms used from the h5 reference: ca (default) or all")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Single-protein mode
# ---------------------------------------------------------------------------

def run_single(args: argparse.Namespace) -> None:
    if args.h5 is None or args.protein is None or args.mlcg_dump is None:
        raise SystemExit("Single mode requires --h5, --protein, and --mlcg-dump")

    args.outdir.mkdir(parents=True, exist_ok=True)

    print(f"Loading h5 {args.atom_selection} coords: {args.h5}")
    h5_coords, run_bounds = load_h5_coords(
        args.h5,
        args.protein,
        args.temp_group,
        atom_selection=args.atom_selection,
    )
    n_h5, n_atoms, _ = h5_coords.shape
    print(f"  {n_h5} frames, {n_atoms} atoms (runs: {run_bounds})")

    pairs = choose_pairs(n_atoms, args.n_pairs, mode="random", seed=args.pair_seed)
    pair_hash = stable_hash_pairs(pairs)
    X_h5 = build_pair_distance_features(h5_coords, pairs)

    print(f"Fitting TICA on {n_h5} h5 frames, lagtime={args.lagtime}")
    tica_model, Y_h5 = fit_tica(X_h5, lagtime=args.lagtime)

    print(f"Loading MLCG dump: {args.mlcg_dump}")
    ml_coords, ml_ts = parse_lammps_dump_coords(args.mlcg_dump, max_frames=args.max_mlcg_frames)
    n_ml = ml_coords.shape[0]
    if ml_coords.shape[1] != n_atoms:
        raise ValueError(
            f"Atom count mismatch: h5 {args.atom_selection} selection has {n_atoms} atoms, "
            f"ML dump has {ml_coords.shape[1]}"
        )
    print(f"  {n_ml} frames")

    X_ml = build_pair_distance_features(ml_coords, pairs)
    Y_ml = np.asarray(tica_model.transform(X_ml), dtype=np.float64)

    prefix = args.prefix
    model_pkl = args.outdir / f"{prefix}_model.pkl"
    pairs_csv = args.outdir / f"{prefix}_pairs.csv"
    h5_proj_csv = args.outdir / f"{prefix}_h5_projection.csv"
    ml_proj_csv = args.outdir / f"{prefix}_mlcg_projection.csv"
    h5_fes_npz = args.outdir / f"{prefix}_h5_fes.npz"
    ml_fes_npz = args.outdir / f"{prefix}_mlcg_fes.npz"
    h5_fes_png = args.outdir / f"{prefix}_h5_fes.png"
    ml_fes_png = args.outdir / f"{prefix}_mlcg_fes.png"
    meta_json = args.outdir / f"{prefix}_metadata.json"

    with model_pkl.open("wb") as f:
        pickle.dump(tica_model, f)
    write_pairs_csv(pairs_csv, pairs)
    write_projection_csv(h5_proj_csv, Y_h5)
    write_projection_csv(ml_proj_csv, Y_ml, frame_indices=ml_ts)

    F_h5, xe_h5, ye_h5 = compute_fes_2d(Y_h5, bins=args.bins, temperature=args.temperature)
    np.savez(h5_fes_npz, F=F_h5, xedges=xe_h5, yedges=ye_h5)
    plot_fes(F_h5, xe_h5, ye_h5, h5_fes_png, title=f"{args.protein} h5 reference")

    if args.shared_bins:
        F_ml, _, _ = compute_fes_2d(Y_ml, bins=args.bins, temperature=args.temperature,
                                    xedges=xe_h5, yedges=ye_h5)
    else:
        F_ml, xe_ml, ye_ml = compute_fes_2d(Y_ml, bins=args.bins, temperature=args.temperature)
        xe_h5, ye_h5 = xe_ml, ye_ml  # use ML edges for saving
    np.savez(ml_fes_npz, F=F_ml, xedges=xe_h5, yedges=ye_h5)
    plot_fes(F_ml, xe_h5, ye_h5, ml_fes_png, title=f"{args.protein} MLCG")

    metadata = {
        "protein": args.protein,
        "h5": str(args.h5.resolve()),
        "temp_group": args.temp_group,
        "atom_selection": args.atom_selection,
        "n_h5_frames": int(n_h5),
        "n_atoms": int(n_atoms),
        "n_pairs": int(pairs.shape[0]),
        "pair_hash": pair_hash,
        "lagtime": int(args.lagtime),
        "bins": int(args.bins),
        "temperature": float(args.temperature),
        "run_boundaries_h5": run_bounds,
        "n_mlcg_frames": int(n_ml),
        "max_mlcg_frames": int(args.max_mlcg_frames),
        "mlcg_dump": str(args.mlcg_dump.resolve()),
        "shared_bins": bool(args.shared_bins),
        "outputs": {
            "model_pkl": str(model_pkl),
            "pairs_csv": str(pairs_csv),
            "h5_projection_csv": str(h5_proj_csv),
            "mlcg_projection_csv": str(ml_proj_csv),
            "h5_fes_npz": str(h5_fes_npz),
            "mlcg_fes_npz": str(ml_fes_npz),
            "h5_fes_png": str(h5_fes_png),
            "mlcg_fes_png": str(ml_fes_png),
        },
    }
    meta_json.write_text(json.dumps(metadata, indent=2, sort_keys=True))
    print(f"Done. Outputs in {args.outdir}")


# ---------------------------------------------------------------------------
# Experiment mode
# ---------------------------------------------------------------------------

def parse_manifest_rows(path: Path, expected_cols: int) -> list[list[str]]:
    rows: list[list[str]] = []
    with path.open() as f:
        for raw in f:
            raw = raw.rstrip("\n")
            if not raw.strip():
                continue
            cols = raw.split("\t")
            if len(cols) >= expected_cols:
                rows.append(cols)
    return rows


def find_h5(h5_dir: Path, protein: str) -> Optional[Path]:
    candidates = [
        h5_dir / f"mdcath_dataset_{protein}.h5",
        h5_dir / f"{protein}.h5",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def run_experiment(args: argparse.Namespace) -> None:
    if args.experiment_root is None or args.h5_dir is None:
        raise SystemExit("Experiment mode requires --experiment-root and --h5-dir")

    exp_root = args.experiment_root.resolve()
    h5_dir = args.h5_dir.resolve()
    out_root = args.outdir.resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    mlcg_manifest = exp_root / "mlcg" / "mlcg_manifest.tsv"
    if not mlcg_manifest.exists():
        mlcg_manifest = exp_root / "mlcg_manifest.tsv"
    if not mlcg_manifest.exists():
        raise SystemExit(f"MLCG manifest not found under {exp_root}")

    selected_modes = {"t0", "eq"} if args.mlcg_set == "both" else {args.mlcg_set}
    # "h5" seed mode is produced by prepare_mlcg_from_h5.py; treat it like t0/eq.

    mlcg_rows_raw = parse_manifest_rows(mlcg_manifest, expected_cols=9)
    mlcg_rows: list[dict] = []
    for cols in mlcg_rows_raw:
        run_id, protein, seed_mode, source_run_id, source_run_dir, run_dir, input_file, status, reason = cols[:9]
        if seed_mode not in selected_modes:
            continue
        if status != "ready":
            continue
        mlcg_rows.append({
            "run_id": run_id,
            "protein": protein,
            "seed_mode": seed_mode,
            "run_dir": Path(run_dir),
            "input_file": Path(input_file),
            "status": status,
        })

    proteins = sorted({r["protein"] for r in mlcg_rows}, key=str.lower)

    index_rows: list[dict] = []
    summary_rows: list[dict] = []

    for protein in proteins:
        protein_out = out_root / protein
        protein_out.mkdir(parents=True, exist_ok=True)

        h5_path = find_h5(h5_dir, protein)
        if h5_path is None:
            summary_rows.append({"protein": protein, "status": "rejected",
                                  "reason": f"no h5 file found in {h5_dir}"})
            continue

        try:
            h5_coords, run_bounds = load_h5_ca_coords(h5_path, protein, args.temp_group)
        except Exception as exc:
            summary_rows.append({"protein": protein, "status": "rejected",
                                  "reason": f"h5 load error: {exc}"})
            continue

        n_h5, n_ca, _ = h5_coords.shape
        pairs = choose_pairs(n_ca, args.n_pairs, mode="random", seed=args.pair_seed)
        pair_hash = stable_hash_pairs(pairs)

        X_h5 = build_pair_distance_features(h5_coords, pairs)
        try:
            tica_model, Y_h5 = fit_tica(X_h5, lagtime=args.lagtime)
        except Exception as exc:
            summary_rows.append({"protein": protein, "status": "rejected",
                                  "reason": f"TICA fit error: {exc}"})
            continue

        # Save h5 reference model and FES
        model_pkl = protein_out / f"tica_{protein}_h5_model.pkl"
        pairs_csv_path = protein_out / f"tica_{protein}_pairs.csv"
        h5_proj_csv = protein_out / f"tica_{protein}_h5_projection.csv"
        h5_fes_npz = protein_out / f"tica_{protein}_h5_fes.npz"
        h5_fes_png = protein_out / f"tica_{protein}_h5_fes.png"

        with model_pkl.open("wb") as f:
            pickle.dump(tica_model, f)
        write_pairs_csv(pairs_csv_path, pairs)
        write_projection_csv(h5_proj_csv, Y_h5)
        F_h5, xe_h5, ye_h5 = compute_fes_2d(Y_h5, bins=args.bins, temperature=args.temperature)
        np.savez(h5_fes_npz, F=F_h5, xedges=xe_h5, yedges=ye_h5)
        plot_fes(F_h5, xe_h5, ye_h5, h5_fes_png,
                 title=f"{protein} h5 reference ({n_h5} frames)")

        n_mlcg_ok = 0
        accepted_modes: set[str] = set()
        protein_mlcg = [r for r in mlcg_rows if r["protein"].lower() == protein.lower()]

        for ml_row in sorted(protein_mlcg, key=lambda r: (r["seed_mode"], r["run_id"])):
            seed_mode = ml_row["seed_mode"]
            run_id = ml_row["run_id"]
            ml_dump = ml_row["run_dir"] / "ca_tica.dump"

            if not ml_dump.exists():
                index_rows.append({"protein": protein, "seed_mode": seed_mode,
                                   "run_id": run_id, "status": "rejected",
                                   "reason": "ca_tica.dump not found"})
                continue

            try:
                ml_coords, ml_ts = parse_lammps_dump_coords(
                    ml_dump, max_frames=args.max_mlcg_frames
                )
            except Exception as exc:
                index_rows.append({"protein": protein, "seed_mode": seed_mode,
                                   "run_id": run_id, "status": "rejected",
                                   "reason": f"dump parse error: {exc}"})
                continue

            if ml_coords.shape[1] != n_ca:
                index_rows.append({"protein": protein, "seed_mode": seed_mode,
                                   "run_id": run_id, "status": "rejected",
                                   "reason": (f"atom count mismatch: h5={n_ca}, "
                                              f"ml={ml_coords.shape[1]}")})
                continue

            n_ml = ml_coords.shape[0]
            X_ml = build_pair_distance_features(ml_coords, pairs)
            Y_ml = np.asarray(tica_model.transform(X_ml), dtype=np.float64)

            base = protein_out / f"tica_{protein}_{seed_mode}__{run_id}"
            ml_proj_csv = base.with_name(base.name + "_projection.csv")
            ml_fes_npz = base.with_name(base.name + "_fes.npz")
            ml_fes_png = base.with_name(base.name + "_fes.png")
            meta_json = base.with_name(base.name + "_metadata.json")

            write_projection_csv(ml_proj_csv, Y_ml, frame_indices=ml_ts)

            if args.shared_bins:
                F_ml, _, _ = compute_fes_2d(Y_ml, bins=args.bins,
                                            temperature=args.temperature,
                                            xedges=xe_h5, yedges=ye_h5)
                xe_plot, ye_plot = xe_h5, ye_h5
            else:
                F_ml, xe_plot, ye_plot = compute_fes_2d(
                    Y_ml, bins=args.bins, temperature=args.temperature
                )

            np.savez(ml_fes_npz, F=F_ml, xedges=xe_plot, yedges=ye_plot)
            plot_fes(F_ml, xe_plot, ye_plot, ml_fes_png,
                     title=f"{protein} MLCG {seed_mode} ({n_ml} frames)")

            meta = {
                "protein": protein,
                "run_id": run_id,
                "seed_mode": seed_mode,
                "h5_path": str(h5_path),
                "temp_group": args.temp_group,
                "n_h5_frames": int(n_h5),
                "run_boundaries_h5": run_bounds,
                "n_ca": int(n_ca),
                "n_pairs": int(pairs.shape[0]),
                "pair_hash": pair_hash,
                "lagtime": int(args.lagtime),
                "bins": int(args.bins),
                "temperature": float(args.temperature),
                "n_mlcg_frames": int(n_ml),
                "max_mlcg_frames": int(args.max_mlcg_frames),
                "shared_bins": bool(args.shared_bins),
                "outputs": {
                    "h5_model_pkl": str(model_pkl),
                    "pairs_csv": str(pairs_csv_path),
                    "h5_projection_csv": str(h5_proj_csv),
                    "h5_fes_npz": str(h5_fes_npz),
                    "h5_fes_png": str(h5_fes_png),
                    "mlcg_projection_csv": str(ml_proj_csv),
                    "mlcg_fes_npz": str(ml_fes_npz),
                    "mlcg_fes_png": str(ml_fes_png),
                },
            }
            meta_json.write_text(json.dumps(meta, indent=2, sort_keys=True))

            accepted_modes.add(seed_mode)
            n_mlcg_ok += 1
            index_rows.append({
                "protein": protein,
                "seed_mode": seed_mode,
                "run_id": run_id,
                "status": "accepted",
                "reason": "",
                "metadata_json": str(meta_json),
                "mlcg_projection_csv": str(ml_proj_csv),
            })

        if n_mlcg_ok == 0:
            summary_rows.append({"protein": protein, "status": "rejected",
                                  "reason": "no accepted MLCG runs"})
        else:
            summary_rows.append({
                "protein": protein,
                "status": "accepted",
                "reason": "",
                "h5_path": str(h5_path),
                "n_h5_frames": int(n_h5),
                "n_ca": int(n_ca),
                "pair_hash": pair_hash,
                "accepted_seed_modes": ";".join(sorted(accepted_modes)),
                "n_mlcg_accepted": int(n_mlcg_ok),
                "h5_model": str(model_pkl),
                "pairs_csv": str(pairs_csv_path),
            })

    index_rows.sort(key=lambda r: (r.get("protein", "").lower(),
                                   r.get("seed_mode", ""), r.get("run_id", "")))
    summary_rows.sort(key=lambda r: r.get("protein", "").lower())

    index_csv = out_root / "tica_h5_comparison_index.csv"
    with index_csv.open("w", newline="") as f:
        fields = ["protein", "seed_mode", "run_id", "status", "reason",
                  "metadata_json", "mlcg_projection_csv"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in index_rows:
            w.writerow({k: r.get(k, "") for k in fields})

    summary_csv = out_root / "tica_h5_summary.csv"
    with summary_csv.open("w", newline="") as f:
        fields = ["protein", "status", "reason", "h5_path", "n_h5_frames",
                  "n_ca", "pair_hash", "accepted_seed_modes", "n_mlcg_accepted",
                  "h5_model", "pairs_csv"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in summary_rows:
            w.writerow({k: r.get(k, "") for k in fields})

    (out_root / "tica_h5_summary.json").write_text(
        json.dumps(summary_rows, indent=2, sort_keys=True)
    )
    (out_root / "tica_h5_comparison_index.json").write_text(
        json.dumps(index_rows, indent=2, sort_keys=True)
    )

    print(json.dumps({
        "mode": "experiment",
        "experiment_root": str(exp_root),
        "outdir": str(out_root),
        "n_proteins": len(summary_rows),
        "n_proteins_accepted": sum(1 for r in summary_rows if r.get("status") == "accepted"),
        "index_csv": str(index_csv),
        "summary_csv": str(summary_csv),
    }, indent=2))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    if args.experiment_root is not None:
        run_experiment(args)
    else:
        run_single(args)


if __name__ == "__main__":
    main()
