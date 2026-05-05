#!/usr/bin/env python3
"""Estimate conditional mean force μ(z) and covariance Σ(z) from mdCATH h5 data.

For each bin in 2-D TICA space we compute:
    μ(z)  = E[F_CA | z]   -- mean force the model should learn (per CA bead, per component)
    Σ(z)  = Cov[F_CA | z] -- thermal noise covariance at that CG state

where F_CA are the raw CA-projected all-atom forces from the h5 file and z is the
2-D TICA coordinate built from CA pair-distance features (same featurisation as
tica_from_h5.py so the models are directly comparable).

Outputs (in --outdir):
    force_noise_estimates.npz   -- arrays: mu, sigma, cov_diag, bin_centers,
                                   bin_counts, xedges, yedges, pairs
    force_noise_summary.csv     -- one row per occupied bin: bin coords, n_frames,
                                   mean_force_rms, mean_noise_std, signal_fraction
    force_noise_fes.png         -- 2-D FES (from h5 frames)
    force_noise_mu_rms.png      -- mean-force RMS per bin overlaid on TICA space
    force_noise_sigma_mean.png  -- mean per-component std overlaid on TICA space
    tica_model.pkl              -- fitted deeptime TICA model (for reuse)
    pairs.csv                   -- pair indices used for featurisation

Usage:
    python estimate_force_noise.py \
        --h5      structures/mdcath_dataset_4zohB01.h5 \
        --protein 4zohB01 \
        --outdir  outputs/exp_1pro_tica/force_noise_4zohB01 \
        --all-temp-groups          # use all available temperature groups
        [--temp-groups 320 348]    # or specify explicitly (overrides --all-temp-groups)
        [--tica-temp-group 320]    # temperature group(s) used only for TICA fitting
        [--n-pairs 200]
        [--pair-seed 42]
        [--lagtime 10]
        [--bins 40]
        [--temperature 320.0]      # kbT for FES weights (K)
        [--min-bin-frames 10]      # minimum frames per bin to compute Σ
        [--tica-model-pkl FILE]    # skip TICA fitting, load from existing pkl
        [--pairs-csv FILE]         # load pair indices from existing csv
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
from pathlib import Path
from typing import Optional

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Helpers shared with tica_from_h5.py  (keep implementations in sync)
# ---------------------------------------------------------------------------

def extract_ca_indices(pdb_protein_atoms: str) -> list[int]:
    """Return 0-based indices of CA atoms within the all-protein-atom coordinate array."""
    indices: list[int] = []
    atom_idx = 0
    for line in pdb_protein_atoms.splitlines():
        if line.startswith("ATOM"):
            if line[12:16].strip() == "CA":
                indices.append(atom_idx)
            atom_idx += 1
    return indices


def choose_pairs(n_atoms: int, n_pairs: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    max_pairs = n_atoms * (n_atoms - 1) // 2
    target = min(max(n_pairs, 1), max_pairs)
    selected: set[tuple[int, int]] = set()
    while len(selected) < target:
        i = int(rng.integers(0, n_atoms - 1))
        j = int(rng.integers(i + 1, n_atoms))
        selected.add((i, j))
    return np.asarray(sorted(selected), dtype=np.int64)


def build_pair_features(coords: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    """coords: (n_frames, n_ca, 3) -> features: (n_frames, n_pairs)"""
    diff = coords[:, pairs[:, 0], :] - coords[:, pairs[:, 1], :]
    return np.linalg.norm(diff, axis=-1).astype(np.float64)


def fit_tica(X: np.ndarray, lagtime: int):
    try:
        from deeptime.decomposition import TICA
    except ImportError as exc:
        raise RuntimeError("deeptime is required. Activate test_env_newsetup venv.") from exc
    if X.shape[0] <= lagtime:
        raise ValueError(f"Need n_frames > lagtime, got {X.shape[0]} <= {lagtime}")
    model = TICA(lagtime=lagtime, dim=2).fit(X).fetch_model()
    return model


def write_pairs_csv(path: Path, pairs: np.ndarray) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pair_index", "atom_i_1based", "atom_j_1based"])
        for k, (i, j) in enumerate(pairs):
            w.writerow([k, int(i + 1), int(j + 1)])


def load_pairs_csv(path: Path) -> np.ndarray:
    pairs: list[tuple[int, int]] = []
    with path.open("r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            i = int(row["atom_i_1based"]) - 1
            j = int(row["atom_j_1based"]) - 1
            pairs.append((min(i, j), max(i, j)))
    return np.asarray(pairs, dtype=np.int64)


# ---------------------------------------------------------------------------
# H5 loading
# ---------------------------------------------------------------------------

def get_temp_groups(h5_path: Path, protein: str) -> list[str]:
    with h5py.File(h5_path, "r") as f:
        prot = f[protein] if protein in f else f[next(iter(f.keys()))]
        return [k for k in prot.keys()
                if k not in ("pdb", "psf", "pdbProteinAtoms", "chain",
                              "element", "resid", "resname", "z")]


def load_ca_data(
    h5_path: Path,
    protein: str,
    temp_groups: list[str],
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Load CA coordinates and CA forces from h5 across specified temperature groups.

    Returns:
        coords:  (N, n_ca, 3)  CA positions in Angstrom
        forces:  (N, n_ca, 3)  raw CA-projected all-atom forces (kcal/mol/Å)
        run_frame_counts: frame count per (temp_group, run) for bookkeeping
    """
    with h5py.File(h5_path, "r") as f:
        prot = f[protein] if protein in f else f[next(iter(f.keys()))]
        pdb_str: str = prot["pdbProteinAtoms"][()].decode()
        ca_idx = extract_ca_indices(pdb_str)
        if not ca_idx:
            raise ValueError(f"No CA atoms found in pdbProteinAtoms for {protein}")

        coords_list: list[np.ndarray] = []
        forces_list: list[np.ndarray] = []
        run_frame_counts: list[int] = []

        for tg in temp_groups:
            if tg not in prot:
                print(f"  [warn] temp group {tg} not found in {protein}, skipping")
                continue
            tg_grp = prot[tg]
            for rk in sorted(tg_grp.keys(), key=int):
                run = tg_grp[rk]
                c_all = run["coords"][()]   # (n_frames, n_atoms, 3)
                f_all = run["forces"][()]   # (n_frames, n_atoms, 3)
                coords_list.append(c_all[:, ca_idx, :].astype(np.float64))
                forces_list.append(f_all[:, ca_idx, :].astype(np.float64))
                run_frame_counts.append(c_all.shape[0])

    if not coords_list:
        raise ValueError(f"No frames loaded for {protein} with temp_groups={temp_groups}")

    return (
        np.concatenate(coords_list, axis=0),
        np.concatenate(forces_list, axis=0),
        run_frame_counts,
    )


# ---------------------------------------------------------------------------
# Binning and per-bin statistics
# ---------------------------------------------------------------------------

def compute_bin_stats(
    Y: np.ndarray,            # (N, 2) TICA projections
    forces: np.ndarray,       # (N, n_ca, 3)
    xedges: np.ndarray,       # (n_bins+1,)
    yedges: np.ndarray,       # (n_bins+1,)
    min_frames: int = 5,
) -> dict:
    """Compute per-bin μ and Σ statistics of the CA forces.

    Returns a dict with arrays indexed by (ix, iy) bin position:
        mu         (nx, ny, n_ca, 3)   mean force per bin
        sigma      (nx, ny, n_ca, 3)   per-component std per bin
        n_frames   (nx, ny)            frame count per bin
        mu_rms     (nx, ny)            RMS of the mean force vector per bin
        sigma_mean (nx, ny)            mean per-component std per bin
        cov_trace  (nx, ny)            trace of per-bin force covariance (scalar)
    """
    N, n_ca, _ = forces.shape
    n_flat = n_ca * 3  # flattened force dimension
    nx = len(xedges) - 1
    ny = len(yedges) - 1

    mu        = np.full((nx, ny, n_ca, 3), np.nan)
    sigma     = np.full((nx, ny, n_ca, 3), np.nan)
    n_frames  = np.zeros((nx, ny), dtype=np.int64)
    mu_rms    = np.full((nx, ny), np.nan)
    sigma_mean= np.full((nx, ny), np.nan)
    cov_trace = np.full((nx, ny), np.nan)

    ix_all = np.searchsorted(xedges[1:-1], Y[:, 0])  # 0-based bin index
    iy_all = np.searchsorted(yedges[1:-1], Y[:, 1])

    for ix in range(nx):
        for iy in range(ny):
            mask = (ix_all == ix) & (iy_all == iy)
            n = int(mask.sum())
            n_frames[ix, iy] = n
            if n < min_frames:
                continue
            F_bin = forces[mask]           # (n, n_ca, 3)
            mu_bin = F_bin.mean(axis=0)    # (n_ca, 3)
            mu[ix, iy] = mu_bin
            mu_rms[ix, iy] = float(np.sqrt(np.mean(mu_bin ** 2)))

            F_flat = F_bin.reshape(n, n_flat)
            std_flat = F_flat.std(axis=0, ddof=1)
            sigma[ix, iy] = std_flat.reshape(n_ca, 3)
            sigma_mean[ix, iy] = float(std_flat.mean())

            if n >= n_flat + 2:
                # Full covariance trace only when we have enough frames
                cov_trace[ix, iy] = float(np.trace(np.cov(F_flat.T)))

    return {
        "mu": mu,
        "sigma": sigma,
        "n_frames": n_frames,
        "mu_rms": mu_rms,
        "sigma_mean": sigma_mean,
        "cov_trace": cov_trace,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _overlay_plot(
    xedges: np.ndarray,
    yedges: np.ndarray,
    values: np.ndarray,
    title: str,
    cbar_label: str,
    out_png: Path,
    vmax: Optional[float] = None,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    cmap = plt.colormaps["plasma"].copy()
    cmap.set_bad("white")
    m = ax.pcolormesh(
        xedges, yedges, np.ma.masked_invalid(values).T,
        cmap=cmap, shading="flat", vmin=0, vmax=vmax,
    )
    ax.set_xlabel("TIC 1")
    ax.set_ylabel("TIC 2")
    ax.set_title(title)
    cbar = fig.colorbar(m, ax=ax)
    cbar.set_label(cbar_label)
    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_fes(
    Y: np.ndarray,
    xedges: np.ndarray,
    yedges: np.ndarray,
    temperature: float,
    out_png: Path,
    title: str = "TICA FES",
) -> None:
    H, _, _ = np.histogram2d(Y[:, 0], Y[:, 1], bins=[xedges, yedges])
    occupied = H > 0
    P = np.zeros_like(H, dtype=np.float64)
    P[occupied] = H[occupied] / H.sum()
    kB = 0.00831446261815324  # kJ/mol/K
    F = np.full_like(P, np.nan)
    F[occupied] = -kB * temperature * np.log(P[occupied])
    F[occupied] -= np.nanmin(F[occupied])
    fig, ax = plt.subplots(figsize=(6, 5))
    cmap = plt.colormaps["turbo"].copy()
    cmap.set_bad("white")
    m = ax.pcolormesh(xedges, yedges, np.ma.masked_invalid(F).T,
                      cmap=cmap, shading="flat", vmin=0)
    ax.set_xlabel("TIC 1")
    ax.set_ylabel("TIC 2")
    ax.set_title(title)
    fig.colorbar(m, ax=ax, label="F [kJ/mol]")
    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Summary CSV
# ---------------------------------------------------------------------------

def write_summary_csv(
    path: Path,
    xedges: np.ndarray,
    yedges: np.ndarray,
    stats: dict,
) -> None:
    nx, ny = stats["n_frames"].shape
    xc = 0.5 * (xedges[:-1] + xedges[1:])
    yc = 0.5 * (yedges[:-1] + yedges[1:])

    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "bin_ix", "bin_iy",
            "tic1_center", "tic2_center",
            "n_frames",
            "mu_rms_kcal_mol_A",
            "sigma_mean_kcal_mol_A",
            "cov_trace",
            "signal_fraction",
        ])
        for ix in range(nx):
            for iy in range(ny):
                n = int(stats["n_frames"][ix, iy])
                if n == 0:
                    continue
                mu_r  = stats["mu_rms"][ix, iy]
                sig   = stats["sigma_mean"][ix, iy]
                ct    = stats["cov_trace"][ix, iy]
                # signal fraction: variance of mean force / (variance of mean force + noise variance)
                # approximate scalar version: mu_rms^2 / (mu_rms^2 + sigma_mean^2)
                if np.isfinite(mu_r) and np.isfinite(sig) and (mu_r ** 2 + sig ** 2) > 0:
                    sf = mu_r ** 2 / (mu_r ** 2 + sig ** 2)
                else:
                    sf = float("nan")
                w.writerow([
                    ix, iy,
                    f"{xc[ix]:.4f}", f"{yc[iy]:.4f}",
                    n,
                    f"{mu_r:.4f}" if np.isfinite(mu_r) else "",
                    f"{sig:.4f}"  if np.isfinite(sig)  else "",
                    f"{ct:.4f}"   if np.isfinite(ct)   else "",
                    f"{sf:.4f}"   if np.isfinite(sf)   else "",
                ])


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--h5",       type=Path, required=True,
                   help="mdCATH h5 file for the target protein")
    p.add_argument("--protein",  type=str,  required=True,
                   help="Protein key inside the h5 file (e.g. 4zohB01)")
    p.add_argument("--outdir",   type=Path, required=True,
                   help="Output directory")

    # Temperature group selection
    p.add_argument("--temp-groups", nargs="+", default=None,
                   help="Temperature group(s) to use for force noise estimation "
                        "(e.g. 320 348 379). Overrides --all-temp-groups.")
    p.add_argument("--all-temp-groups", action="store_true",
                   help="Use all temperature groups present in the h5 file.")
    p.add_argument("--tica-temp-groups", nargs="+", default=None,
                   help="Temperature group(s) for TICA fitting only. "
                        "Defaults to same as --temp-groups / --all-temp-groups.")

    # Featurisation
    p.add_argument("--n-pairs",   type=int, default=200)
    p.add_argument("--pair-seed", type=int, default=42)
    p.add_argument("--lagtime",   type=int, default=10,
                   help="TICA lagtime in frames")

    # Binning and analysis
    p.add_argument("--bins",           type=int,   default=40,
                   help="TICA histogram bins per axis")
    p.add_argument("--temperature",    type=float, default=320.0,
                   help="Temperature for FES Boltzmann weights (K)")
    p.add_argument("--min-bin-frames", type=int,   default=10,
                   help="Minimum frames per bin to compute noise statistics")

    # Reuse existing TICA model / pairs
    p.add_argument("--tica-model-pkl", type=Path, default=None,
                   help="Skip TICA fitting; load pre-fitted model from pkl.")
    p.add_argument("--pairs-csv",      type=Path, default=None,
                   help="Load pair indices from existing pairs.csv.")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Determine temperature groups
    # -----------------------------------------------------------------------
    available_tgs = get_temp_groups(args.h5, args.protein)
    print(f"[noise] protein={args.protein}  available temp groups: {available_tgs}")

    if args.temp_groups is not None:
        force_tgs = args.temp_groups
    elif args.all_temp_groups:
        force_tgs = available_tgs
    else:
        force_tgs = ["320"]
        print(f"[noise] defaulting to temp_groups={force_tgs}; use --all-temp-groups for all")

    tica_tgs = args.tica_temp_groups if args.tica_temp_groups is not None else force_tgs
    print(f"[noise] force noise temp groups : {force_tgs}")
    print(f"[noise] TICA fitting temp groups: {tica_tgs}")

    # -----------------------------------------------------------------------
    # Load CA data for TICA fitting
    # -----------------------------------------------------------------------
    print(f"[noise] Loading CA coords for TICA from {tica_tgs} ...")
    tica_coords, _, tica_run_counts = load_ca_data(args.h5, args.protein, tica_tgs)
    n_tica, n_ca, _ = tica_coords.shape
    print(f"[noise]   TICA frames: {n_tica}  CA atoms: {n_ca}")

    # -----------------------------------------------------------------------
    # Pair indices
    # -----------------------------------------------------------------------
    pairs_csv_path = args.outdir / "pairs.csv"
    if args.pairs_csv is not None and args.pairs_csv.exists():
        pairs = load_pairs_csv(args.pairs_csv)
        print(f"[noise] Loaded {len(pairs)} pairs from {args.pairs_csv}")
    else:
        pairs = choose_pairs(n_ca, args.n_pairs, args.pair_seed)
        write_pairs_csv(pairs_csv_path, pairs)
        print(f"[noise] Generated {len(pairs)} pairs (seed={args.pair_seed})")

    # -----------------------------------------------------------------------
    # TICA model
    # -----------------------------------------------------------------------
    tica_model_path = args.outdir / "tica_model.pkl"
    X_tica = build_pair_features(tica_coords, pairs)

    if args.tica_model_pkl is not None and args.tica_model_pkl.exists():
        with args.tica_model_pkl.open("rb") as fh:
            tica_model = pickle.load(fh)
        print(f"[noise] Loaded TICA model from {args.tica_model_pkl}")
    else:
        print(f"[noise] Fitting TICA (lagtime={args.lagtime}) on {n_tica} frames ...")
        tica_model = fit_tica(X_tica, args.lagtime)
        with tica_model_path.open("wb") as fh:
            pickle.dump(tica_model, fh)
        print(f"[noise] TICA model saved to {tica_model_path}")

    Y_tica = np.asarray(tica_model.transform(X_tica), dtype=np.float64)

    # -----------------------------------------------------------------------
    # Load CA data for force noise estimation (may overlap with tica_tgs)
    # -----------------------------------------------------------------------
    if set(force_tgs) == set(tica_tgs):
        force_coords = tica_coords
        force_forces: np.ndarray
        _, force_forces, force_run_counts = load_ca_data(
            args.h5, args.protein, force_tgs
        )
        Y_force = Y_tica
        print(f"[noise] Reusing TICA frames for force noise (same temp groups)")
    else:
        print(f"[noise] Loading CA data for force noise from {force_tgs} ...")
        force_coords, force_forces, force_run_counts = load_ca_data(
            args.h5, args.protein, force_tgs
        )
        X_force = build_pair_features(force_coords, pairs)
        Y_force = np.asarray(tica_model.transform(X_force), dtype=np.float64)

    n_force = force_coords.shape[0]
    print(f"[noise] Force noise frames: {n_force}")
    print(f"[noise]   Raw CA force RMS: {np.sqrt(np.mean(force_forces ** 2)):.3f} kcal/mol/Å")

    # -----------------------------------------------------------------------
    # 2D TICA histogram edges (from TICA reference data)
    # -----------------------------------------------------------------------
    _, xedges, yedges = np.histogram2d(
        Y_tica[:, 0], Y_tica[:, 1], bins=args.bins
    )

    # -----------------------------------------------------------------------
    # Per-bin force statistics
    # -----------------------------------------------------------------------
    print(f"[noise] Computing per-bin statistics (bins={args.bins}×{args.bins}, "
          f"min_frames={args.min_bin_frames}) ...")
    stats = compute_bin_stats(
        Y_force, force_forces, xedges, yedges, min_frames=args.min_bin_frames
    )

    n_occupied = int((stats["n_frames"] >= args.min_bin_frames).sum())
    print(f"[noise]   Occupied bins (>= {args.min_bin_frames} frames): "
          f"{n_occupied} / {args.bins ** 2}")

    mu_rms_vals = stats["mu_rms"][np.isfinite(stats["mu_rms"])]
    sig_vals    = stats["sigma_mean"][np.isfinite(stats["sigma_mean"])]
    if len(mu_rms_vals) > 0:
        print(f"[noise]   μ RMS across bins: "
              f"min={mu_rms_vals.min():.2f}  median={np.median(mu_rms_vals):.2f}  "
              f"max={mu_rms_vals.max():.2f} kcal/mol/Å")
        print(f"[noise]   σ mean across bins: "
              f"min={sig_vals.min():.2f}  median={np.median(sig_vals):.2f}  "
              f"max={sig_vals.max():.2f} kcal/mol/Å")

    # -----------------------------------------------------------------------
    # Save arrays
    # -----------------------------------------------------------------------
    npz_path = args.outdir / "force_noise_estimates.npz"
    np.savez(
        npz_path,
        mu=stats["mu"],                  # (nx, ny, n_ca, 3)
        sigma=stats["sigma"],            # (nx, ny, n_ca, 3)
        n_frames=stats["n_frames"],      # (nx, ny)
        mu_rms=stats["mu_rms"],          # (nx, ny)
        sigma_mean=stats["sigma_mean"],  # (nx, ny)
        cov_trace=stats["cov_trace"],    # (nx, ny)
        xedges=xedges,
        yedges=yedges,
        Y_tica=Y_tica,
        Y_force=Y_force,
        pairs=pairs,
    )
    print(f"[noise] Saved arrays to {npz_path}")

    # -----------------------------------------------------------------------
    # Summary CSV
    # -----------------------------------------------------------------------
    csv_path = args.outdir / "force_noise_summary.csv"
    write_summary_csv(csv_path, xedges, yedges, stats)
    print(f"[noise] Saved summary CSV to {csv_path}")

    # -----------------------------------------------------------------------
    # Plots
    # -----------------------------------------------------------------------
    # FES from h5 reference frames
    plot_fes(
        Y_tica, xedges, yedges,
        temperature=args.temperature,
        out_png=args.outdir / "force_noise_fes.png",
        title=f"{args.protein} h5 reference FES ({n_tica} frames, {tica_tgs})",
    )

    # Mean-force RMS per bin
    _overlay_plot(
        xedges, yedges, stats["mu_rms"],
        title=f"{args.protein}  ||μ(z)|| RMS per bin",
        cbar_label="‖μ‖ RMS [kcal/mol/Å]",
        out_png=args.outdir / "force_noise_mu_rms.png",
    )

    # Mean per-component noise std per bin
    _overlay_plot(
        xedges, yedges, stats["sigma_mean"],
        title=f"{args.protein}  mean σ(z) per bin",
        cbar_label="mean σ [kcal/mol/Å]",
        out_png=args.outdir / "force_noise_sigma_mean.png",
    )

    # Signal fraction: mu_rms^2 / (mu_rms^2 + sigma_mean^2)
    with np.errstate(invalid="ignore", divide="ignore"):
        signal_frac = (stats["mu_rms"] ** 2 /
                       (stats["mu_rms"] ** 2 + stats["sigma_mean"] ** 2))
    _overlay_plot(
        xedges, yedges, signal_frac,
        title=f"{args.protein}  signal fraction = ||μ||² / (||μ||² + σ²)",
        cbar_label="signal fraction",
        out_png=args.outdir / "force_noise_signal_fraction.png",
        vmax=1.0,
    )

    # -----------------------------------------------------------------------
    # Metadata JSON
    # -----------------------------------------------------------------------
    meta = {
        "protein": args.protein,
        "h5": str(args.h5.resolve()),
        "force_temp_groups": force_tgs,
        "tica_temp_groups": tica_tgs,
        "n_tica_frames": int(n_tica),
        "n_force_frames": int(n_force),
        "n_ca": int(n_ca),
        "n_pairs": int(len(pairs)),
        "lagtime": int(args.lagtime),
        "bins": int(args.bins),
        "temperature_K": float(args.temperature),
        "min_bin_frames": int(args.min_bin_frames),
        "n_occupied_bins": int(n_occupied),
        "mu_rms_median_kcal_mol_A": (
            float(np.median(mu_rms_vals)) if len(mu_rms_vals) > 0 else None
        ),
        "sigma_mean_median_kcal_mol_A": (
            float(np.median(sig_vals)) if len(sig_vals) > 0 else None
        ),
        "raw_ca_force_rms_kcal_mol_A": float(
            np.sqrt(np.mean(force_forces ** 2))
        ),
        "outputs": {
            "npz": str(npz_path),
            "csv": str(csv_path),
            "tica_model_pkl": str(tica_model_path),
            "pairs_csv": str(pairs_csv_path),
        },
    }
    meta_path = args.outdir / "force_noise_metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"[noise] Metadata saved to {meta_path}")
    print(f"[noise] Done. All outputs in {args.outdir}")


if __name__ == "__main__":
    main()
