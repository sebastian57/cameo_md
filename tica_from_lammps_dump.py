#!/usr/bin/env python3
"""Run TICA on LAMMPS dump trajectories.

Supports two modes:
1) legacy single-run mode (explicit --dump)
2) experiment mode (manifest-driven via --experiment-root)
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
import re
from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)

    # Legacy mode
    p.add_argument("--dump", type=Path, default=None, help="LAMMPS dump file (legacy mode)")
    p.add_argument("--outdir", type=Path, required=True, help="Output directory")
    p.add_argument("--prefix", type=str, default="tica", help="Output filename prefix (legacy mode)")

    # Shared numerics
    p.add_argument("--lagtime", type=int, default=10, help="TICA lagtime in frames")
    p.add_argument("--bins", type=int, default=80, help="Histogram bins per axis")
    p.add_argument("--temperature", type=float, default=320.0, help="Temperature for FES in K")
    p.add_argument("--max-frames", type=int, default=0, help="If >0, keep at most this many frames")
    p.add_argument("--frame-stride", type=int, default=1, help="Keep every Nth parsed frame")
    p.add_argument("--pair-mode", choices=("random", "sequential"), default="random")
    p.add_argument("--n-pairs", type=int, default=200, help="Number of bead pairs used as features")
    p.add_argument("--pair-seed", type=int, default=42, help="RNG seed for random pair selection")
    p.add_argument("--standardize", action="store_true", help="Standardize distance features before TICA")

    # Legacy projection mode
    p.add_argument(
        "--reference-model",
        type=Path,
        default=None,
        help="Project onto this prefit TICA model (legacy mode)",
    )
    p.add_argument(
        "--reference-pairs",
        type=Path,
        default=None,
        help="CSV pair list used by reference model (legacy mode)",
    )

    # Experiment mode
    p.add_argument("--experiment-root", type=Path, default=None, help="outputs/<experiment_name>")
    p.add_argument("--mlcg-set", choices=("both", "t0", "eq"), default="both")
    p.add_argument("--match-length", choices=("min", "classical"), default="min")
    p.add_argument(
        "--strict-shape",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require equal atom count/order between classical and MLCG trajectories",
    )
    p.add_argument(
        "--include-incomplete",
        action="store_true",
        help="Include runs even if final timestep < sample_steps when required files exist",
    )
    return p.parse_args()


def _choose_coord_columns(atom_header: list[str]) -> tuple[str, str, str]:
    x_col = "xu" if "xu" in atom_header else "x"
    y_col = "yu" if "yu" in atom_header else "y"
    z_col = "zu" if "zu" in atom_header else "z"
    for c in (x_col, y_col, z_col):
        if c not in atom_header:
            raise ValueError(f"Missing coordinate column '{c}' in dump header: {atom_header}")
    return x_col, y_col, z_col


def parse_lammps_dump_coords(path: Path, frame_stride: int = 1, max_frames: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Parse coordinates from LAMMPS custom dump.

    Returns:
        coords: (n_frames, n_atoms, 3)
        timesteps: (n_frames,)
    """
    coords_frames: list[np.ndarray] = []
    timesteps: list[int] = []

    with path.open("r") as fh:
        parsed_idx = 0
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
            fh.readline()
            fh.readline()
            fh.readline()

            atom_line = fh.readline().strip()
            if not atom_line.startswith("ITEM: ATOMS"):
                continue
            atom_header = atom_line.split()[2:]

            if "id" not in atom_header:
                raise ValueError("Dump must include atom id column")
            id_idx = atom_header.index("id")
            x_col, y_col, z_col = _choose_coord_columns(atom_header)
            x_idx = atom_header.index(x_col)
            y_idx = atom_header.index(y_col)
            z_idx = atom_header.index(z_col)

            raw_rows: list[tuple[int, float, float, float]] = []
            ok = True
            for _ in range(n_atoms):
                row = fh.readline()
                if not row:
                    ok = False
                    break
                cols = row.split()
                try:
                    raw_rows.append((
                        int(cols[id_idx]),
                        float(cols[x_idx]),
                        float(cols[y_idx]),
                        float(cols[z_idx]),
                    ))
                except Exception:
                    ok = False
                    break

            if not ok:
                continue

            raw_rows.sort(key=lambda r: r[0])
            frame = np.empty((n_atoms, 3), dtype=np.float64)
            for slot, (_, x, y, z) in enumerate(raw_rows):
                frame[slot, 0] = x
                frame[slot, 1] = y
                frame[slot, 2] = z

            take = parsed_idx % max(frame_stride, 1) == 0
            if take:
                coords_frames.append(frame)
                timesteps.append(timestep)
                if max_frames > 0 and len(coords_frames) >= max_frames:
                    break
            parsed_idx += 1

    if not coords_frames:
        raise ValueError(f"No complete frames parsed from {path}")

    return np.stack(coords_frames, axis=0), np.asarray(timesteps, dtype=np.int64)


def choose_pairs(n_atoms: int, n_pairs: int, mode: str, seed: int) -> np.ndarray:
    if n_atoms < 2:
        raise ValueError("Need at least 2 atoms for pair-distance features")

    if mode == "sequential":
        pairs = np.array([(i, i + 1) for i in range(n_atoms - 1)], dtype=np.int64)
        return pairs[: min(n_pairs, len(pairs))] if n_pairs > 0 else pairs

    rng = np.random.default_rng(seed)
    max_pairs = n_atoms * (n_atoms - 1) // 2
    target = min(max(n_pairs, 1), max_pairs)

    selected = set()
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
            if i < 0 or j < 0:
                raise ValueError(f"Invalid 1-based pair indices in {path}: {row}")
            if i == j:
                raise ValueError(f"Degenerate pair in {path}: {row}")
            if i > j:
                i, j = j, i
            pairs.append((i, j))
    if not pairs:
        raise ValueError(f"No pairs found in {path}")
    return np.asarray(pairs, dtype=np.int64)


def build_pair_distance_features(coords: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    i_idx = pairs[:, 0]
    j_idx = pairs[:, 1]
    diff = coords[:, i_idx, :] - coords[:, j_idx, :]
    X = np.linalg.norm(diff, axis=-1)
    return X.astype(np.float64)


def fit_tica(X: np.ndarray, lagtime: int) -> tuple[object, np.ndarray]:
    try:
        from deeptime.decomposition import TICA
    except Exception as exc:
        raise RuntimeError("deeptime is required for TICA. Use test_env_newsetup venv.") from exc

    if lagtime < 1:
        raise ValueError("lagtime must be >= 1")
    if X.shape[0] <= lagtime:
        raise ValueError(f"Need n_frames > lagtime, got n_frames={X.shape[0]}, lagtime={lagtime}")

    tica_model = TICA(lagtime=lagtime, dim=2).fit(X).fetch_model()
    Y = np.asarray(tica_model.transform(X), dtype=np.float64)
    if Y.ndim != 2 or Y.shape[1] < 2:
        raise ValueError(f"Unexpected TICA output shape: {Y.shape}")
    return tica_model, Y


def project_tica(X: np.ndarray, model_path: Path) -> tuple[object, np.ndarray]:
    with model_path.open("rb") as f:
        tica_model = pickle.load(f)
    Y = np.asarray(tica_model.transform(X), dtype=np.float64)
    if Y.ndim != 2 or Y.shape[1] < 2:
        raise ValueError(f"Unexpected projected TICA output shape: {Y.shape}")
    return tica_model, Y


def compute_fes_2d(Y: np.ndarray, bins: int, temperature: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = Y[:, 0]
    y = Y[:, 1]
    H, xedges, yedges = np.histogram2d(x, y, bins=bins)
    occupied = H > 0

    P = np.zeros_like(H, dtype=np.float64)
    P[occupied] = H[occupied] / np.sum(H)

    kB = 0.00831446261815324  # kJ / mol / K
    F = np.full_like(P, np.nan, dtype=np.float64)
    F[occupied] = -kB * temperature * np.log(P[occupied])
    F[occupied] -= np.nanmin(F[occupied])
    return F, xedges, yedges


def plot_fes(F: np.ndarray, xedges: np.ndarray, yedges: np.ndarray, out_png: Path, title: str = "TICA Free Energy Surface") -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    cmap = plt.colormaps["turbo"].copy()
    cmap.set_bad("white")
    m = ax.pcolormesh(xedges, yedges, np.ma.masked_invalid(F).T, cmap=cmap, shading="flat")
    ax.set_xlabel("TIC 1")
    ax.set_ylabel("TIC 2")
    ax.set_title(title)
    cbar = fig.colorbar(m, ax=ax)
    cbar.set_label("F [kJ/mol]")
    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def parse_last_timestep(path: Path) -> int | None:
    if not path.exists():
        return None
    last = None
    with path.open() as fh:
        for line in fh:
            if line.startswith("ITEM: TIMESTEP"):
                try:
                    last = int(next(fh).strip())
                except Exception:
                    pass
    return last


def parse_int_var_from_input(path: Path, key: str) -> int | None:
    if not path.exists():
        return None
    txt = path.read_text()
    m = re.search(rf"^\s*variable\s+{re.escape(key)}\s+equal\s+([0-9]+)\s*$", txt, re.M)
    return int(m.group(1)) if m else None


def parse_manifest_rows(path: Path, expected_cols: int) -> list[list[str]]:
    rows: list[list[str]] = []
    with path.open() as f:
        for raw in f:
            # Keep trailing tab-separated empty fields (e.g. blank "reason" col).
            raw = raw.rstrip("\n")
            if not raw.strip():
                continue
            cols = raw.split("\t")
            if len(cols) < expected_cols:
                continue
            rows.append(cols)
    return rows


def stable_hash_pairs(pairs: np.ndarray) -> str:
    payload = ";".join(f"{int(i)}-{int(j)}" for i, j in pairs.tolist()).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def write_projection_csv(path: Path, timesteps: np.ndarray, Y: np.ndarray) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame_index", "timestep", "tic1", "tic2"])
        for i in range(Y.shape[0]):
            w.writerow([i, int(timesteps[i]), float(Y[i, 0]), float(Y[i, 1])])


def write_pairs_csv(path: Path, pairs: np.ndarray) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pair_index", "atom_i_1based", "atom_j_1based"])
        for k, (i, j) in enumerate(pairs):
            w.writerow([k, int(i + 1), int(j + 1)])


def run_single_mode(args: argparse.Namespace) -> None:
    if args.dump is None:
        raise SystemExit("Single mode requires --dump (or use --experiment-root mode)")

    args.outdir.mkdir(parents=True, exist_ok=True)

    project_mode = args.reference_model is not None
    if project_mode:
        if args.reference_pairs is None:
            raise ValueError("--reference-pairs is required when --reference-model is provided")
        if args.standardize:
            raise ValueError("--standardize is not supported in projection mode")

    coords, timesteps = parse_lammps_dump_coords(
        args.dump,
        frame_stride=args.frame_stride,
        max_frames=args.max_frames,
    )

    n_frames, n_atoms, _ = coords.shape

    if project_mode:
        pairs = load_pairs_csv(args.reference_pairs)
        if int(np.max(pairs)) >= n_atoms:
            raise ValueError(
                f"Reference pair indices require atom index {int(np.max(pairs)) + 1}, "
                f"but dump has only {n_atoms} atoms."
            )
    else:
        pairs = choose_pairs(n_atoms=n_atoms, n_pairs=args.n_pairs, mode=args.pair_mode, seed=args.pair_seed)

    X = build_pair_distance_features(coords, pairs)

    if args.standardize:
        mu = X.mean(axis=0, keepdims=True)
        sigma = X.std(axis=0, keepdims=True)
        sigma[sigma < 1e-12] = 1.0
        X = (X - mu) / sigma

    if project_mode:
        tica_model, Y = project_tica(X, model_path=args.reference_model)
    else:
        tica_model, Y = fit_tica(X, lagtime=args.lagtime)

    F, xedges, yedges = compute_fes_2d(Y, bins=args.bins, temperature=args.temperature)

    prefix = args.prefix
    png_path = args.outdir / f"{prefix}_tic_fes.png"
    proj_csv = args.outdir / f"{prefix}_tica_projection.csv"
    pairs_csv = args.outdir / f"{prefix}_pair_indices.csv"
    model_pkl = args.outdir / f"{prefix}_tica_model.pkl"
    fes_npz = args.outdir / f"{prefix}_fes_grid.npz"
    meta_json = args.outdir / f"{prefix}_metadata.json"

    plot_fes(F, xedges, yedges, png_path)
    write_projection_csv(proj_csv, timesteps, Y)
    write_pairs_csv(pairs_csv, pairs)

    if not project_mode:
        with model_pkl.open("wb") as f:
            pickle.dump(tica_model, f)

    np.savez(fes_npz, F=F, xedges=xedges, yedges=yedges)

    metadata = {
        "mode": "project" if project_mode else "fit",
        "dump": str(args.dump.resolve()),
        "n_frames": int(n_frames),
        "n_atoms": int(n_atoms),
        "n_pairs": int(pairs.shape[0]),
        "pair_mode": "from_reference_pairs" if project_mode else args.pair_mode,
        "lagtime": None if project_mode else int(args.lagtime),
        "bins": int(args.bins),
        "temperature": float(args.temperature),
        "standardize": bool(args.standardize),
        "reference_model": str(args.reference_model.resolve()) if project_mode else None,
        "reference_pairs": str(args.reference_pairs.resolve()) if project_mode else None,
        "coord_source": "auto (prefers xu/yu/zu, falls back to x/y/z)",
        "outputs": {
            "fes_png": str(png_path),
            "tica_projection_csv": str(proj_csv),
            "pairs_csv": str(pairs_csv),
            "tica_model_pkl": str(model_pkl) if not project_mode else None,
            "fes_grid_npz": str(fes_npz),
        },
    }
    meta_json.write_text(json.dumps(metadata, indent=2, sort_keys=True))

    print(f"Parsed frames: {n_frames}, atoms/frame: {n_atoms}")
    print(f"Mode: {'projection' if project_mode else 'fit'}")
    print(f"Using {pairs.shape[0]} pair-distance features")
    print(f"Saved FES plot: {png_path}")
    print(f"Saved metadata: {meta_json}")


def _seed_modes_from_arg(v: str) -> list[str]:
    return ["t0", "eq"] if v == "both" else [v]


def _iter_by_protein(rows: Iterable[dict]) -> list[str]:
    return sorted({str(r["protein"]) for r in rows}, key=str.lower)


def run_experiment_mode(args: argparse.Namespace) -> None:
    exp_root = args.experiment_root.resolve()
    out_root = args.outdir.resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    free_manifest = exp_root / "classical_free_manifest.tsv"
    if not free_manifest.exists():
        raise SystemExit(f"Missing classical free manifest: {free_manifest}")

    mlcg_manifest_candidates = [
        exp_root / "mlcg" / "mlcg_manifest.tsv",
        exp_root / "mlcg_manifest.tsv",
    ]
    mlcg_manifest = next((p for p in mlcg_manifest_candidates if p.exists()), None)
    if mlcg_manifest is None:
        raise SystemExit(
            "Missing MLCG manifest. Expected one of: "
            + ", ".join(str(p) for p in mlcg_manifest_candidates)
        )

    free_rows_raw = parse_manifest_rows(free_manifest, expected_cols=5)
    free_rows: list[dict] = []
    for cols in free_rows_raw:
        run_id, protein, mode, run_dir, input_file = cols[:5]
        if mode != "free":
            continue
        free_rows.append(
            {
                "run_id": run_id,
                "protein": protein,
                "run_dir": Path(run_dir),
                "input_file": Path(input_file),
            }
        )

    mlcg_rows_raw = parse_manifest_rows(mlcg_manifest, expected_cols=9)
    mlcg_rows: list[dict] = []
    for cols in mlcg_rows_raw:
        run_id, protein, seed_mode, source_run_id, source_run_dir, run_dir, input_file, status, reason = cols[:9]
        if seed_mode not in ("t0", "eq"):
            continue
        mlcg_rows.append(
            {
                "run_id": run_id,
                "protein": protein,
                "seed_mode": seed_mode,
                "source_run_id": source_run_id,
                "source_run_dir": Path(source_run_dir),
                "run_dir": Path(run_dir),
                "input_file": Path(input_file),
                "status": status,
                "reason": reason,
            }
        )

    selected_seed_modes = set(_seed_modes_from_arg(args.mlcg_set))

    index_rows: list[dict] = []
    summary_rows: list[dict] = []

    for protein in _iter_by_protein(free_rows):
        protein_out = out_root / protein
        protein_out.mkdir(parents=True, exist_ok=True)

        row_free = next((r for r in free_rows if r["protein"].lower() == protein.lower()), None)
        if row_free is None:
            summary_rows.append({"protein": protein, "status": "rejected", "reason": "missing free classical row"})
            continue

        classical_dump = row_free["run_dir"] / "ca_tica.dump"
        classical_steps = parse_int_var_from_input(row_free["input_file"], "sample_steps")
        classical_last = parse_last_timestep(classical_dump)

        classical_complete = (
            classical_dump.exists()
            and classical_steps is not None
            and classical_last is not None
            and classical_last >= classical_steps
        )
        if not classical_complete and not args.include_incomplete:
            summary_rows.append(
                {
                    "protein": protein,
                    "status": "rejected",
                    "reason": "classical free trajectory incomplete",
                }
            )
            continue

        mlcg_for_protein = [
            r for r in mlcg_rows
            if r["protein"].lower() == protein.lower()
            and r["seed_mode"] in selected_seed_modes
            and r["status"] == "ready"
        ]
        if not mlcg_for_protein:
            summary_rows.append({"protein": protein, "status": "rejected", "reason": "no ready MLCG runs"})
            continue

        try:
            classical_coords, classical_ts = parse_lammps_dump_coords(
                classical_dump,
                frame_stride=args.frame_stride,
                max_frames=args.max_frames,
            )
        except Exception as exc:  # noqa: BLE001
            summary_rows.append({"protein": protein, "status": "rejected", "reason": f"classical parse error: {exc}"})
            continue

        n_classical_frames, n_atoms, _ = classical_coords.shape
        pairs = choose_pairs(n_atoms=n_atoms, n_pairs=args.n_pairs, mode=args.pair_mode, seed=args.pair_seed)
        pair_hash = stable_hash_pairs(pairs)

        X_classical = build_pair_distance_features(classical_coords, pairs)
        if args.standardize:
            mu = X_classical.mean(axis=0, keepdims=True)
            sigma = X_classical.std(axis=0, keepdims=True)
            sigma[sigma < 1e-12] = 1.0
            X_classical = (X_classical - mu) / sigma
        else:
            mu = None
            sigma = None

        try:
            tica_model, Y_classical = fit_tica(X_classical, lagtime=args.lagtime)
        except Exception as exc:  # noqa: BLE001
            summary_rows.append({"protein": protein, "status": "rejected", "reason": f"TICA fit error: {exc}"})
            continue

        model_pkl = protein_out / f"tica_{protein}_classical_model.pkl"
        pairs_csv = protein_out / f"tica_{protein}_pairs.csv"
        class_proj_csv = protein_out / f"tica_{protein}_classical_projection.csv"
        class_fes_npz = protein_out / f"tica_{protein}_classical_fes.npz"
        class_fes_png = protein_out / f"tica_{protein}_classical_fes.png"

        with model_pkl.open("wb") as f:
            pickle.dump(tica_model, f)
        write_pairs_csv(pairs_csv, pairs)
        write_projection_csv(class_proj_csv, classical_ts, Y_classical)
        F_cls, x_cls, y_cls = compute_fes_2d(Y_classical, bins=args.bins, temperature=args.temperature)
        np.savez(class_fes_npz, F=F_cls, xedges=x_cls, yedges=y_cls)
        plot_fes(F_cls, x_cls, y_cls, class_fes_png, title=f"{protein} classical")

        accepted_modes: set[str] = set()
        n_mlcg_ok = 0

        for ml_row in sorted(mlcg_for_protein, key=lambda r: (r["seed_mode"], r["run_id"])):
            seed_mode = str(ml_row["seed_mode"])
            run_id = str(ml_row["run_id"])
            ml_dump = ml_row["run_dir"] / "ca_tica.dump"
            ml_steps = parse_int_var_from_input(ml_row["input_file"], "sample_steps")
            ml_last = parse_last_timestep(ml_dump)

            ml_complete = (
                ml_dump.exists()
                and ml_steps is not None
                and ml_last is not None
                and ml_last >= ml_steps
            )
            if not ml_complete and not args.include_incomplete:
                index_rows.append(
                    {
                        "protein": protein,
                        "seed_mode": seed_mode,
                        "run_id": run_id,
                        "status": "rejected",
                        "reason": "mlcg trajectory incomplete",
                    }
                )
                continue

            try:
                ml_coords, ml_ts = parse_lammps_dump_coords(
                    ml_dump,
                    frame_stride=args.frame_stride,
                    max_frames=args.max_frames,
                )
            except Exception as exc:  # noqa: BLE001
                index_rows.append(
                    {
                        "protein": protein,
                        "seed_mode": seed_mode,
                        "run_id": run_id,
                        "status": "rejected",
                        "reason": f"mlcg parse error: {exc}",
                    }
                )
                continue

            if args.strict_shape and ml_coords.shape[1] != n_atoms:
                index_rows.append(
                    {
                        "protein": protein,
                        "seed_mode": seed_mode,
                        "run_id": run_id,
                        "status": "rejected",
                        "reason": f"atom count mismatch classical={n_atoms} mlcg={ml_coords.shape[1]}",
                    }
                )
                continue

            if args.match_length == "min":
                n_use = min(ml_coords.shape[0], classical_coords.shape[0])
            else:
                n_use = classical_coords.shape[0]
                if ml_coords.shape[0] < n_use:
                    index_rows.append(
                        {
                            "protein": protein,
                            "seed_mode": seed_mode,
                            "run_id": run_id,
                            "status": "rejected",
                            "reason": "mlcg shorter than classical for match-length=classical",
                        }
                    )
                    continue

            cls_coords_use = classical_coords[:n_use]
            cls_ts_use = classical_ts[:n_use]
            ml_coords_use = ml_coords[:n_use]
            ml_ts_use = ml_ts[:n_use]

            X_cls_use = build_pair_distance_features(cls_coords_use, pairs)
            X_ml_use = build_pair_distance_features(ml_coords_use, pairs)
            if args.standardize and mu is not None and sigma is not None:
                X_cls_use = (X_cls_use - mu) / sigma
                X_ml_use = (X_ml_use - mu) / sigma

            Y_cls_use = np.asarray(tica_model.transform(X_cls_use), dtype=np.float64)
            Y_ml_use = np.asarray(tica_model.transform(X_ml_use), dtype=np.float64)

            base = protein_out / f"tica_{protein}_{seed_mode}__{run_id}"
            ml_proj_csv = base.with_name(base.name + "_projection.csv")
            ml_fes_npz = base.with_name(base.name + "_fes.npz")
            ml_fes_png = base.with_name(base.name + "_fes.png")
            meta_json = base.with_name(base.name + "_metadata.json")

            write_projection_csv(ml_proj_csv, ml_ts_use, Y_ml_use)
            F_ml, x_ml, y_ml = compute_fes_2d(Y_ml_use, bins=args.bins, temperature=args.temperature)
            np.savez(ml_fes_npz, F=F_ml, xedges=x_ml, yedges=y_ml)
            plot_fes(F_ml, x_ml, y_ml, ml_fes_png, title=f"{protein} MLCG {seed_mode}")

            meta = {
                "protein": protein,
                "run_id": run_id,
                "seed_mode": seed_mode,
                "classical_run_id": row_free["run_id"],
                "pair_hash": pair_hash,
                "pairs_csv": str(pairs_csv),
                "model_pkl": str(model_pkl),
                "lagtime": int(args.lagtime),
                "bins": int(args.bins),
                "temperature": float(args.temperature),
                "match_length": args.match_length,
                "strict_shape": bool(args.strict_shape),
                "n_atoms_classical": int(n_atoms),
                "n_atoms_mlcg": int(ml_coords.shape[1]),
                "frames": {
                    "classical_raw": int(classical_coords.shape[0]),
                    "mlcg_raw": int(ml_coords.shape[0]),
                    "classical_used": int(n_use),
                    "mlcg_used": int(n_use),
                },
                "timesteps": {
                    "classical_first": int(cls_ts_use[0]),
                    "classical_last": int(cls_ts_use[-1]),
                    "mlcg_first": int(ml_ts_use[0]),
                    "mlcg_last": int(ml_ts_use[-1]),
                },
                "outputs": {
                    "classical_projection_csv": str(class_proj_csv),
                    "mlcg_projection_csv": str(ml_proj_csv),
                    "classical_fes_npz": str(class_fes_npz),
                    "mlcg_fes_npz": str(ml_fes_npz),
                    "classical_fes_png": str(class_fes_png),
                    "mlcg_fes_png": str(ml_fes_png),
                },
            }
            meta_json.write_text(json.dumps(meta, indent=2, sort_keys=True))

            accepted_modes.add(seed_mode)
            n_mlcg_ok += 1
            index_rows.append(
                {
                    "protein": protein,
                    "seed_mode": seed_mode,
                    "run_id": run_id,
                    "status": "accepted",
                    "reason": "",
                    "metadata_json": str(meta_json),
                    "projection_csv": str(ml_proj_csv),
                }
            )

        if n_mlcg_ok == 0:
            summary_rows.append(
                {
                    "protein": protein,
                    "status": "rejected",
                    "reason": "no accepted MLCG projections",
                }
            )
        else:
            summary_rows.append(
                {
                    "protein": protein,
                    "status": "accepted",
                    "reason": "",
                    "classical_run_id": row_free["run_id"],
                    "n_classical_frames": int(n_classical_frames),
                    "n_atoms": int(n_atoms),
                    "pair_hash": pair_hash,
                    "accepted_seed_modes": ";".join(sorted(accepted_modes)),
                    "n_mlcg_accepted": int(n_mlcg_ok),
                    "classical_model": str(model_pkl),
                    "pairs_csv": str(pairs_csv),
                }
            )

    # Stable ordering
    index_rows.sort(key=lambda r: (str(r.get("protein", "")).lower(), str(r.get("seed_mode", "")), str(r.get("run_id", ""))))
    summary_rows.sort(key=lambda r: (str(r.get("protein", "")).lower()))

    index_csv = out_root / "tica_comparison_index.csv"
    with index_csv.open("w", newline="") as f:
        fields = ["protein", "seed_mode", "run_id", "status", "reason", "metadata_json", "projection_csv"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in index_rows:
            w.writerow({k: r.get(k, "") for k in fields})

    index_json = out_root / "tica_comparison_index.json"
    index_json.write_text(json.dumps(index_rows, indent=2, sort_keys=True))

    summary_csv = out_root / "tica_summary.csv"
    with summary_csv.open("w", newline="") as f:
        fields = [
            "protein",
            "status",
            "reason",
            "classical_run_id",
            "n_classical_frames",
            "n_atoms",
            "pair_hash",
            "accepted_seed_modes",
            "n_mlcg_accepted",
            "classical_model",
            "pairs_csv",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in summary_rows:
            w.writerow({k: r.get(k, "") for k in fields})

    summary_json = out_root / "tica_summary.json"
    summary_json.write_text(json.dumps(summary_rows, indent=2, sort_keys=True))

    print(
        json.dumps(
            {
                "mode": "experiment",
                "experiment_root": str(exp_root),
                "outdir": str(out_root),
                "mlcg_manifest": str(mlcg_manifest),
                "n_pairs": int(args.n_pairs),
                "n_proteins": len(summary_rows),
                "n_proteins_accepted": sum(1 for r in summary_rows if r.get("status") == "accepted"),
                "index_csv": str(index_csv),
                "summary_csv": str(summary_csv),
            },
            indent=2,
        )
    )


def main() -> None:
    args = parse_args()
    if args.experiment_root is not None:
        run_experiment_mode(args)
    else:
        run_single_mode(args)


if __name__ == "__main__":
    main()
