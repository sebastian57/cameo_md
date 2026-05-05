#!/usr/bin/env python3
"""Option 3 fixed-CA force analysis.

Supports two modes:
1) legacy single-run mode (explicit --protein/--dump/--cg-npz/--n-protein)
2) experiment mode (manifest-driven via --experiment-root)
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import subprocess
import sys
from pathlib import Path
from typing import Iterator, Optional

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)

    # Legacy/single-run mode
    p.add_argument("--protein", help="Protein name for single-run mode")
    p.add_argument("--dump", type=Path, help="protein_forces.dump path (single-run mode)")
    p.add_argument("--cg-npz", type=Path, help="CG NPZ path (single-run mode)")
    p.add_argument("--n-protein", type=int, help="Number of protein atoms in dump (single-run mode)")
    p.add_argument("--max-frames", type=int, default=0, help="If >0, subsample frames")

    # Model evaluation
    p.add_argument("--mlir-agg", type=Path, default=None, help="Agg model .mlir (legacy mode)")
    p.add_argument("--mlir-noagg", type=Path, default=None, help="Noagg model .mlir (legacy mode)")
    p.add_argument("--model-mlir", type=Path, default=None, help="Single model .mlir (experiment mode)")
    p.add_argument(
        "--lammps-bin",
        type=Path,
        default=Path("/p/project1/cameo/schmidt36/lammps/build/lmp"),
        help="LAMMPS binary",
    )

    # Experiment mode
    p.add_argument("--experiment-root", type=Path, default=None, help="outputs/<experiment_name>")
    p.add_argument("--cg-npz-map", type=Path, default=None, help="CSV/JSON mapping protein -> cg npz")
    p.add_argument("--include-incomplete", action="store_true", help="Include incomplete runs if dumps exist")

    p.add_argument("--out-dir", type=Path, default=Path("option3_results"), help="Output directory")
    return p.parse_args()


# ---------------------- parsing helpers ----------------------

def parse_dump_frames(path: Path, n_atoms: int) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield (R, F) per frame, shapes (n_atoms, 3)."""
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

            atom_line = fh.readline().strip()
            if not atom_line.startswith("ITEM: ATOMS"):
                continue
            header = atom_line.split()[2:]
            if "id" not in header:
                continue

            id_idx = header.index("id")
            x_col = "xu" if "xu" in header else "x"
            y_col = "yu" if "yu" in header else "y"
            z_col = "zu" if "zu" in header else "z"
            if not all(c in header for c in (x_col, y_col, z_col, "fx", "fy", "fz")):
                continue

            x_idx = header.index(x_col)
            y_idx = header.index(y_col)
            z_idx = header.index(z_col)
            fx_idx = header.index("fx")
            fy_idx = header.index("fy")
            fz_idx = header.index("fz")

            R = np.empty((n_atoms, 3), dtype=np.float64)
            F = np.empty((n_atoms, 3), dtype=np.float64)
            ok = True
            for _ in range(n_atoms):
                raw = fh.readline()
                if not raw:
                    ok = False
                    break
                cols = raw.split()
                try:
                    i = int(cols[id_idx]) - 1
                    R[i, 0] = float(cols[x_idx])
                    R[i, 1] = float(cols[y_idx])
                    R[i, 2] = float(cols[z_idx])
                    F[i, 0] = float(cols[fx_idx])
                    F[i, 1] = float(cols[fy_idx])
                    F[i, 2] = float(cols[fz_idx])
                except Exception:
                    ok = False
                    break
            if ok:
                yield R, F


def parse_last_timestep(path: Path) -> Optional[int]:
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


def parse_int_var_from_input(path: Path, key: str) -> Optional[int]:
    if not path.exists():
        return None
    txt = path.read_text()
    m = re.search(rf"^\s*variable\s+{re.escape(key)}\s+equal\s+([0-9]+)\s*$", txt, re.M)
    return int(m.group(1)) if m else None


def parse_n_protein_from_input(path: Path) -> Optional[int]:
    if not path.exists():
        return None
    txt = path.read_text()
    m = re.search(r"^\s*group\s+protein\s+id\s+1:(\d+)\s*$", txt, re.M)
    return int(m.group(1)) if m else None


def load_cg_map(path: Path) -> dict[str, Path]:
    if not path.exists():
        raise FileNotFoundError(f"cg-npz map not found: {path}")

    out: dict[str, Path] = {}
    if path.suffix.lower() == ".json":
        obj = json.loads(path.read_text())
        if isinstance(obj, dict):
            for k, v in obj.items():
                out[str(k).lower()] = Path(str(v)).resolve()
        elif isinstance(obj, list):
            for row in obj:
                k = str(row["protein"]).lower()
                out[k] = Path(str(row["cg_npz"])).resolve()
        else:
            raise ValueError("Unsupported JSON cg map format")
    else:
        with path.open(newline="") as f:
            r = csv.DictReader(f)
            if r.fieldnames and "protein" in r.fieldnames and "cg_npz" in r.fieldnames:
                for row in r:
                    out[str(row["protein"]).lower()] = Path(str(row["cg_npz"])).resolve()
            else:
                f.seek(0)
                rr = csv.reader(f)
                for row in rr:
                    if len(row) < 2:
                        continue
                    if row[0].strip().lower() == "protein":
                        continue
                    out[row[0].strip().lower()] = Path(row[1].strip()).resolve()

    if not out:
        raise ValueError(f"No entries parsed from cg map: {path}")
    return out


def parse_fixed_manifest(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open() as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            cols = raw.split("\t")
            if len(cols) < 5:
                continue
            run_id, protein, mode, run_dir, input_file = cols[:5]
            rows.append(
                {
                    "run_id": run_id,
                    "protein": protein,
                    "mode": mode,
                    "run_dir": Path(run_dir),
                    "input_file": Path(input_file),
                }
            )
    rows.sort(key=lambda r: (str(r["protein"]).lower(), str(r["run_id"]).lower()))
    return rows


# ---------------------- analysis helpers ----------------------

def load_dump_data(
    dump: Path,
    n_protein: int,
    ca_indices: np.ndarray,
    max_frames: int,
) -> dict:
    R_list: list[np.ndarray] = []
    F_ca_list: list[np.ndarray] = []

    for R_prot, F_prot in parse_dump_frames(dump, n_protein):
        R_list.append(R_prot[ca_indices])
        F_ca_list.append(F_prot[ca_indices])

    if not R_list:
        raise RuntimeError(f"No complete frames parsed from {dump}")

    n_total = len(R_list)
    if max_frames > 0 and n_total > max_frames:
        idx = np.round(np.linspace(0, n_total - 1, max_frames)).astype(int)
        R_list = [R_list[i] for i in idx]
        F_ca_list = [F_ca_list[i] for i in idx]

    R_ca_fixed = R_list[0]
    F_ca_raw = np.stack(F_ca_list, axis=0)
    return {"n_frames": len(R_list), "R_ca_fixed": R_ca_fixed, "F_ca_raw": F_ca_raw}


def compute_force_statistics(F: np.ndarray) -> dict:
    mean = F.mean(axis=0)
    std = F.std(axis=0, ddof=1)
    rms_mean = float(np.sqrt(np.mean(mean**2)))
    rms_noise = float(np.sqrt(np.mean(std**2)))
    return {"mean": mean, "std": std, "rms_mean": rms_mean, "rms_noise_per_comp": rms_noise}


def evaluate_model_at_fixed_ca(
    R_ca_fixed: np.ndarray,
    species_0indexed: np.ndarray,
    mlir_path: Path,
    lammps_bin: Path,
    work_dir: Path,
    label: str,
) -> np.ndarray:
    n_ca = R_ca_fixed.shape[0]
    n_types = int(species_0indexed.max()) + 1

    work_dir.mkdir(parents=True, exist_ok=True)
    data_file = work_dir / f"{label}.data"
    in_file = work_dir / f"{label}.in"
    dump_file = work_dir / f"{label}.dump"
    log_file = work_dir / f"{label}.log"

    pad = 15.0
    xlo, xhi = R_ca_fixed[:, 0].min() - pad, R_ca_fixed[:, 0].max() + pad
    ylo, yhi = R_ca_fixed[:, 1].min() - pad, R_ca_fixed[:, 1].max() + pad
    zlo, zhi = R_ca_fixed[:, 2].min() - pad, R_ca_fixed[:, 2].max() + pad

    with data_file.open("w") as fh:
        fh.write("LAMMPS data file — option3 fixed-CA single-point\n\n")
        fh.write(f"{n_ca} atoms\n")
        fh.write(f"{n_types} atom types\n\n")
        fh.write(f"{xlo:.6f} {xhi:.6f} xlo xhi\n")
        fh.write(f"{ylo:.6f} {yhi:.6f} ylo yhi\n")
        fh.write(f"{zlo:.6f} {zhi:.6f} zlo zhi\n\n")
        fh.write("Masses\n\n")
        for t in range(1, n_types + 1):
            fh.write(f"{t} 12.01\n")
        fh.write("\nAtoms\n\n")
        for i in range(n_ca):
            t = int(species_0indexed[i]) + 1
            x, y, z = R_ca_fixed[i]
            fh.write(f"{i + 1} {t} {x:.8f} {y:.8f} {z:.8f}\n")

    with in_file.open("w") as fh:
        fh.write(
            f"""units           real
atom_style      atomic
dimension       3
boundary        s s s

neighbor        2.0 bin
neigh_modify    every 1 delay 0 check yes once no

read_data       {data_file}

mass * 12.01

comm_modify     cutoff 14.0
pair_style      chemtrain_deploy cuda12 0.90
pair_coeff      * * {mlir_path} 1.2 1.1

thermo          1
thermo_style    custom step temp press pe ke etotal

dump            d1 all custom 1 {dump_file} id type xu yu zu fx fy fz
dump_modify     d1 sort id format line "%d %d %.8f %.8f %.8f %.10f %.10f %.10f"

run             0
"""
        )

    proc = subprocess.run(
        [str(lammps_bin), "-in", str(in_file), "-log", str(log_file)],
        capture_output=True,
        text=True,
        timeout=300,
    )
    if proc.returncode != 0:
        msg = (proc.stderr or proc.stdout or "")[-2000:]
        raise RuntimeError(f"Model LAMMPS evaluation failed: {msg}")

    frames = list(parse_dump_frames(dump_file, n_ca))
    if not frames:
        raise RuntimeError(f"No model dump frames parsed from {dump_file}")
    _, F_model = frames[0]
    return F_model


def flatten_corr(a: np.ndarray, b: np.ndarray) -> float:
    x = a.reshape(-1)
    y = b.reshape(-1)
    sx = float(np.std(x))
    sy = float(np.std(y))
    if sx < 1e-12 or sy < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    x = a.reshape(-1)
    y = b.reshape(-1)
    nx = float(np.linalg.norm(x))
    ny = float(np.linalg.norm(y))
    if nx < 1e-12 or ny < 1e-12:
        return float("nan")
    return float(np.dot(x, y) / (nx * ny))


def analyze_single_run(
    *,
    protein: str,
    run_id: str,
    run_dir: Path,
    input_file: Path,
    cg_npz: Path,
    model_mlir: Path,
    lammps_bin: Path,
    out_dir: Path,
    max_frames: int,
) -> dict:
    protein_dump = run_dir / "protein_forces.dump"

    cg_data = np.load(cg_npz, allow_pickle=True)
    ca_indices = np.asarray(cg_data["ca_indices"], dtype=int)
    species_0indexed = np.asarray(cg_data["species"][0], dtype=int)

    n_protein = parse_n_protein_from_input(input_file)
    if n_protein is None:
        raise RuntimeError(f"Could not parse n_protein from {input_file}")

    data = load_dump_data(protein_dump, n_protein, ca_indices, max_frames=max_frames)
    stats = compute_force_statistics(data["F_ca_raw"])

    tmp_model_dir = out_dir / "_tmp_model_eval" / protein / run_id
    F_model = evaluate_model_at_fixed_ca(
        data["R_ca_fixed"],
        species_0indexed,
        model_mlir,
        lammps_bin,
        tmp_model_dir,
        f"{protein}_{run_id}",
    )

    diff = F_model - stats["mean"]
    rmse_systematic = float(np.sqrt(np.mean(diff**2)))
    thermal_noise = float(np.sqrt(np.mean(stats["std"]**2)))
    total_expected = float(math.sqrt(rmse_systematic**2 + thermal_noise**2))

    per_ca_rmse = np.sqrt(np.mean(diff**2, axis=1))
    per_ca_noise = np.sqrt(np.mean(stats["std"]**2, axis=1))

    corr = flatten_corr(F_model, stats["mean"])
    cos = cosine_similarity(F_model, stats["mean"])

    run_metrics = {
        "protein": protein,
        "run_id": run_id,
        "run_dir": str(run_dir),
        "n_frames": int(data["n_frames"]),
        "n_ca": int(len(ca_indices)),
        "rmse_systematic": rmse_systematic,
        "thermal_noise": thermal_noise,
        "expected_total_rmse": total_expected,
        "pearson_flat": corr,
        "cosine": cos,
        "rms_mean_force": float(stats["rms_mean"]),
        "rms_noise_per_comp": float(stats["rms_noise_per_comp"]),
    }

    per_ca_csv = out_dir / f"option3_{protein}__{run_id}_per_ca.csv"
    with per_ca_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "ca_index_0based",
                "mean_fx",
                "mean_fy",
                "mean_fz",
                "std_fx",
                "std_fy",
                "std_fz",
                "pred_fx",
                "pred_fy",
                "pred_fz",
                "per_ca_rmse",
                "per_ca_noise",
            ]
        )
        for k, ca in enumerate(ca_indices):
            w.writerow(
                [
                    int(ca),
                    float(stats["mean"][k, 0]),
                    float(stats["mean"][k, 1]),
                    float(stats["mean"][k, 2]),
                    float(stats["std"][k, 0]),
                    float(stats["std"][k, 1]),
                    float(stats["std"][k, 2]),
                    float(F_model[k, 0]),
                    float(F_model[k, 1]),
                    float(F_model[k, 2]),
                    float(per_ca_rmse[k]),
                    float(per_ca_noise[k]),
                ]
            )

    run_json = out_dir / f"option3_{protein}__{run_id}_stats.json"
    run_json.write_text(json.dumps(run_metrics, indent=2))

    return {
        "metrics": run_metrics,
        "json": str(run_json),
        "per_ca_csv": str(per_ca_csv),
    }


# ---------------------- execution modes ----------------------

def run_single_mode(args: argparse.Namespace) -> None:
    required = [args.protein, args.dump, args.cg_npz, args.n_protein]
    if any(v is None for v in required):
        raise SystemExit(
            "Single-run mode requires --protein --dump --cg-npz --n-protein "
            "(or use --experiment-root mode)."
        )

    model_path = args.model_mlir or args.mlir_noagg or args.mlir_agg
    if model_path is None:
        raise SystemExit("Single-run mode requires one model path (--model-mlir or --mlir-noagg/--mlir-agg)")

    if not args.dump.exists():
        raise SystemExit(f"Dump not found: {args.dump}")
    if not args.cg_npz.exists():
        raise SystemExit(f"CG NPZ not found: {args.cg_npz}")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # fabricate minimal input-like metadata via explicit n_protein
    fake_input = args.out_dir / "_single_mode_input_hint.in"
    fake_input.write_text(f"group protein id 1:{int(args.n_protein)}\n")

    result = analyze_single_run(
        protein=str(args.protein),
        run_id="single",
        run_dir=args.dump.parent,
        input_file=fake_input,
        cg_npz=args.cg_npz,
        model_mlir=Path(model_path),
        lammps_bin=args.lammps_bin,
        out_dir=args.out_dir,
        max_frames=args.max_frames,
    )
    summary = {"mode": "single", "result": result["metrics"]}
    (args.out_dir / f"option3_{args.protein}_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def run_experiment_mode(args: argparse.Namespace) -> None:
    if args.experiment_root is None:
        raise SystemExit("--experiment-root is required in experiment mode")
    if args.cg_npz_map is None:
        raise SystemExit("--cg-npz-map is required in experiment mode")

    model_path = args.model_mlir or args.mlir_noagg or args.mlir_agg
    if model_path is None:
        raise SystemExit("Experiment mode requires --model-mlir (or --mlir-noagg/--mlir-agg)")
    model_path = Path(model_path).resolve()

    exp_root = args.experiment_root.resolve()
    manifest = exp_root / "classical_fixed_manifest.tsv"
    if not manifest.exists():
        raise SystemExit(f"Missing manifest: {manifest}")

    cg_map = load_cg_map(args.cg_npz_map)

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = parse_fixed_manifest(manifest)
    proteins_in_manifest = sorted({str(r["protein"]).lower() for r in rows if str(r["mode"]) == "fixed"})
    missing_mappings = [p for p in proteins_in_manifest if p not in cg_map]
    if missing_mappings:
        raise SystemExit(
            "Missing CG NPZ mappings for proteins: " + ", ".join(missing_mappings)
        )

    run_index: list[dict] = []
    per_protein_runs: dict[str, list[dict]] = {}

    for row in rows:
        if row["mode"] != "fixed":
            continue

        run_id = row["run_id"]
        protein = str(row["protein"])
        protein_key = protein.lower()
        run_dir: Path = row["run_dir"]
        input_file: Path = row["input_file"]

        entry = {
            "run_id": run_id,
            "protein": protein,
            "run_dir": str(run_dir),
            "status": "rejected",
            "reason": "",
            "complete": False,
            "sample_steps": None,
            "last_timestep_protein": None,
            "last_timestep_ca": None,
        }

        if protein_key not in cg_map:
            entry["reason"] = f"missing cg map entry for protein={protein}"
            run_index.append(entry)
            continue

        protein_dump = run_dir / "protein_forces.dump"
        ca_dump = run_dir / "ca_forces.dump"

        sample_steps = parse_int_var_from_input(input_file, "sample_steps")
        last_prot = parse_last_timestep(protein_dump)
        last_ca = parse_last_timestep(ca_dump)

        entry["sample_steps"] = sample_steps
        entry["last_timestep_protein"] = last_prot
        entry["last_timestep_ca"] = last_ca

        complete = (
            sample_steps is not None
            and protein_dump.exists()
            and ca_dump.exists()
            and last_prot is not None
            and last_ca is not None
            and last_prot >= sample_steps
            and last_ca >= sample_steps
        )
        entry["complete"] = bool(complete)

        if not complete and not args.include_incomplete:
            entry["reason"] = "incomplete outputs (missing files or final timestep < sample_steps)"
            run_index.append(entry)
            continue

        if not protein_dump.exists():
            entry["reason"] = "missing protein_forces.dump"
            run_index.append(entry)
            continue

        try:
            analyzed = analyze_single_run(
                protein=protein,
                run_id=run_id,
                run_dir=run_dir,
                input_file=input_file,
                cg_npz=cg_map[protein_key],
                model_mlir=model_path,
                lammps_bin=args.lammps_bin,
                out_dir=out_dir,
                max_frames=args.max_frames,
            )
            entry["status"] = "accepted"
            entry["reason"] = ""
            entry["result_json"] = analyzed["json"]
            entry["result_csv"] = analyzed["per_ca_csv"]
            per_protein_runs.setdefault(protein, []).append(analyzed["metrics"])
        except Exception as exc:  # noqa: BLE001
            entry["status"] = "rejected"
            entry["reason"] = f"analysis error: {exc}"

        run_index.append(entry)

    protein_aggregate_rows: list[dict] = []
    for protein in sorted(per_protein_runs.keys(), key=str.lower):
        metrics = per_protein_runs[protein]
        rmse = np.array([m["rmse_systematic"] for m in metrics], dtype=float)
        noise = np.array([m["thermal_noise"] for m in metrics], dtype=float)
        corr = np.array([m["pearson_flat"] for m in metrics if not math.isnan(m["pearson_flat"])], dtype=float)
        cos = np.array([m["cosine"] for m in metrics if not math.isnan(m["cosine"])], dtype=float)

        rejected = [r for r in run_index if r["protein"].lower() == protein.lower() and r["status"] != "accepted"]
        accepted = [r for r in run_index if r["protein"].lower() == protein.lower() and r["status"] == "accepted"]

        agg = {
            "protein": protein,
            "n_total": len(accepted) + len(rejected),
            "n_accepted": len(accepted),
            "n_rejected": len(rejected),
            "rmse_systematic_mean": float(np.mean(rmse)) if len(rmse) else float("nan"),
            "rmse_systematic_std": float(np.std(rmse, ddof=1)) if len(rmse) > 1 else 0.0,
            "thermal_noise_mean": float(np.mean(noise)) if len(noise) else float("nan"),
            "thermal_noise_std": float(np.std(noise, ddof=1)) if len(noise) > 1 else 0.0,
            "pearson_mean": float(np.mean(corr)) if len(corr) else float("nan"),
            "cosine_mean": float(np.mean(cos)) if len(cos) else float("nan"),
            "accepted_run_ids": [r["run_id"] for r in accepted],
            "rejected_run_ids": [r["run_id"] for r in rejected],
        }
        protein_aggregate_rows.append(agg)

        (out_dir / f"option3_{protein}_aggregate.json").write_text(json.dumps(agg, indent=2))

        agg_csv = out_dir / f"option3_{protein}_aggregate.csv"
        with agg_csv.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["metric", "value"])
            for k, v in agg.items():
                if isinstance(v, list):
                    w.writerow([k, ";".join(map(str, v))])
                else:
                    w.writerow([k, v])

    summary_csv = out_dir / "option3_summary.csv"
    with summary_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "protein",
                "n_total",
                "n_accepted",
                "n_rejected",
                "rmse_systematic_mean",
                "rmse_systematic_std",
                "thermal_noise_mean",
                "thermal_noise_std",
                "pearson_mean",
                "cosine_mean",
            ]
        )
        for row in protein_aggregate_rows:
            w.writerow(
                [
                    row["protein"],
                    row["n_total"],
                    row["n_accepted"],
                    row["n_rejected"],
                    row["rmse_systematic_mean"],
                    row["rmse_systematic_std"],
                    row["thermal_noise_mean"],
                    row["thermal_noise_std"],
                    row["pearson_mean"],
                    row["cosine_mean"],
                ]
            )

    run_index_json = out_dir / "option3_run_index.json"
    run_index_json.write_text(json.dumps(run_index, indent=2))

    print(
        json.dumps(
            {
                "mode": "experiment",
                "experiment_root": str(exp_root),
                "out_dir": str(out_dir),
                "n_runs_indexed": len(run_index),
                "n_runs_accepted": sum(1 for r in run_index if r["status"] == "accepted"),
                "n_proteins_with_results": len(protein_aggregate_rows),
                "summary_csv": str(summary_csv),
                "run_index_json": str(run_index_json),
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
