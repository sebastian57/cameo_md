#!/usr/bin/env python3
"""Prepare multi-protein classical experiment runs from mdCATH .h5 inputs.

Creates per-run LAMMPS assets and run-specific input files for:
- free classical runs (1 frame/protein)
- fixed-CA classical runs (N frames/protein)

Also writes deterministic manifests for SLURM array submission.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np


DEFAULT_IN_SCOPE = ["2gy5A01", "4q5wA02"]
DEFAULT_OUT_SCOPE = ["3h7jA02"]
DEFAULT_FF_FILES = [
    "top_all22_prot_mdcath.rtf",
    "par_all22_prot_mdcath.prm",
    "charmm22.cmap",
]


@dataclass
class RunRow:
    run_id: str
    protein: str
    scope: str
    mode: str
    frame_idx: int
    run_dir: Path
    input_file: Path
    status: str
    reason: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--in-scope-dir", type=Path, required=True)
    p.add_argument("--out-scope-dir", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--input-free", type=Path, required=True)
    p.add_argument("--input-fixed", type=Path, required=True)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--samples-per-protein", type=int, default=5)
    p.add_argument("--temp-group", default="320")
    p.add_argument("--run-group", default="0")
    p.add_argument(
        "--ff-src",
        type=Path,
        default=Path("/p/project1/cameo/schmidt36/cameo_md/runs/4q5wA02_charmm22_cmap_fixed"),
    )
    p.add_argument(
        "--charmm2lammps",
        type=Path,
        default=Path("/p/project1/cameo/schmidt36/lammps/tools/ch2lmp/charmm2lammps.pl"),
    )
    p.add_argument("--fixed-eq-steps", type=int, default=50000)
    p.add_argument("--skip-free", action="store_true",
                   help="Only prepare fixed-CA runs; skip free runs entirely")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--proteins-in", default=",".join(DEFAULT_IN_SCOPE))
    p.add_argument("--proteins-out", default=",".join(DEFAULT_OUT_SCOPE))
    return p.parse_args()


def normalize_list(spec: str) -> list[str]:
    return [x.strip() for x in spec.split(",") if x.strip()]


def discover_h5(dir_path: Path) -> dict[str, tuple[str, Path]]:
    mapping: dict[str, tuple[str, Path]] = {}
    for h5 in sorted(dir_path.glob("*.h5")):
        stem = h5.stem
        pid = stem
        if stem.startswith("mdcath_dataset_"):
            pid = stem.replace("mdcath_dataset_", "", 1)
        mapping[pid.lower()] = (pid, h5.resolve())
    return mapping


def parse_int_variable(template_text: str, key: str) -> int:
    pattern = re.compile(rf"^\s*variable\s+{re.escape(key)}\s+equal\s+([0-9]+)\s*$", re.MULTILINE)
    m = pattern.search(template_text)
    if not m:
        raise ValueError(f"Could not parse integer variable '{key}' from template")
    return int(m.group(1))


def split_exact(total: int, n_parts: int) -> list[int]:
    if n_parts <= 0:
        raise ValueError("n_parts must be > 0")
    base = total // n_parts
    rem = total % n_parts
    return [base + (1 if i < rem else 0) for i in range(n_parts)]


def replace_protein_coords(pdb_str: str, new_coords: np.ndarray) -> str:
    out_lines: list[str] = []
    prot_idx = 0
    for line in pdb_str.splitlines():
        if line.startswith(("ATOM", "HETATM")):
            seg = line[72:76].strip()
            if seg == "P0":
                x, y, z = new_coords[prot_idx]
                line = line[:30] + f"{x:8.3f}{y:8.3f}{z:8.3f}" + line[54:]
                prot_idx += 1
        out_lines.append(line)
    return "\n".join(out_lines) + "\n"


def detect_water_angle_type(data_protein_path: Path) -> int:
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
    ids: list[int] = []
    atom_idx = 0
    for line in pdb_protein_atoms_str.splitlines():
        if line.startswith("ATOM"):
            atom_name = line[12:16].strip()
            if atom_name == "CA":
                ids.append(atom_idx + 1)
            atom_idx += 1
    return ids


def ensure_string_var(text: str, key: str, value: str) -> str:
    repl = f'variable {key:<14} string "{value}"'
    pat = re.compile(rf"^\s*variable\s+{re.escape(key)}\s+string\s+.*$", re.MULTILINE)
    if pat.search(text):
        return pat.sub(repl, text)
    return repl + "\n" + text


def ensure_equal_var(text: str, key: str, value: int) -> str:
    repl = f"variable {key:<14} equal {value}"
    pat = re.compile(rf"^\s*variable\s+{re.escape(key)}\s+equal\s+.*$", re.MULTILINE)
    if pat.search(text):
        return pat.sub(repl, text)
    return repl + "\n" + text


def patch_input_template(
    template_path: Path,
    out_path: Path,
    *,
    data_file: Path,
    cmap_file: Path,
    dump_dir: Path,
    ca_group_file: Path,
    n_protein: int,
    sample_steps: int | None,
    eq_steps: int | None,
    dry_run: bool,
) -> None:
    text = template_path.read_text()

    lines = text.splitlines()
    filtered: list[str] = []
    inserted_include = False
    for line in lines:
        if re.match(r"^\s*group\s+ca\s+id\b", line):
            continue
        if re.match(r"^\s*include\s+.*ca_group\.lmp", line):
            continue
        if re.match(r"^\s*group\s+mobile\s+subtract\s+all\s+ca\b", line):
            continue
        filtered.append(line)
        if re.match(r"^\s*read_data\b", line):
            filtered.append(f'include         "{ca_group_file}"')
            inserted_include = True
    if not inserted_include:
        filtered.append(f'include         "{ca_group_file}"')

    patched = "\n".join(filtered) + "\n"

    patched = ensure_string_var(patched, "data_file", str(data_file))
    patched = ensure_string_var(patched, "cmap_file", str(cmap_file))
    patched = ensure_string_var(patched, "dump_dir", str(dump_dir))

    if sample_steps is not None:
        patched = ensure_equal_var(patched, "sample_steps", int(sample_steps))
    if eq_steps is not None:
        patched = ensure_equal_var(patched, "eq_steps", int(eq_steps))

    pat_protein = re.compile(r"^\s*group\s+protein\s+id\s+1:\d+\s*$", re.MULTILINE)
    repl_protein = f"group           protein id 1:{n_protein}"
    if pat_protein.search(patched):
        patched = pat_protein.sub(repl_protein, patched)
    else:
        patched += repl_protein + "\n"

    if not dry_run:
        out_path.write_text(patched)


def run_charmm2lammps(run_dir: Path, perl_script: Path, dry_run: bool) -> tuple[bool, str]:
    if dry_run:
        return True, "dry-run"

    perl = shutil.which("perl") or "perl"
    cmd = [perl, str(perl_script), "-cmap=22", "all22_prot_mdcath", "protein"]
    result = subprocess.run(cmd, cwd=run_dir, capture_output=True, text=True)
    (run_dir / "conversion.log").write_text((result.stdout or "") + (result.stderr or ""))
    if result.returncode != 0:
        tail = (result.stderr or result.stdout or "")[-700:]
        return False, f"charmm2lammps failed: {tail.strip()}"

    src = run_dir / "protein.data"
    dst = run_dir / "data.protein"
    if not src.exists():
        return False, "charmm2lammps produced no protein.data"
    shutil.copy(src, dst)
    return True, "ok"


def write_ca_group(run_dir: Path, ca_ids: Iterable[int], dry_run: bool) -> Path:
    ca_ids = list(ca_ids)
    group_file = run_dir / "ca_group.lmp"
    if dry_run:
        return group_file
    id_list = " ".join(str(i) for i in ca_ids)
    group_text = (
        "# Auto-generated CA groups\n"
        f"group ca id {id_list}\n"
        "group mobile subtract all ca\n"
    )
    group_file.write_text(group_text)
    (run_dir / "ca_atom_ids.txt").write_text("\n".join(str(i) for i in ca_ids) + "\n")
    return group_file


def prepare_single_run(
    *,
    protein_label: str,
    scope: str,
    h5_path: Path,
    frame_idx: int,
    mode: str,
    run_dir: Path,
    input_template: Path,
    ff_src: Path,
    ff_files: list[str],
    charmm2lammps_pl: Path,
    temp_group: str,
    run_group: str,
    sample_steps: int | None,
    eq_steps: int | None,
    dry_run: bool,
) -> tuple[str, str, Path | None]:
    """Returns (status, reason, input_file)."""
    try:
        with h5py.File(h5_path, "r") as h5f:
            if protein_label in h5f:
                prot = h5f[protein_label]
            else:
                first_key = next(iter(h5f.keys()))
                prot = h5f[first_key]

            pdb_str = prot["pdb"][()].decode()
            psf_str = prot["psf"][()].decode()
            pdb_prot_atoms = prot["pdbProteinAtoms"][()].decode()
            coords = prot[temp_group][run_group]["coords"][()]

        n_frames, n_prot, _ = coords.shape
        if frame_idx < 0 or frame_idx >= n_frames:
            return "failed", f"frame {frame_idx} out of range 0..{n_frames - 1}", None

        run_dir.mkdir(parents=True, exist_ok=True)
        if not dry_run:
            (run_dir / "protein.pdb").write_text(replace_protein_coords(pdb_str, coords[frame_idx].astype(float)))
            (run_dir / "protein.psf").write_text(psf_str)
            for ff in ff_files:
                src = ff_src / ff
                if not src.exists():
                    return "failed", f"missing FF file: {src}", None
                shutil.copy(src, run_dir / ff)

        ok, reason = run_charmm2lammps(run_dir, charmm2lammps_pl, dry_run)
        if not ok:
            return "failed", reason, None

        ca_ids = extract_ca_ids_from_pdb(pdb_prot_atoms)
        if not ca_ids:
            return "failed", "no CA atoms parsed from pdbProteinAtoms", None

        data_protein = run_dir / "data.protein"
        water_angle_type = 110 if dry_run else detect_water_angle_type(data_protein)
        _ = water_angle_type

        ca_group_file = write_ca_group(run_dir, ca_ids, dry_run)
        input_file = run_dir / f"in.{mode}.lmp"

        patch_input_template(
            input_template,
            input_file,
            data_file=(run_dir / "data.protein"),
            cmap_file=(run_dir / "charmm22.cmap"),
            dump_dir=run_dir,
            ca_group_file=ca_group_file,
            n_protein=n_prot,
            sample_steps=sample_steps,
            eq_steps=eq_steps,
            dry_run=dry_run,
        )

        if not dry_run:
            meta = {
                "protein": protein_label,
                "scope": scope,
                "mode": mode,
                "frame_idx": int(frame_idx),
                "n_frames": int(n_frames),
                "n_protein_atoms": int(n_prot),
                "h5_path": str(h5_path),
                "run_dir": str(run_dir),
                "input_file": str(input_file),
            }
            (run_dir / "run_meta.json").write_text(json.dumps(meta, indent=2))

        return "ready", "", input_file

    except Exception as exc:  # noqa: BLE001
        return "failed", str(exc), None


def write_rows(path: Path, rows: list[RunRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(
                "\t".join(
                    [
                        r.run_id,
                        r.protein,
                        r.scope,
                        r.mode,
                        str(r.frame_idx),
                        str(r.run_dir),
                        str(r.input_file),
                        r.status,
                        r.reason.replace("\t", " "),
                    ]
                )
                + "\n"
            )


def write_ready_manifest(path: Path, rows: list[RunRow]) -> None:
    with path.open("w") as f:
        for r in rows:
            if r.status != "ready":
                continue
            f.write("\t".join([r.run_id, r.protein, r.mode, str(r.run_dir), str(r.input_file)]) + "\n")


def main() -> None:
    args = parse_args()

    in_scope_targets = normalize_list(args.proteins_in)
    out_scope_targets = normalize_list(args.proteins_out)
    if not in_scope_targets or not out_scope_targets:
        raise ValueError("Both --proteins-in and --proteins-out must be non-empty")

    in_scope_map = discover_h5(args.in_scope_dir.resolve())
    out_scope_map = discover_h5(args.out_scope_dir.resolve())

    free_template = args.input_free.resolve()
    fixed_template = args.input_fixed.resolve()
    if not free_template.exists():
        raise FileNotFoundError(f"free template not found: {free_template}")
    if not fixed_template.exists():
        raise FileNotFoundError(f"fixed template not found: {fixed_template}")

    if args.skip_free:
        # Each fixed run uses the sample_steps from the fixed template as-is.
        fixed_step_splits = [None] * args.samples_per_protein
    else:
        free_template_text = free_template.read_text()
        free_total_sample_steps = parse_int_variable(free_template_text, "sample_steps")
        fixed_step_splits = split_exact(free_total_sample_steps, args.samples_per_protein)

    rng = np.random.default_rng(args.seed)
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    all_rows: list[RunRow] = []
    free_rows: list[RunRow] = []
    fixed_rows: list[RunRow] = []

    targets: list[tuple[str, str, dict[str, tuple[str, Path]]]] = []
    targets.extend((pid, "in_scope", in_scope_map) for pid in in_scope_targets)
    targets.extend((pid, "out_scope", out_scope_map) for pid in out_scope_targets)

    for requested_pid, scope, scope_map in targets:
        key = requested_pid.lower()
        if key not in scope_map:
            fail_row = RunRow(
                run_id=f"{requested_pid}__missing",
                protein=requested_pid,
                scope=scope,
                mode="missing",
                frame_idx=-1,
                run_dir=output_root / "classical" / "missing" / requested_pid,
                input_file=Path(""),
                status="failed",
                reason=f"protein not found in {scope} dir",
            )
            all_rows.append(fail_row)
            continue

        canonical_pid, h5_path = scope_map[key]
        with h5py.File(h5_path, "r") as h5f:
            group = h5f[canonical_pid] if canonical_pid in h5f else h5f[next(iter(h5f.keys()))]
            coords = group[args.temp_group][args.run_group]["coords"][()]
            n_frames = int(coords.shape[0])

        needed = args.samples_per_protein if args.skip_free else 1 + args.samples_per_protein
        if n_frames < needed:
            fail_row = RunRow(
                run_id=f"{canonical_pid}__insufficient_frames",
                protein=canonical_pid,
                scope=scope,
                mode="sampling",
                frame_idx=-1,
                run_dir=output_root / "classical" / "missing" / canonical_pid,
                input_file=Path(""),
                status="failed",
                reason=f"need >= {needed} frames, got {n_frames}",
            )
            all_rows.append(fail_row)
            continue

        selected = rng.choice(n_frames, size=needed, replace=False).tolist()

        if args.skip_free:
            fixed_frames = [int(x) for x in selected]
        else:
            free_frame = int(selected[0])
            fixed_frames = [int(x) for x in selected[1:]]

            free_run_dir = output_root / "classical" / "free" / canonical_pid / f"frame_{free_frame:05d}"
            free_run_id = f"{canonical_pid}__free__f{free_frame:05d}"
            free_status, free_reason, free_input = prepare_single_run(
                protein_label=canonical_pid,
                scope=scope,
                h5_path=h5_path,
                frame_idx=free_frame,
                mode="free",
                run_dir=free_run_dir,
                input_template=free_template,
                ff_src=args.ff_src.resolve(),
                ff_files=DEFAULT_FF_FILES,
                charmm2lammps_pl=args.charmm2lammps.resolve(),
                temp_group=args.temp_group,
                run_group=args.run_group,
                sample_steps=None,
                eq_steps=None,
                dry_run=args.dry_run,
            )
            free_row = RunRow(
                run_id=free_run_id,
                protein=canonical_pid,
                scope=scope,
                mode="free",
                frame_idx=free_frame,
                run_dir=free_run_dir,
                input_file=free_input or free_run_dir / "in.free.lmp",
                status=free_status,
                reason=free_reason,
            )
            all_rows.append(free_row)
            free_rows.append(free_row)

        for i, (frame_idx, sample_steps) in enumerate(zip(fixed_frames, fixed_step_splits), start=1):
            fixed_run_dir = output_root / "classical" / "fixed" / canonical_pid / f"x0_{i:02d}_frame_{frame_idx:05d}"
            fixed_run_id = f"{canonical_pid}__fixed__x{i:02d}__f{frame_idx:05d}"
            fixed_status, fixed_reason, fixed_input = prepare_single_run(
                protein_label=canonical_pid,
                scope=scope,
                h5_path=h5_path,
                frame_idx=frame_idx,
                mode="fixed",
                run_dir=fixed_run_dir,
                input_template=fixed_template,
                ff_src=args.ff_src.resolve(),
                ff_files=DEFAULT_FF_FILES,
                charmm2lammps_pl=args.charmm2lammps.resolve(),
                temp_group=args.temp_group,
                run_group=args.run_group,
                sample_steps=int(sample_steps) if sample_steps is not None else None,
                eq_steps=args.fixed_eq_steps,
                dry_run=args.dry_run,
            )
            row = RunRow(
                run_id=fixed_run_id,
                protein=canonical_pid,
                scope=scope,
                mode="fixed",
                frame_idx=frame_idx,
                run_dir=fixed_run_dir,
                input_file=fixed_input or fixed_run_dir / "in.fixed.lmp",
                status=fixed_status,
                reason=fixed_reason,
            )
            all_rows.append(row)
            fixed_rows.append(row)

    all_manifest = output_root / "classical_manifest.tsv"
    free_manifest = output_root / "classical_free_manifest.tsv"
    fixed_manifest = output_root / "classical_fixed_manifest.tsv"

    write_rows(all_manifest, all_rows)
    write_ready_manifest(free_manifest, free_rows)
    write_ready_manifest(fixed_manifest, fixed_rows)

    failed_rows = [r for r in all_rows if r.status != "ready"]
    summary = {
        "seed": args.seed,
        "samples_per_protein": args.samples_per_protein,
        "skip_free": bool(args.skip_free),
        "free_total_sample_steps": None if args.skip_free else free_total_sample_steps,
        "fixed_step_splits": fixed_step_splits,
        "counts": {
            "free_ready": sum(1 for r in free_rows if r.status == "ready"),
            "fixed_ready": sum(1 for r in fixed_rows if r.status == "ready"),
            "failed": len(failed_rows),
        },
        "manifests": {
            "all": str(all_manifest),
            "free_ready": str(free_manifest),
            "fixed_ready": str(fixed_manifest),
        },
        "dry_run": bool(args.dry_run),
    }
    (output_root / "classical_prepare_summary.json").write_text(json.dumps(summary, indent=2))

    print(json.dumps(summary, indent=2))
    if failed_rows:
        print("\\nFailed rows:")
        for r in failed_rows:
            print(f"- {r.run_id}: {r.reason}")


if __name__ == "__main__":
    main()
