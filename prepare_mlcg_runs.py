#!/usr/bin/env python3
"""Prepare MLCG experiment runs from classical fixed-run outputs."""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass
class MlcgRow:
    run_id: str
    protein: str
    seed_mode: str
    source_run_id: str
    source_run_dir: Path
    run_dir: Path
    input_file: Path
    status: str
    reason: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--classical-output", type=Path, required=True)
    p.add_argument("--trained-model", type=Path, required=True)
    p.add_argument("--input-template", type=Path, required=True)
    p.add_argument("--converter", type=Path, default=Path("/p/project1/cameo/schmidt36/cameo_md/build_ml_data_from_charmm_eq_dump.py"))
    p.add_argument(
        "--cg-template-data",
        type=Path,
        default=Path("/p/project1/cameo/schmidt36/cameo_md/structures/config_44.data"),
        help="Fallback template file or directory with template .data files",
    )
    p.add_argument(
        "--cg-npz-map",
        type=Path,
        default=None,
        help="Optional CSV/JSON mapping: protein -> cg_npz. Preferred atom-type source.",
    )
    p.add_argument(
        "--cg-npz-search-root",
        type=Path,
        default=Path("/p/project1/cameo/schmidt36/cameo_cg/data_prep/datasets"),
        help="Fallback search root for <protein>_cg.npz under **/02_cg_npz/",
    )
    p.add_argument(
        "--mlcg-out-dir",
        type=Path,
        default=None,
        help=(
            "Output directory for mlcg manifests and run subdirs. "
            "Defaults to <classical-output>/mlcg/. Use this to create named "
            "variant subdirs (e.g. .../mlcg_ml_only, .../mlcg_brownian)."
        ),
    )
    p.add_argument(
        "--run-id-suffix",
        type=str,
        default="",
        help=(
            "Suffix appended to every run_id in the manifest (e.g. '__ml_only'). "
            "Useful to distinguish variants when inspecting TICA output filenames."
        ),
    )
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def patch_mlcg_input(template_text: str, *, data_file: Path, model_file: Path, dump_dir: Path) -> str:
    text = template_text

    def ensure_string_var(src: str, key: str, value: str) -> str:
        repl = f'variable {key:<14} string "{value}"'
        pat = re.compile(rf"^\s*variable\s+{re.escape(key)}\s+string\s+.*$", re.MULTILINE)
        if pat.search(src):
            return pat.sub(repl, src)
        return repl + "\n" + src

    text = ensure_string_var(text, "data_file", str(data_file))
    text = ensure_string_var(text, "model_file", str(model_file))
    text = ensure_string_var(text, "dump_dir", str(dump_dir))
    return text


def parse_fixed_manifest(path: Path) -> list[tuple[str, str, str, Path, Path]]:
    rows: list[tuple[str, str, str, Path, Path]] = []
    with path.open() as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            cols = raw.split("\t")
            if len(cols) < 5:
                continue
            run_id, protein, mode, run_dir, input_file = cols[:5]
            rows.append((run_id, protein, mode, Path(run_dir), Path(input_file)))
    return rows


def load_cg_map(path: Path | None) -> dict[str, Path]:
    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError(f"cg map not found: {path}")

    out: dict[str, Path] = {}
    if path.suffix.lower() == ".json":
        obj = json.loads(path.read_text())
        if isinstance(obj, dict):
            for k, v in obj.items():
                out[str(k).lower()] = Path(str(v)).resolve()
        elif isinstance(obj, list):
            for row in obj:
                out[str(row["protein"]).lower()] = Path(str(row["cg_npz"])).resolve()
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
    return out


def parse_data_atom_count(path: Path) -> int | None:
    if not path.exists():
        return None
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if line.endswith(" atoms"):
            try:
                return int(line.split()[0])
            except Exception:
                return None
    return None


def parse_dump_atom_count(path: Path) -> int | None:
    if not path.exists():
        return None
    with path.open() as f:
        for line in f:
            if line.startswith("ITEM: NUMBER OF ATOMS"):
                try:
                    return int(next(f).strip())
                except Exception:
                    return None
    return None


def collect_template_candidates(base: Path) -> list[Path]:
    if base.is_dir():
        return sorted(base.glob("*.data"))
    if base.exists():
        cands = [base]
        sibling = sorted(base.parent.glob("*.data"))
        for p in sibling:
            if p not in cands:
                cands.append(p)
        return cands
    return []


def resolve_template_for_natoms(n_atoms: int, candidates: list[Path]) -> Path | None:
    for p in candidates:
        c = parse_data_atom_count(p)
        if c == n_atoms:
            return p
    return None


def resolve_species_npz(protein: str, cg_map: dict[str, Path], search_root: Path) -> Path | None:
    key = protein.lower()
    mapped = cg_map.get(key)
    if mapped is not None and mapped.exists():
        return mapped

    pattern = f"{protein}_cg.npz"
    if search_root.exists():
        hits = sorted(search_root.glob(f"**/02_cg_npz/{pattern}"))
        if hits:
            return hits[0]
    return None


def run_converter(
    converter: Path,
    dump: Path,
    template: Path | None,
    species_npz: Path | None,
    out: Path,
    dry_run: bool,
) -> tuple[bool, str]:
    if dry_run:
        return True, "dry-run"

    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        __import__("sys").executable,
        str(converter),
        "--dump",
        str(dump),
        "--out",
        str(out),
    ]
    if species_npz is not None:
        cmd.extend(["--species-npz", str(species_npz)])
    elif template is not None:
        cmd.extend(["--template", str(template)])
    else:
        return False, "converter setup failed: no template/species source"

    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        msg = (proc.stderr or proc.stdout or "")[-900:]
        return False, f"converter failed: {msg.strip()}"
    return True, "ok"


def main() -> None:
    args = parse_args()
    classical_root = args.classical_output.resolve()
    input_template = args.input_template.resolve()
    trained_model = args.trained_model.resolve()
    converter = args.converter.resolve()
    template_base = args.cg_template_data.resolve()
    cg_search_root = args.cg_npz_search_root.resolve()

    mlcg_root = (args.mlcg_out_dir.resolve() if args.mlcg_out_dir is not None
                 else classical_root / "mlcg")
    run_id_suffix = args.run_id_suffix

    cg_map = load_cg_map(args.cg_npz_map)

    if not input_template.exists():
        raise FileNotFoundError(f"missing input template: {input_template}")
    if not trained_model.exists() and not args.dry_run:
        raise FileNotFoundError(f"missing trained model: {trained_model}")
    if not converter.exists():
        raise FileNotFoundError(f"missing converter: {converter}")

    template_candidates = collect_template_candidates(template_base)
    if not template_candidates and not args.dry_run:
        raise FileNotFoundError(f"no template candidates found from: {template_base}")

    fixed_manifest = classical_root / "classical_fixed_manifest.tsv"
    if not fixed_manifest.exists():
        raise FileNotFoundError(f"missing classical fixed manifest: {fixed_manifest}")

    mlcg_root.mkdir(parents=True, exist_ok=True)

    rows = parse_fixed_manifest(fixed_manifest)
    if not rows:
        raise RuntimeError("No fixed-run rows found in classical_fixed_manifest.tsv")

    template_text = input_template.read_text()

    all_rows: list[MlcgRow] = []
    ready_rows: list[MlcgRow] = []

    for source_run_id, protein, mode, source_run_dir, _source_input in rows:
        if mode != "fixed":
            continue

        t0_dump = source_run_dir / "ca_t0.dump"
        eq_dump = source_run_dir / "ca_eq_final.dump"

        for seed_mode, dump_path in (("t0", t0_dump), ("eq", eq_dump)):
            run_id = f"{source_run_id}__mlcg_{seed_mode}{run_id_suffix}"
            run_dir = mlcg_root / seed_mode / protein / source_run_id
            data_out = run_dir / f"seed_{seed_mode}.data"
            input_out = run_dir / "in.mlcg.lmp"

            if not dump_path.exists() and not args.dry_run:
                row = MlcgRow(
                    run_id=run_id,
                    protein=protein,
                    seed_mode=seed_mode,
                    source_run_id=source_run_id,
                    source_run_dir=source_run_dir,
                    run_dir=run_dir,
                    input_file=input_out,
                    status="failed",
                    reason=f"missing seed dump: {dump_path}",
                )
                all_rows.append(row)
                continue

            dump_n_atoms = parse_dump_atom_count(dump_path) if dump_path.exists() else None
            species_npz = resolve_species_npz(protein, cg_map, cg_search_root)
            template_for_run: Path | None = None

            if species_npz is None:
                if dump_n_atoms is None and not args.dry_run:
                    row = MlcgRow(
                        run_id=run_id,
                        protein=protein,
                        seed_mode=seed_mode,
                        source_run_id=source_run_id,
                        source_run_dir=source_run_dir,
                        run_dir=run_dir,
                        input_file=input_out,
                        status="failed",
                        reason="could not parse dump atom count for template matching",
                    )
                    all_rows.append(row)
                    continue
                if dump_n_atoms is not None:
                    template_for_run = resolve_template_for_natoms(dump_n_atoms, template_candidates)
                if template_for_run is None and not args.dry_run:
                    row = MlcgRow(
                        run_id=run_id,
                        protein=protein,
                        seed_mode=seed_mode,
                        source_run_id=source_run_id,
                        source_run_dir=source_run_dir,
                        run_dir=run_dir,
                        input_file=input_out,
                        status="failed",
                        reason=(
                            f"no species npz and no template with atom count={dump_n_atoms}; "
                            f"protein={protein}"
                        ),
                    )
                    all_rows.append(row)
                    continue

            ok, reason = run_converter(
                converter,
                dump_path,
                template_for_run,
                species_npz,
                data_out,
                args.dry_run,
            )
            if not ok:
                row = MlcgRow(
                    run_id=run_id,
                    protein=protein,
                    seed_mode=seed_mode,
                    source_run_id=source_run_id,
                    source_run_dir=source_run_dir,
                    run_dir=run_dir,
                    input_file=input_out,
                    status="failed",
                    reason=reason,
                )
                all_rows.append(row)
                continue

            patched = patch_mlcg_input(
                template_text,
                data_file=data_out,
                model_file=trained_model,
                dump_dir=run_dir,
            )
            if not args.dry_run:
                run_dir.mkdir(parents=True, exist_ok=True)
                input_out.write_text(patched)
                meta = {
                    "run_id": run_id,
                    "protein": protein,
                    "seed_mode": seed_mode,
                    "source_run_id": source_run_id,
                    "source_run_dir": str(source_run_dir),
                    "seed_dump": str(dump_path),
                    "cg_data": str(data_out),
                    "input_file": str(input_out),
                    "model_file": str(trained_model),
                    "species_npz": str(species_npz) if species_npz is not None else None,
                    "template_data": str(template_for_run) if template_for_run is not None else None,
                }
                (run_dir / "run_meta.json").write_text(json.dumps(meta, indent=2))

            row = MlcgRow(
                run_id=run_id,
                protein=protein,
                seed_mode=seed_mode,
                source_run_id=source_run_id,
                source_run_dir=source_run_dir,
                run_dir=run_dir,
                input_file=input_out,
                status="ready",
                reason="",
            )
            all_rows.append(row)
            ready_rows.append(row)

    all_manifest = mlcg_root / "mlcg_manifest.tsv"
    ready_manifest = mlcg_root / "mlcg_ready_manifest.tsv"

    with all_manifest.open("w") as f:
        for r in all_rows:
            f.write(
                "\t".join(
                    [
                        r.run_id,
                        r.protein,
                        r.seed_mode,
                        r.source_run_id,
                        str(r.source_run_dir),
                        str(r.run_dir),
                        str(r.input_file),
                        r.status,
                        r.reason.replace("\t", " "),
                    ]
                )
                + "\n"
            )

    with ready_manifest.open("w") as f:
        for r in ready_rows:
            f.write("\t".join([r.run_id, r.protein, r.seed_mode, str(r.run_dir), str(r.input_file)]) + "\n")

    summary = {
        "classical_output": str(classical_root),
        "mlcg_root": str(mlcg_root),
        "run_id_suffix": run_id_suffix,
        "counts": {
            "ready": len(ready_rows),
            "failed": len([r for r in all_rows if r.status != "ready"]),
        },
        "manifests": {
            "all": str(all_manifest),
            "ready": str(ready_manifest),
        },
        "cg_npz_map": str(args.cg_npz_map.resolve()) if args.cg_npz_map else None,
        "cg_npz_search_root": str(cg_search_root),
        "template_base": str(template_base),
        "dry_run": bool(args.dry_run),
    }
    (mlcg_root / "mlcg_prepare_summary.json").write_text(json.dumps(summary, indent=2))

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
