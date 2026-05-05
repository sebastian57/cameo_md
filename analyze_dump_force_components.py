#!/usr/bin/env python3
"""Analyze selected LAMMPS dump frames with a Python-initialized CAMEO CG model.

This script:
1. Scans a LAMMPS custom dump trajectory.
2. Takes user-selected frame indices plus N random non-overlapping frames.
3. Recomputes energies and force components with a freshly initialized
   CombinedModel loaded from config + params.pkl (not MLIR).
4. Writes per-frame scalar summaries plus full force-component tensors.

The recommended species source is a matching CG NPZ that contains `species`
and, ideally, `aa_to_id`/`resname`, because typed priors may require the
original species IDs and residue-name mapping.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
WORK_ROOT = SCRIPT_DIR.parent
CAMEO_CG_ROOT = WORK_ROOT / "cameo_cg"
if str(CAMEO_CG_ROOT) not in sys.path:
    sys.path.insert(0, str(CAMEO_CG_ROOT))

from config.manager import ConfigManager
from utils.jax_setup import apply_jax_compat_shims as _apply_jax_compat_shims_global


@dataclass
class DumpFrame:
    frame_index: int
    timestep: int
    box_lengths: np.ndarray
    coords: np.ndarray
    atom_types: Optional[np.ndarray]
    forces: Optional[np.ndarray]


@dataclass
class DumpScanMeta:
    n_frames: int
    timesteps: np.ndarray
    box_lengths: np.ndarray
    atom_counts: np.ndarray
    has_types: bool
    has_forces: bool


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dump", type=Path, required=True, help="LAMMPS custom dump trajectory")
    p.add_argument("--config", type=Path, required=True, help="Training/runtime config YAML")
    p.add_argument("--params", type=Path, required=True, help="Model params pickle")
    p.add_argument(
        "--selected-frames",
        type=str,
        required=True,
        help="Comma-separated zero-based frame indices, e.g. 5,17,80",
    )
    p.add_argument(
        "--n-random",
        type=int,
        default=10,
        help="Number of random frames to add, excluding selected frames",
    )
    p.add_argument("--seed", type=int, default=42, help="RNG seed for random frame sampling")
    p.add_argument(
        "--species-npz",
        type=Path,
        default=None,
        help="Preferred source of species IDs and AA mapping",
    )
    p.add_argument(
        "--species-from-dump-types",
        action="store_true",
        help="Use dump atom types as species IDs minus one. Only use this if the dump type column matches the Python model species encoding.",
    )
    p.add_argument(
        "--outdir",
        type=Path,
        default=None,
        help="Output directory (default: dump.parent/force_component_analysis_<timestamp>)",
    )
    return p.parse_args()


def _parse_index_list(raw: str) -> list[int]:
    values: list[int] = []
    for piece in raw.replace("[", "").replace("]", "").split(","):
        text = piece.strip()
        if not text:
            continue
        value = int(text)
        if value < 0:
            raise ValueError(f"Frame indices must be >= 0, got {value}")
        values.append(value)
    if not values:
        raise ValueError("No frame indices parsed from --selected-frames")
    return values


def _choose_coord_columns(header: list[str]) -> tuple[str, str, str]:
    x_col = "xu" if "xu" in header else "x"
    y_col = "yu" if "yu" in header else "y"
    z_col = "zu" if "zu" in header else "z"
    for col in (x_col, y_col, z_col):
        if col not in header:
            raise ValueError(f"Missing coordinate column '{col}' in dump header: {header}")
    return x_col, y_col, z_col


def _parse_box_lengths(fh) -> np.ndarray:
    bounds = []
    for _ in range(3):
        cols = fh.readline().split()
        if len(cols) < 2:
            raise ValueError("Malformed BOX BOUNDS section in dump")
        lo = float(cols[0])
        hi = float(cols[1])
        bounds.append(hi - lo)
    return np.asarray(bounds, dtype=np.float32)


def scan_dump(path: Path) -> DumpScanMeta:
    timesteps: list[int] = []
    boxes: list[np.ndarray] = []
    atom_counts: list[int] = []
    has_types = False
    has_forces = False

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
                raise ValueError("Malformed dump: missing 'ITEM: NUMBER OF ATOMS'")
            n_atoms = int(fh.readline().strip())

            if not fh.readline().startswith("ITEM: BOX BOUNDS"):
                raise ValueError("Malformed dump: missing 'ITEM: BOX BOUNDS'")
            box_lengths = _parse_box_lengths(fh)

            atom_line = fh.readline().strip()
            if not atom_line.startswith("ITEM: ATOMS"):
                raise ValueError("Malformed dump: missing 'ITEM: ATOMS'")
            header = atom_line.split()[2:]

            if "id" not in header:
                raise ValueError("Dump must include atom id column")
            has_types = has_types or ("type" in header)
            has_forces = has_forces or all(col in header for col in ("fx", "fy", "fz"))

            for _ in range(n_atoms):
                row = fh.readline()
                if not row:
                    raise ValueError("Unexpected EOF while scanning dump")

            timesteps.append(timestep)
            boxes.append(box_lengths)
            atom_counts.append(n_atoms)

    if not timesteps:
        raise ValueError(f"No complete frames parsed from dump: {path}")

    return DumpScanMeta(
        n_frames=len(timesteps),
        timesteps=np.asarray(timesteps, dtype=np.int64),
        box_lengths=np.stack(boxes, axis=0),
        atom_counts=np.asarray(atom_counts, dtype=np.int32),
        has_types=has_types,
        has_forces=has_forces,
    )


def load_dump_frames(path: Path, wanted_indices: Iterable[int]) -> list[DumpFrame]:
    wanted = sorted(set(int(i) for i in wanted_indices))
    wanted_set = set(wanted)
    frames: list[DumpFrame] = []
    current_index = -1

    with path.open("r") as fh:
        while True:
            line = fh.readline()
            if not line:
                break
            if not line.startswith("ITEM: TIMESTEP"):
                continue

            current_index += 1
            ts_line = fh.readline()
            if not ts_line:
                break
            timestep = int(ts_line.strip())

            if not fh.readline().startswith("ITEM: NUMBER OF ATOMS"):
                raise ValueError("Malformed dump: missing 'ITEM: NUMBER OF ATOMS'")
            n_atoms = int(fh.readline().strip())

            if not fh.readline().startswith("ITEM: BOX BOUNDS"):
                raise ValueError("Malformed dump: missing 'ITEM: BOX BOUNDS'")
            box_lengths = _parse_box_lengths(fh)

            atom_line = fh.readline().strip()
            if not atom_line.startswith("ITEM: ATOMS"):
                raise ValueError("Malformed dump: missing 'ITEM: ATOMS'")
            header = atom_line.split()[2:]
            id_idx = header.index("id")
            x_col, y_col, z_col = _choose_coord_columns(header)
            x_idx = header.index(x_col)
            y_idx = header.index(y_col)
            z_idx = header.index(z_col)
            type_idx = header.index("type") if "type" in header else None
            fx_idx = header.index("fx") if "fx" in header else None
            fy_idx = header.index("fy") if "fy" in header else None
            fz_idx = header.index("fz") if "fz" in header else None

            take = current_index in wanted_set
            rows: list[tuple[int, float, float, float, Optional[int], Optional[float], Optional[float], Optional[float]]] = []
            for _ in range(n_atoms):
                raw = fh.readline()
                if not raw:
                    raise ValueError("Unexpected EOF while loading selected dump frames")
                if not take:
                    continue
                cols = raw.split()
                rows.append((
                    int(cols[id_idx]),
                    float(cols[x_idx]),
                    float(cols[y_idx]),
                    float(cols[z_idx]),
                    int(cols[type_idx]) if type_idx is not None else None,
                    float(cols[fx_idx]) if fx_idx is not None else None,
                    float(cols[fy_idx]) if fy_idx is not None else None,
                    float(cols[fz_idx]) if fz_idx is not None else None,
                ))

            if not take:
                continue

            rows.sort(key=lambda r: r[0])
            coords = np.empty((n_atoms, 3), dtype=np.float32)
            atom_types = np.empty((n_atoms,), dtype=np.int32) if type_idx is not None else None
            forces = np.empty((n_atoms, 3), dtype=np.float32) if fx_idx is not None and fy_idx is not None and fz_idx is not None else None
            for slot, row in enumerate(rows):
                _, x, y, z, atom_type, fx, fy, fz = row
                coords[slot] = (x, y, z)
                if atom_types is not None and atom_type is not None:
                    atom_types[slot] = atom_type
                if forces is not None and fx is not None and fy is not None and fz is not None:
                    forces[slot] = (fx, fy, fz)

            frames.append(
                DumpFrame(
                    frame_index=current_index,
                    timestep=timestep,
                    box_lengths=box_lengths,
                    coords=coords,
                    atom_types=atom_types,
                    forces=forces,
                )
            )

            if len(frames) == len(wanted):
                break

    frame_map = {frame.frame_index: frame for frame in frames}
    missing = [idx for idx in wanted if idx not in frame_map]
    if missing:
        raise ValueError(f"Failed to load requested frame indices: {missing}")
    return [frame_map[idx] for idx in wanted]


def _load_params(params_path: Path) -> Dict[str, Any]:
    with params_path.open("rb") as fh:
        payload = pickle.load(fh)

    params = payload
    if isinstance(payload, dict):
        if isinstance(payload.get("params"), dict):
            params = payload["params"]
        elif isinstance(payload.get("trainer_state"), dict) and isinstance(payload["trainer_state"].get("params"), dict):
            params = payload["trainer_state"]["params"]

    if isinstance(params, dict) and "ml" not in params and "allegro" in params:
        params = dict(params)
        params["ml"] = params["allegro"]

    if not isinstance(params, dict):
        raise TypeError(f"Unsupported params payload type: {type(params)}")
    return params


def _resolve_spline_path_if_needed(config: ConfigManager) -> None:
    if not config.use_spline_priors_enabled():
        return
    spline_path = Path(config.get_spline_file_path())
    if spline_path.is_absolute():
        resolved = spline_path
    else:
        candidates = [
            Path.cwd() / spline_path,
            CAMEO_CG_ROOT / spline_path,
            config.config_path.parent / spline_path,
        ]
        resolved = spline_path
        for candidate in candidates:
            if candidate.exists():
                resolved = candidate
                break
    config.set("model", "priors", "spline_file", str(resolved.resolve()))


def _make_species_mapping_from_resname(resname: np.ndarray) -> tuple[dict[str, int], dict[int, str]]:
    unique_aas = sorted(set(str(x) for x in np.asarray(resname).flatten()))
    aa_to_id = {aa: idx for idx, aa in enumerate(unique_aas)}
    id_to_aa = {idx: aa for aa, idx in aa_to_id.items()}
    return aa_to_id, id_to_aa


def load_species_metadata(
    *,
    dump_frame: DumpFrame,
    species_npz: Optional[Path],
    species_from_dump_types: bool,
) -> tuple[np.ndarray, dict[int, str], str]:
    if species_npz is not None:
        with np.load(species_npz, allow_pickle=True) as data:
            if "species" not in data:
                raise ValueError(f"NPZ missing species array: {species_npz}")

            species_raw = np.asarray(data["species"])
            if species_raw.ndim == 2:
                species_raw = species_raw[0]
            species = np.asarray(species_raw, dtype=np.int32)

            if "mask" in data:
                mask_raw = np.asarray(data["mask"])
                if mask_raw.ndim == 2:
                    mask_raw = mask_raw[0]
                valid = np.asarray(mask_raw) > 0.5
                if valid.shape[0] == species.shape[0]:
                    species = species[valid]

            if species.shape[0] != dump_frame.coords.shape[0]:
                raise ValueError(
                    f"species_npy atom count mismatch: dump has {dump_frame.coords.shape[0]} atoms, "
                    f"species source has {species.shape[0]}"
                )

            if "aa_to_id" in data:
                aa_to_id_payload = data["aa_to_id"].item()
                id_to_aa = {int(v): str(k) for k, v in aa_to_id_payload.items()}
            elif "resname" in data:
                resname = np.asarray(data["resname"])
                if resname.ndim == 2:
                    resname = resname[0]
                if "mask" in data and valid.shape[0] == resname.shape[0]:
                    resname = resname[valid]
                if resname.shape[0] != species.shape[0]:
                    raise ValueError(
                        f"resname/species mismatch in {species_npz}: {resname.shape[0]} vs {species.shape[0]}"
                    )
                _, id_to_aa = _make_species_mapping_from_resname(resname)
            else:
                id_to_aa = {}

            return species, id_to_aa, f"species_npz:{species_npz}"

    if species_from_dump_types:
        if dump_frame.atom_types is None:
            raise ValueError("Dump has no atom type column; cannot use --species-from-dump-types")
        species = np.asarray(dump_frame.atom_types, dtype=np.int32) - 1
        return species, {}, "dump_types_minus_one"

    raise ValueError(
        "No species source available. Provide --species-npz, or use "
        "--species-from-dump-types if the dump type column already matches the Python-model species encoding."
    )


def select_frame_indices(n_frames: int, selected: list[int], n_random: int, seed: int) -> tuple[list[int], list[int], list[int]]:
    selected_unique: list[int] = []
    seen = set()
    for idx in selected:
        if idx >= n_frames:
            raise ValueError(f"Selected frame index {idx} out of range for {n_frames} parsed frames")
        if idx in seen:
            continue
        selected_unique.append(idx)
        seen.add(idx)

    remaining = np.asarray([i for i in range(n_frames) if i not in seen], dtype=np.int32)
    if n_random < 0:
        raise ValueError("--n-random must be >= 0")
    if n_random > remaining.size:
        n_random = int(remaining.size)

    rng = np.random.default_rng(seed)
    if n_random > 0:
        random_indices = rng.choice(remaining, size=n_random, replace=False).tolist()
        random_indices = [int(x) for x in random_indices]
    else:
        random_indices = []

    final_indices = selected_unique + random_indices
    return selected_unique, random_indices, final_indices


def validate_selected_atom_counts(scan_meta: DumpScanMeta, frame_indices: Iterable[int]) -> int:
    chosen = [int(i) for i in frame_indices]
    if not chosen:
        raise ValueError("No frames selected for evaluation")
    counts = scan_meta.atom_counts[np.asarray(chosen, dtype=np.int32)]
    ref = int(counts[0])
    mismatched = [
        (idx, int(count))
        for idx, count in zip(chosen, counts.tolist())
        if int(count) != ref
    ]
    if mismatched:
        details = ", ".join(f"frame {idx}: {count} atoms" for idx, count in mismatched)
        raise ValueError(
            "Selected frames do not share a constant atom count. "
            f"Reference frame has {ref} atoms; mismatches: {details}"
        )
    return ref


def _resolve_outdir(args: argparse.Namespace) -> Path:
    if args.outdir is not None:
        return args.outdir.resolve()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return (args.dump.resolve().parent / f"force_component_analysis_{stamp}").resolve()


def _to_float(value: Any) -> float:
    return float(np.asarray(value, dtype=np.float64))


def evaluate_frame_components(
    *,
    config_path: Path,
    params: Dict[str, Any],
    frame: DumpFrame,
    species: np.ndarray,
    id_to_aa: dict[int, str],
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    import jax
    import jax.numpy as jnp
    from models.combined_model import CombinedModel

    config = ConfigManager(str(config_path))
    _resolve_spline_path_if_needed(config)
    config.set("model", "use_priors", bool(config.export_combined_ml_priors_enabled()))
    config.set("model", "train_priors", False)

    R = jnp.asarray(frame.coords, dtype=jnp.float32)
    mask = jnp.ones((frame.coords.shape[0],), dtype=jnp.float32)
    species_jax = jnp.asarray(species, dtype=jnp.int32)
    box = jnp.asarray(frame.box_lengths, dtype=jnp.float32)

    model = CombinedModel(
        config=config,
        R0=np.asarray(frame.coords, dtype=np.float32),
        box=np.asarray(frame.box_lengths, dtype=np.float32),
        species=np.asarray(species, dtype=np.int32),
        N_max=frame.coords.shape[0],
        id_to_aa=id_to_aa,
        prior_only=False,
    )

    base_components = model.compute_components(params, R, mask, species_jax, None)
    energy_keys = [str(k) for k in base_components.keys()]

    def component_tuple(R_):
        comps = model.compute_components(params, R_, mask, species_jax, None)
        return tuple(comps[key] for key in energy_keys)

    _, vjp_fn = jax.vjp(component_tuple, R)

    scalar_components = {key: _to_float(base_components[key]) for key in energy_keys}
    force_components: dict[str, np.ndarray] = {}
    for idx, key in enumerate(energy_keys):
        cotangent = tuple(1.0 if i == idx else 0.0 for i in range(len(energy_keys)))
        force_key = f"F_{key[2:]}" if key.startswith("E_") else f"F_{key}"
        force_components[force_key] = np.asarray(-vjp_fn(cotangent)[0], dtype=np.float32)

    scalar_components["box_lx"] = _to_float(box[0])
    scalar_components["box_ly"] = _to_float(box[1])
    scalar_components["box_lz"] = _to_float(box[2])
    return scalar_components, force_components


def write_outputs(
    *,
    outdir: Path,
    dump_path: Path,
    config_path: Path,
    params_path: Path,
    species_source: str,
    selected_indices: list[int],
    random_indices: list[int],
    evaluated_frames: list[DumpFrame],
    roles: dict[int, str],
    scalar_rows: list[dict[str, Any]],
    force_payload: dict[str, np.ndarray],
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    summary = {
        "dump": str(dump_path),
        "config": str(config_path),
        "params": str(params_path),
        "species_source": species_source,
        "selected_frames": selected_indices,
        "random_frames": random_indices,
        "evaluated_frames": [frame.frame_index for frame in evaluated_frames],
        "n_evaluated": len(evaluated_frames),
        "outputs": {
            "summary_json": str((outdir / "summary.json").resolve()),
            "frame_metrics_csv": str((outdir / "frame_metrics.csv").resolve()),
            "force_components_npz": str((outdir / "force_components.npz").resolve()),
        },
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))

    fieldnames = list(scalar_rows[0].keys()) if scalar_rows else []
    with (outdir / "frame_metrics.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in scalar_rows:
            writer.writerow(row)

    np.savez_compressed(outdir / "force_components.npz", **force_payload)


def main() -> None:
    args = parse_args()
    _apply_jax_compat_shims_global()

    dump_path = args.dump.resolve()
    config_path = args.config.resolve()
    params_path = args.params.resolve()
    species_npz = args.species_npz.resolve() if args.species_npz is not None else None
    outdir = _resolve_outdir(args)

    selected_requested = _parse_index_list(args.selected_frames)
    scan_meta = scan_dump(dump_path)
    selected_indices, random_indices, final_indices = select_frame_indices(
        scan_meta.n_frames,
        selected_requested,
        args.n_random,
        args.seed,
    )
    validate_selected_atom_counts(scan_meta, final_indices)

    loaded_frames = load_dump_frames(dump_path, final_indices)
    frame_map = {frame.frame_index: frame for frame in loaded_frames}
    ordered_frames = [frame_map[idx] for idx in final_indices]

    species, id_to_aa, species_source = load_species_metadata(
        dump_frame=ordered_frames[0],
        species_npz=species_npz,
        species_from_dump_types=args.species_from_dump_types,
    )
    params = _load_params(params_path)

    roles = {idx: "selected" for idx in selected_indices}
    roles.update({idx: "random" for idx in random_indices})

    scalar_rows: list[dict[str, Any]] = []
    array_accumulator: dict[str, list[np.ndarray]] = {
        "coords": [],
        "species": [],
        "mask": [],
        "box_lengths": [],
    }
    frame_indices_out: list[int] = []
    timesteps_out: list[int] = []
    role_out: list[str] = []
    dump_force_present = any(frame.forces is not None for frame in ordered_frames)
    if dump_force_present:
        array_accumulator["dump_forces"] = []

    for frame in ordered_frames:
        scalar_components, force_components = evaluate_frame_components(
            config_path=config_path,
            params=params,
            frame=frame,
            species=species,
            id_to_aa=id_to_aa,
        )

        row: dict[str, Any] = {
            "frame_index": frame.frame_index,
            "timestep": frame.timestep,
            "role": roles[frame.frame_index],
        }
        row.update(scalar_components)

        for force_key, force_value in force_components.items():
            real_force = np.asarray(force_value, dtype=np.float32)
            row[f"{force_key}_rms"] = float(np.sqrt(np.mean(np.sum(real_force ** 2, axis=1))))
            row[f"{force_key}_max_norm"] = float(np.max(np.linalg.norm(real_force, axis=1)))
            array_accumulator.setdefault(force_key, []).append(real_force)

        if frame.forces is not None:
            dump_forces = np.asarray(frame.forces, dtype=np.float32)
            total_forces = force_components["F_total"]
            diff = total_forces - dump_forces
            row["dump_force_rmse"] = float(np.sqrt(np.mean(diff ** 2)))
            row["dump_force_mae"] = float(np.mean(np.abs(diff)))
            row["dump_force_max_error"] = float(np.max(np.linalg.norm(diff, axis=1)))
            array_accumulator["dump_forces"].append(dump_forces)

        scalar_rows.append(row)
        frame_indices_out.append(frame.frame_index)
        timesteps_out.append(frame.timestep)
        role_out.append(roles[frame.frame_index])
        array_accumulator["coords"].append(np.asarray(frame.coords, dtype=np.float32))
        array_accumulator["species"].append(np.asarray(species, dtype=np.int32))
        array_accumulator["mask"].append(np.ones((frame.coords.shape[0],), dtype=np.float32))
        array_accumulator["box_lengths"].append(np.asarray(frame.box_lengths, dtype=np.float32))

    force_payload: dict[str, np.ndarray] = {
        "frame_indices": np.asarray(frame_indices_out, dtype=np.int32),
        "timesteps": np.asarray(timesteps_out, dtype=np.int64),
        "role": np.asarray(role_out, dtype=object),
    }
    for key, values in array_accumulator.items():
        force_payload[key] = np.stack(values, axis=0)

    write_outputs(
        outdir=outdir,
        dump_path=dump_path,
        config_path=config_path,
        params_path=params_path,
        species_source=species_source,
        selected_indices=selected_indices,
        random_indices=random_indices,
        evaluated_frames=ordered_frames,
        roles=roles,
        scalar_rows=scalar_rows,
        force_payload=force_payload,
    )

    print(json.dumps(
        {
            "outdir": str(outdir),
            "n_total_frames": scan_meta.n_frames,
            "selected_frames": selected_indices,
            "random_frames": random_indices,
            "evaluated_frames": final_indices,
            "species_source": species_source,
        },
        indent=2,
        sort_keys=True,
    ))


if __name__ == "__main__":
    main()
