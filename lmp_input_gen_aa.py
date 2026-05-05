#!/usr/bin/env python3
"""Generate a LAMMPS atomic data file from an all-atom NPZ dataset frame.

This is the all-atom counterpart to lmp_input_gen.py.  It preserves the
dataset species IDs as LAMMPS atom types via type = species + 1, matching the
chemtrain_deploy convention that converts LAMMPS types back with species - 1.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = SCRIPT_DIR.parent
DEFAULT_DATASET = (
    WORKSPACE_ROOT
    / "cameo_cg"
    / "data_prep"
    / "datasets"
    / "1pro_4zohB01_alltemp_aa"
    / "4zohB01_alltemp_aa.npz"
)

ELEMENT_MASSES = {
    "H": 1.008,
    "C": 12.011,
    "N": 14.007,
    "O": 15.999,
    "S": 32.06,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a LAMMPS all-atom data file from an NPZ dataset frame."
    )
    parser.add_argument(
        "--dataset",
        default=str(DEFAULT_DATASET),
        help=f"Path to all-atom NPZ dataset (default: {DEFAULT_DATASET}).",
    )
    parser.add_argument(
        "--frame",
        type=int,
        default=0,
        help="Frame index to export.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output LAMMPS data path (default: cameo_md/structures/config_aa_<frame>.data).",
    )
    parser.add_argument(
        "--cutoff",
        type=float,
        default=12.0,
        help=(
            "Clearance cutoff in Angstrom. Use the LAMMPS communication cutoff "
            "or larger to avoid periodic-image interactions (default: 12.0)."
        ),
    )
    parser.add_argument(
        "--padding",
        type=float,
        default=5.0,
        help="Extra clearance in Angstrom beyond --cutoff (default: 5.0).",
    )
    return parser.parse_args()


def _center_and_box(
    pos: np.ndarray, cutoff: float, padding: float
) -> tuple[np.ndarray, float, float, float, float, float, float]:
    centroid = pos.mean(axis=0)
    pos_centered = pos - centroid
    max_extent = np.max(np.abs(pos_centered), axis=0)
    half_width = max_extent + cutoff + padding
    xh, yh, zh = (float(v) for v in half_width)
    return pos_centered, -xh, xh, -yh, yh, -zh, zh


def _frame_or_static(array: np.ndarray, frame: int, n_frames: int, name: str) -> np.ndarray:
    if array.shape[0] == n_frames:
        return np.asarray(array[frame])
    return np.asarray(array)


def _load_valid_mask(data: np.lib.npyio.NpzFile, frame: int, n_total: int) -> np.ndarray:
    if "mask" in data.files:
        mask = np.asarray(data["mask"][frame]) > 0.5
    elif "n_atoms" in data.files:
        n_real = int(np.asarray(data["n_atoms"])[frame])
        mask = np.zeros(n_total, dtype=bool)
        mask[:n_real] = True
    else:
        mask = np.ones(n_total, dtype=bool)

    if mask.shape[0] != n_total:
        raise ValueError(f"mask length ({mask.shape[0]}) does not match atom count ({n_total})")
    return mask


def _species_masses(
    species: np.ndarray,
    elements: np.ndarray | None,
    lammps_types: np.ndarray,
) -> dict[int, float]:
    masses: dict[int, float] = {}
    for atom_type in sorted(set(int(t) for t in lammps_types.tolist())):
        atom_mask = lammps_types == atom_type
        if elements is None:
            masses[atom_type] = 12.011
            continue

        elems = sorted(set(str(e) for e in elements[atom_mask].tolist()))
        if len(elems) != 1:
            sp_values = sorted(set(int(s) for s in species[atom_mask].tolist()))
            raise ValueError(
                f"LAMMPS type {atom_type} maps to multiple elements {elems} "
                f"for species {sp_values}; cannot assign a single mass."
            )
        elem = elems[0]
        if elem not in ELEMENT_MASSES:
            raise ValueError(f"No mass configured for element {elem!r}")
        masses[atom_type] = ELEMENT_MASSES[elem]
    return masses


def main() -> None:
    args = parse_args()

    dataset_path = Path(args.dataset).resolve()
    data = np.load(dataset_path, allow_pickle=True)

    if "R" not in data.files:
        raise ValueError(f"Dataset is missing required 'R' array: {dataset_path}")

    n_frames = data["R"].shape[0]
    if args.frame < 0 or args.frame >= n_frames:
        raise IndexError(f"--frame {args.frame} out of range [0, {n_frames - 1}]")

    pos_all = np.asarray(data["R"][args.frame], dtype=np.float64)
    n_total = pos_all.shape[0]
    valid = _load_valid_mask(data, args.frame, n_total)
    pos = pos_all[valid]

    if "species" not in data.files:
        raise ValueError("All-atom export requires a 'species' array in the NPZ dataset.")
    species_all = _frame_or_static(np.asarray(data["species"], dtype=np.int32), args.frame, n_frames, "species")
    if species_all.shape[0] != n_total:
        raise ValueError(
            f"species length ({species_all.shape[0]}) does not match atom count ({n_total})"
        )
    species = species_all[valid]
    if np.min(species) < 0:
        raise ValueError("Negative species IDs cannot be exported to LAMMPS atom types.")

    lammps_types = species.astype(np.int32) + 1
    n_atom_types = int(np.max(lammps_types))

    elements = None
    if "element" in data.files:
        elements_all = _frame_or_static(np.asarray(data["element"]), args.frame, n_frames, "element")
        if elements_all.shape[0] != n_total:
            raise ValueError(
                f"element length ({elements_all.shape[0]}) does not match atom count ({n_total})"
            )
        elements = elements_all[valid]
    masses = _species_masses(species, elements, lammps_types)

    pos_centered, xlo, xhi, ylo, yhi, zlo, zhi = _center_and_box(
        pos, args.cutoff, args.padding
    )

    max_extent = np.max(np.abs(pos_centered), axis=0)
    clearance = np.array([xhi, yhi, zhi]) - max_extent
    box_lengths = np.array([xhi - xlo, yhi - ylo, zhi - zlo])

    out_path = (
        Path(args.out).resolve()
        if args.out
        else SCRIPT_DIR / "structures" / f"config_aa_{args.frame}.data"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w") as f:
        f.write("LAMMPS all-atom data file via lmp_input_gen_aa.py\n\n")
        f.write(f"{pos.shape[0]} atoms\n")
        f.write(f"{n_atom_types} atom types\n\n")
        f.write(f"{xlo:.6f} {xhi:.6f} xlo xhi\n")
        f.write(f"{ylo:.6f} {yhi:.6f} ylo yhi\n")
        f.write(f"{zlo:.6f} {zhi:.6f} zlo zhi\n\n")
        f.write("Masses\n\n")
        for atom_type in range(1, n_atom_types + 1):
            if atom_type not in masses:
                raise ValueError(f"No atoms found for required LAMMPS type {atom_type}")
            f.write(f"{atom_type} {masses[atom_type]:.6f}\n")
        f.write("\nAtoms\n\n")

        for idx, (atom_type, xyz) in enumerate(zip(lammps_types, pos_centered), start=1):
            x, y, z = xyz
            f.write(f"{idx} {int(atom_type)} {x:.8f} {y:.8f} {z:.8f}\n")

    print(f"dataset      : {dataset_path}")
    print(f"frame        : {args.frame}")
    print(f"atoms total  : {n_total}")
    print(f"atoms kept   : {pos.shape[0]}")
    print(f"atom types   : {n_atom_types} (LAMMPS type = species + 1)")
    print(f"cutoff       : {args.cutoff} Ang  (clearance basis)")
    print(f"padding      : {args.padding} Ang")
    print(f"box x        : [{xlo:.3f}, {xhi:.3f}]  width={box_lengths[0]:.3f} Ang  clearance={clearance[0]:.3f} Ang")
    print(f"box y        : [{ylo:.3f}, {yhi:.3f}]  width={box_lengths[1]:.3f} Ang  clearance={clearance[1]:.3f} Ang")
    print(f"box z        : [{zlo:.3f}, {zhi:.3f}]  width={box_lengths[2]:.3f} Ang  clearance={clearance[2]:.3f} Ang")
    print(f"wrote        : {out_path}")


if __name__ == "__main__":
    main()
