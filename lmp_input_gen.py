import argparse
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a LAMMPS data file from an NPZ dataset frame.")
    parser.add_argument(
        "--dataset",
        default="structures/datasets/2g4q4z5k_320K_kcalmol_1bead_notnorm_aggforce.npz",
        help="Path to NPZ dataset.",
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
        help="Output LAMMPS data path (default: structures/config_<frame>.data).",
    )
    parser.add_argument(
        "--cutoff",
        type=float,
        default=12.0,
        help="Model pair cutoff in Å. Box is padded so no atom is within this distance "
             "of a periodic image. Should match the LAMMPS pair_style cutoff "
             "(training cutoff + any buffer). Default: 12.0.",
    )
    parser.add_argument(
        "--padding",
        type=float,
        default=5.0,
        help="Extra clearance in Å beyond the cutoff to the box edge (default: 5.0).",
    )
    return parser.parse_args()


def _center_and_box(
    pos: np.ndarray, cutoff: float, padding: float
) -> tuple[np.ndarray, float, float, float, float, float, float]:
    """
    Center protein at origin and compute symmetric box bounds such that no
    atom is within (cutoff + padding) Å of the nearest periodic image.

    The half-width in each axis = max_extent_from_center + cutoff + padding,
    where max_extent is the farthest atom from the centroid along that axis.
    """
    centroid = pos.mean(axis=0)
    pos_centered = pos - centroid

    # Per-axis maximum absolute displacement from centroid
    max_extent = np.max(np.abs(pos_centered), axis=0)  # (3,)

    half_width = max_extent + cutoff + padding

    xh, yh, zh = float(half_width[0]), float(half_width[1]), float(half_width[2])
    return pos_centered, -xh, xh, -yh, yh, -zh, zh


def main() -> None:
    args = parse_args()

    data = np.load(args.dataset, allow_pickle=True)

    n_frames = data["R"].shape[0]
    if args.frame < 0 or args.frame >= n_frames:
        raise IndexError(f"--frame {args.frame} out of range [0, {n_frames - 1}]")

    pos_all = np.asarray(data["R"][args.frame], dtype=np.float64)
    n_total = pos_all.shape[0]

    # Keep only real atoms (exclude padded rows).
    if "mask" in data.files:
        valid = np.asarray(data["mask"][args.frame]) > 0.5
    elif "n_atoms" in data.files:
        n_real = int(data["n_atoms"][args.frame])
        valid = np.zeros(n_total, dtype=bool)
        valid[:n_real] = True
    else:
        valid = np.ones(n_total, dtype=bool)

    if valid.shape[0] != n_total:
        raise ValueError(f"mask length ({valid.shape[0]}) does not match atom count ({n_total})")

    pos = pos_all[valid]
    n_atoms = pos.shape[0]

    if "species" in data.files:
        species_all = np.asarray(data["species"][args.frame], dtype=np.int32)
    else:
        resnames = np.asarray(data["resname"][args.frame], dtype=object)
        unique_aa = sorted({str(x) for x in resnames})
        aa_to_id = {aa: idx for idx, aa in enumerate(unique_aa)}
        species_all = np.array([aa_to_id[str(aa)] for aa in resnames], dtype=np.int32)

    if species_all.shape[0] != n_total:
        raise ValueError(
            f"species length ({species_all.shape[0]}) does not match atom count ({n_total}) for frame {args.frame}"
        )

    # Preserve the original dataset species IDs exactly.
    # The exported MLIR path expects LAMMPS atom types to be species + 1,
    # so any local remapping would change model semantics.
    species_kept = species_all[valid]
    if species_kept.size == 0:
        raise ValueError(f"no valid species found for frame {args.frame}")
    if np.min(species_kept) < 0:
        raise ValueError(
            f"species IDs must be non-negative, got range "
            f"[{int(np.min(species_kept))}, {int(np.max(species_kept))}]"
        )
    lammps_types = np.asarray(species_kept, dtype=np.int32) + 1
    n_species = int(np.max(species_kept)) + 1

    # Center protein at origin and compute box with guaranteed clearance.
    pos_centered, xlo, xhi, ylo, yhi, zlo, zhi = _center_and_box(
        pos, args.cutoff, args.padding
    )

    max_extent = np.max(np.abs(pos_centered), axis=0)
    clearance = np.array([xhi, yhi, zhi]) - max_extent  # should equal cutoff + padding
    box_lengths = np.array([xhi - xlo, yhi - ylo, zhi - zlo])

    out_path = Path(args.out) if args.out else Path(f"structures/config_{args.frame}.data")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w") as f:
        f.write("LAMMPS data file via lmp_input_gen.py\n\n")
        f.write(f"{n_atoms} atoms\n")
        f.write(f"{n_species} atom types\n\n")
        f.write(f"{xlo:.6f} {xhi:.6f} xlo xhi\n")
        f.write(f"{ylo:.6f} {yhi:.6f} ylo yhi\n")
        f.write(f"{zlo:.6f} {zhi:.6f} zlo zhi\n\n")
        f.write("Atoms\n\n")

        for idx in range(n_atoms):
            x, y, z = pos_centered[idx]
            f.write(f"{idx + 1} {int(lammps_types[idx])} {x:.8f} {y:.8f} {z:.8f}\n")

    print(f"dataset      : {args.dataset}")
    print(f"frame        : {args.frame}")
    print(f"atoms total  : {n_total}")
    print(f"atoms kept   : {n_atoms}")
    print(f"atom types   : {n_species}")
    print(f"species min  : {int(np.min(species_kept))}")
    print(f"species max  : {int(np.max(species_kept))}")
    print(f"cutoff       : {args.cutoff} Å  (pair interaction range)")
    print(f"padding      : {args.padding} Å  (extra clearance beyond cutoff)")
    print(f"box x        : [{xlo:.3f}, {xhi:.3f}]  width={box_lengths[0]:.3f} Å  clearance={clearance[0]:.3f} Å")
    print(f"box y        : [{ylo:.3f}, {yhi:.3f}]  width={box_lengths[1]:.3f} Å  clearance={clearance[1]:.3f} Å")
    print(f"box z        : [{zlo:.3f}, {zhi:.3f}]  width={box_lengths[2]:.3f} Å  clearance={clearance[2]:.3f} Å")
    print(f"wrote        : {out_path}")


if __name__ == "__main__":
    main()
