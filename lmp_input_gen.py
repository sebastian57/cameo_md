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
        default=2234,
        help="Frame index to export.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output LAMMPS data path (default: structures/config_<frame>.data).",
    )
    parser.add_argument(
        "--box-mode",
        choices=["auto", "manual"],
        default="auto",
        help="How to set box bounds. auto uses coordinate extents with a safety factor.",
    )
    parser.add_argument(
        "--safety-factor",
        type=float,
        default=1.20,
        help="Multiplier for farthest absolute extent in each axis when --box-mode auto is used.",
    )
    parser.add_argument(
        "--min-half-width",
        type=float,
        default=20.0,
        help="Minimum half box width per axis for auto mode.",
    )
    parser.add_argument("--xlo", type=float, default=-400.0)
    parser.add_argument("--xhi", type=float, default=400.0)
    parser.add_argument("--ylo", type=float, default=-400.0)
    parser.add_argument("--yhi", type=float, default=400.0)
    parser.add_argument("--zlo", type=float, default=-400.0)
    parser.add_argument("--zhi", type=float, default=400.0)
    return parser.parse_args()


def _compute_auto_bounds(pos: np.ndarray, safety_factor: float, min_half_width: float) -> tuple[float, float, float, float, float, float]:
    if safety_factor <= 1.0:
        raise ValueError("--safety-factor must be > 1.0")
    if min_half_width <= 0.0:
        raise ValueError("--min-half-width must be > 0")

    max_abs = np.max(np.abs(pos), axis=0)
    half_width = np.maximum(max_abs * safety_factor, min_half_width)

    xh, yh, zh = float(half_width[0]), float(half_width[1]), float(half_width[2])
    return -xh, xh, -yh, yh, -zh, zh


def main() -> None:
    args = parse_args()

    # NPZ contains object arrays (e.g. resname/paths/mappings), so pickle support is required.
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

    # Remap species on kept atoms so LAMMPS types are contiguous 1..N.
    species_kept = species_all[valid]
    unique_species = sorted(set(int(x) for x in species_kept.tolist()))
    species_to_type = {sp: idx + 1 for idx, sp in enumerate(unique_species)}
    lammps_types = np.array([species_to_type[int(sp)] for sp in species_kept], dtype=np.int32)
    n_species = len(unique_species)

    if args.box_mode == "auto":
        xlo, xhi, ylo, yhi, zlo, zhi = _compute_auto_bounds(pos, args.safety_factor, args.min_half_width)
    else:
        xlo, xhi, ylo, yhi, zlo, zhi = args.xlo, args.xhi, args.ylo, args.yhi, args.zlo, args.zhi

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
            x, y, z = pos[idx]
            f.write(f"{idx + 1} {int(lammps_types[idx])} {x:.8f} {y:.8f} {z:.8f}\n")

    print(f"dataset      : {args.dataset}")
    print(f"frame        : {args.frame}")
    print(f"atoms total  : {n_total}")
    print(f"atoms kept   : {n_atoms}")
    print(f"atom types   : {n_species}")
    print(f"box mode     : {args.box_mode}")
    print(f"box x        : [{xlo:.6f}, {xhi:.6f}]")
    print(f"box y        : [{ylo:.6f}, {yhi:.6f}]")
    print(f"box z        : [{zlo:.6f}, {zhi:.6f}]")
    print(f"wrote        : {out_path}")


if __name__ == "__main__":
    main()
