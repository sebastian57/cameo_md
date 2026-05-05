"""Build CA-only LAMMPS data from the last frame of a CHARMM-equilibrated CA dump."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dump', type=Path, required=True, help='CA-only LAMMPS dump')
    parser.add_argument('--template', type=Path, default=None, help='Template CA LAMMPS data file')
    parser.add_argument('--species-npz', type=Path, default=None, help='CG NPZ containing species array')
    parser.add_argument('--out', type=Path, required=True, help='Output CA LAMMPS data file')
    parser.add_argument('--box-pad', type=float, default=15.0, help='Padding for box bounds (Angstrom)')
    return parser.parse_args()


def parse_last_dump_frame(path: Path):
    last = None
    with path.open('r') as fh:
        while True:
            line = fh.readline()
            if not line:
                break
            if not line.startswith('ITEM: TIMESTEP'):
                continue

            fh.readline()
            if not fh.readline().startswith('ITEM: NUMBER OF ATOMS'):
                continue
            n_atoms = int(fh.readline().strip())

            if not fh.readline().startswith('ITEM: BOX BOUNDS'):
                continue
            fh.readline()
            fh.readline()
            fh.readline()

            atom_line = fh.readline().strip()
            if not atom_line.startswith('ITEM: ATOMS'):
                continue
            header = atom_line.split()[2:]

            if 'id' not in header:
                raise ValueError('Dump must contain id column')
            id_idx = header.index('id')
            x_col = 'xu' if 'xu' in header else 'x'
            y_col = 'yu' if 'yu' in header else 'y'
            z_col = 'zu' if 'zu' in header else 'z'
            if x_col not in header or y_col not in header or z_col not in header:
                raise ValueError(f'Missing coordinate columns in dump header: {header}')
            x_idx = header.index(x_col)
            y_idx = header.index(y_col)
            z_idx = header.index(z_col)

            rows = []
            ok = True
            for _ in range(n_atoms):
                raw = fh.readline()
                if not raw:
                    ok = False
                    break
                cols = raw.split()
                try:
                    rows.append((int(cols[id_idx]), float(cols[x_idx]), float(cols[y_idx]), float(cols[z_idx])))
                except Exception:
                    ok = False
                    break

            if ok:
                rows.sort(key=lambda r: r[0])
                last = rows

    if last is None:
        raise ValueError(f'No complete frames parsed from dump: {path}')
    return last


def parse_template_atoms(path: Path):
    atoms = []
    in_atoms = False
    with path.open('r') as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped == 'Atoms':
                in_atoms = True
                continue
            if not in_atoms:
                continue
            if stripped[0].isalpha():
                break
            cols = line.split()
            if len(cols) < 5:
                continue
            atom_id = int(cols[0])
            atom_type = int(cols[1])
            atoms.append((atom_id, atom_type))

    if not atoms:
        raise ValueError(f'No atoms parsed from template: {path}')
    return atoms


def parse_species_types(npz_path: Path) -> list[int]:
    if not npz_path.exists():
        raise FileNotFoundError(f'species npz not found: {npz_path}')

    with np.load(npz_path, allow_pickle=True) as data:
        if 'species' not in data:
            raise ValueError(f'NPZ missing species array: {npz_path}')
        species = np.asarray(data['species'])

    if species.ndim == 2:
        if species.shape[0] < 1:
            raise ValueError(f'Empty species matrix in: {npz_path}')
        species_0 = species[0]
    elif species.ndim == 1:
        species_0 = species
    else:
        raise ValueError(f'Unsupported species shape {species.shape} in {npz_path}')

    species_0 = np.asarray(species_0, dtype=int)
    if species_0.size == 0:
        raise ValueError(f'Empty species vector in: {npz_path}')
    if species_0.min() < 0:
        raise ValueError(f'Negative species indices in: {npz_path}')

    return [int(v) + 1 for v in species_0.tolist()]


def write_data_file(out: Path, atom_types, xyz, pad: float) -> None:
    xs = [c[0] for c in xyz]
    ys = [c[1] for c in xyz]
    zs = [c[2] for c in xyz]
    xlo, xhi = min(xs) - pad, max(xs) + pad
    ylo, yhi = min(ys) - pad, max(ys) + pad
    zlo, zhi = min(zs) - pad, max(zs) + pad

    n_atoms = len(atom_types)
    n_types = max(t for _, t in atom_types)

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w') as fh:
        fh.write('LAMMPS data file from CHARMM22-equilibrated CA snapshot\n\n')
        fh.write(f'{n_atoms} atoms\n')
        fh.write(f'{n_types} atom types\n\n')
        fh.write(f'{xlo:.6f} {xhi:.6f} xlo xhi\n')
        fh.write(f'{ylo:.6f} {yhi:.6f} ylo yhi\n')
        fh.write(f'{zlo:.6f} {zhi:.6f} zlo zhi\n\n')
        fh.write('Atoms\n\n')
        for (atom_id, atom_type), (x, y, z) in zip(atom_types, xyz):
            fh.write(f'{atom_id} {atom_type} {x:.8f} {y:.8f} {z:.8f}\n')


def main() -> None:
    args = parse_args()
    if args.template is None and args.species_npz is None:
        raise ValueError('Provide either --template or --species-npz')

    last_frame = parse_last_dump_frame(args.dump)
    xyz = [(x, y, z) for _, x, y, z in last_frame]

    if args.species_npz is not None:
        species_types = parse_species_types(args.species_npz)
        if len(species_types) != len(last_frame):
            raise ValueError(
                f'Atom-count mismatch: dump has {len(last_frame)} atoms, species has {len(species_types)} entries'
            )
        atom_types = [(i + 1, t) for i, t in enumerate(species_types)]
        source = f'species-npz={args.species_npz}'
    else:
        template_atoms = parse_template_atoms(args.template)
        if len(last_frame) != len(template_atoms):
            raise ValueError(
                f'Atom-count mismatch: dump has {len(last_frame)} atoms, template has {len(template_atoms)} atoms'
            )
        atom_types = template_atoms
        source = f'template={args.template}'

    write_data_file(args.out, atom_types, xyz, args.box_pad)

    print(f'Wrote: {args.out}')
    print(f'Source dump: {args.dump}')
    print(f'Atom-type source: {source}')


if __name__ == '__main__':
    main()
