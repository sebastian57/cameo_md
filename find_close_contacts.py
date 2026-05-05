#!/usr/bin/env python3
"""
find_close_contacts.py — detect bead pairs below a distance cutoff in a LAMMPS
dump trajectory and report whether they precede protein dissociation.

For each frame where at least one close contact exists, prints:
  - timestep
  - all pairs (atom IDs, types, distance)
  - forces on those atoms (if fx/fy/fz are present in the dump)
  - radius of gyration at that frame and the next N frames (dissociation proxy)

Forces are only available if the LAMMPS dump includes them. To add forces,
change the dump command in the LAMMPS input to:
    dump dprod all custom <stride> <file> id type xu yu zu fx fy fz

Usage:
    python find_close_contacts.py <dump_file> [options]

    --cutoff       Distance cutoff in Å (default: 3.0)
    --context      Number of frames after a contact to track Rg (default: 10)
    --rg-threshold Rg increase (Å) from contact frame to flag dissociation (default: 5.0)
    --csv          Write summary table to this CSV file
    --quiet        Suppress per-frame verbose output, only print summary
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Generator, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# LAMMPS dump parser
# ---------------------------------------------------------------------------

@dataclass
class Frame:
    timestep: int
    n_atoms: int
    box: np.ndarray          # (3, 2)  [[xlo, xhi], [ylo, yhi], [zlo, zhi]]
    columns: List[str]
    data: np.ndarray         # (n_atoms, n_cols)  sorted by atom id

    def col(self, name: str) -> Optional[np.ndarray]:
        if name in self.columns:
            return self.data[:, self.columns.index(name)]
        return None

    @property
    def positions(self) -> np.ndarray:
        x = self.col("xu") if self.col("xu") is not None else self.col("x")
        y = self.col("yu") if self.col("yu") is not None else self.col("y")
        z = self.col("zu") if self.col("zu") is not None else self.col("z")
        if x is None:
            raise ValueError("No position columns found in dump")
        return np.stack([x, y, z], axis=1)

    @property
    def forces(self) -> Optional[np.ndarray]:
        fx, fy, fz = self.col("fx"), self.col("fy"), self.col("fz")
        if fx is None:
            return None
        return np.stack([fx, fy, fz], axis=1)

    @property
    def atom_ids(self) -> np.ndarray:
        return self.col("id").astype(int)

    @property
    def atom_types(self) -> np.ndarray:
        return self.col("type").astype(int)

    @property
    def box_lengths(self) -> np.ndarray:
        return self.box[:, 1] - self.box[:, 0]


def _parse_dump(path: Path) -> Generator[Frame, None, None]:
    with open(path) as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        # Expect ITEM: TIMESTEP
        if not lines[i].startswith("ITEM: TIMESTEP"):
            i += 1
            continue
        timestep = int(lines[i + 1].strip())
        n_atoms  = int(lines[i + 3].strip())

        # Box bounds
        box = np.zeros((3, 2))
        for dim in range(3):
            lo, hi = map(float, lines[i + 5 + dim].split())
            box[dim] = [lo, hi]

        # Column header
        col_line = lines[i + 8].strip()  # "ITEM: ATOMS id type xu yu zu ..."
        columns = col_line.replace("ITEM: ATOMS", "").split()

        # Atom data
        rows = []
        for j in range(n_atoms):
            rows.append(list(map(float, lines[i + 9 + j].split())))
        data = np.array(rows)

        # Sort by atom id
        id_col = columns.index("id")
        order = np.argsort(data[:, id_col])
        data = data[order]

        yield Frame(timestep=timestep, n_atoms=n_atoms, box=box,
                    columns=columns, data=data)
        i += 9 + n_atoms


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _min_image_dist(ri: np.ndarray, rj: np.ndarray, box_lengths: np.ndarray) -> float:
    """Minimum-image distance between two positions in a periodic box."""
    dr = rj - ri
    dr -= box_lengths * np.round(dr / box_lengths)
    return float(np.linalg.norm(dr))


def _pairwise_distances(pos: np.ndarray, box_lengths: np.ndarray) -> np.ndarray:
    """
    Return upper-triangle pairwise distances, shape (n*(n-1)/2,).
    Also return index pairs.
    """
    n = len(pos)
    dists, pairs = [], []
    for i in range(n):
        for j in range(i + 1, n):
            d = _min_image_dist(pos[i], pos[j], box_lengths)
            dists.append(d)
            pairs.append((i, j))
    return np.array(dists), pairs


def _radius_of_gyration(pos: np.ndarray) -> float:
    com = pos.mean(axis=0)
    return float(np.sqrt(np.mean(np.sum((pos - com) ** 2, axis=1))))


# ---------------------------------------------------------------------------
# Close-contact event
# ---------------------------------------------------------------------------

@dataclass
class DistanceSample:
    frame_idx: int
    timestep: int
    mean_dist: float   # mean of all pairwise distances
    min_dist: float    # minimum pairwise distance
    max_dist: float    # maximum pairwise distance (proxy for extent)


@dataclass
class ContactEvent:
    timestep: int
    frame_idx: int
    pairs: List[Tuple[int, int, int, int, float]]   # (id_i, id_j, type_i, type_j, dist)
    forces_on_contact: Optional[np.ndarray]          # (n_contact_atoms, 3)
    contact_atom_ids: List[int]
    rg_at_contact: float
    rg_context: List[float] = field(default_factory=list)   # Rg in subsequent frames

    @property
    def rg_increase(self) -> Optional[float]:
        if not self.rg_context:
            return None
        return max(self.rg_context) - self.rg_at_contact


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def analyse(dump_path: Path, cutoff: float, context_frames: int,
            rg_threshold: float, quiet: bool,
            n_sample: int = 10) -> Tuple[List[ContactEvent], List[DistanceSample]]:

    # First pass: collect all frames into memory (needed for context Rg)
    print(f"Reading {dump_path} ...", flush=True)
    frames = list(_parse_dump(dump_path))
    print(f"  {len(frames)} frames,  {frames[0].n_atoms} atoms per frame")

    has_forces = frames[0].forces is not None
    if not has_forces:
        print("  NOTE: dump does not contain forces (fx fy fz).")
        print("        Add 'fx fy fz' to the LAMMPS dump command to enable force reporting.")
    print()

    # Distance samples at n_sample evenly-spaced frames
    n_frames = len(frames)
    sample_indices = [int(round(i * (n_frames - 1) / max(n_sample - 1, 1)))
                      for i in range(n_sample)]
    sample_indices = sorted(set(sample_indices))  # deduplicate if n_sample > n_frames
    distance_samples: List[DistanceSample] = []
    for fi in sample_indices:
        frame = frames[fi]
        d, _ = _pairwise_distances(frame.positions, frame.box_lengths)
        distance_samples.append(DistanceSample(
            frame_idx=fi,
            timestep=frame.timestep,
            mean_dist=float(np.mean(d)),
            min_dist=float(np.min(d)),
            max_dist=float(np.max(d)),
        ))

    events: List[ContactEvent] = []

    for fi, frame in enumerate(frames):
        pos    = frame.positions
        bl     = frame.box_lengths
        ids    = frame.atom_ids
        types  = frame.atom_types
        dists, pairs = _pairwise_distances(pos, bl)

        close_mask = dists < cutoff
        if not close_mask.any():
            continue

        # Collect close pairs
        close_pairs = []
        contact_atom_set = set()
        for k, (i, j) in enumerate(pairs):
            if close_mask[k]:
                close_pairs.append((int(ids[i]), int(ids[j]),
                                    int(types[i]), int(types[j]),
                                    float(dists[k])))
                contact_atom_set.add(i)
                contact_atom_set.add(j)

        contact_idxs = sorted(contact_atom_set)
        contact_ids  = [int(ids[ci]) for ci in contact_idxs]

        # Forces on contact atoms
        if has_forces:
            frc = frame.forces
            forces_on_contact = frc[contact_idxs]
        else:
            forces_on_contact = None

        rg_now = _radius_of_gyration(pos)

        ev = ContactEvent(
            timestep=frame.timestep,
            frame_idx=fi,
            pairs=close_pairs,
            forces_on_contact=forces_on_contact,
            contact_atom_ids=contact_ids,
            rg_at_contact=rg_now,
        )

        # Collect context Rg
        for fj in range(fi + 1, min(fi + context_frames + 1, len(frames))):
            ev.rg_context.append(_radius_of_gyration(frames[fj].positions))

        events.append(ev)

        if not quiet:
            _print_event(ev, has_forces, rg_threshold)

    return events, distance_samples


def _print_event(ev: ContactEvent, has_forces: bool, rg_threshold: float) -> None:
    dissoc = ev.rg_increase is not None and ev.rg_increase >= rg_threshold
    flag   = "  *** POSSIBLE DISSOCIATION ***" if dissoc else ""
    print(f"{'='*70}")
    print(f"Timestep {ev.timestep:>10d}  (frame {ev.frame_idx}){flag}")
    print(f"  Rg at contact: {ev.rg_at_contact:.2f} Å")
    if ev.rg_context:
        print(f"  Rg next {len(ev.rg_context)} frames: "
              f"min={min(ev.rg_context):.2f}  max={max(ev.rg_context):.2f}  "
              f"final={ev.rg_context[-1]:.2f} Å")
        if ev.rg_increase is not None:
            print(f"  Max Rg increase: {ev.rg_increase:+.2f} Å")

    print(f"\n  Close pairs ({len(ev.pairs)}):")
    print(f"    {'id_i':>6} {'type_i':>6} {'id_j':>6} {'type_j':>6} {'dist (Å)':>10}")
    for id_i, id_j, t_i, t_j, d in sorted(ev.pairs, key=lambda x: x[4]):
        print(f"    {id_i:>6} {t_i:>6} {id_j:>6} {t_j:>6} {d:>10.4f}")

    if has_forces and ev.forces_on_contact is not None:
        print(f"\n  Forces on contact atoms [kcal/(mol·Å)]:")
        print(f"    {'atom_id':>8} {'fx':>12} {'fy':>12} {'fz':>12} {'|f|':>12}")
        for ci, aid in enumerate(ev.contact_atom_ids):
            fx, fy, fz = ev.forces_on_contact[ci]
            fnorm = np.linalg.norm(ev.forces_on_contact[ci])
            print(f"    {aid:>8} {fx:>12.4f} {fy:>12.4f} {fz:>12.4f} {fnorm:>12.4f}")
    print()


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def _print_forces_block(ev: ContactEvent) -> None:
    """Print forces on all contact atoms for one event."""
    if ev.forces_on_contact is None:
        print("    (forces not available — add fx fy fz to LAMMPS dump command)")
        return
    print(f"    {'atom_id':>8} {'fx':>12} {'fy':>12} {'fz':>12} {'|f|':>12}")
    for ci, aid in enumerate(ev.contact_atom_ids):
        fx, fy, fz = ev.forces_on_contact[ci]
        fnorm = float(np.linalg.norm(ev.forces_on_contact[ci]))
        print(f"    {aid:>8} {fx:>12.4f} {fy:>12.4f} {fz:>12.4f} {fnorm:>12.4f}")


def _print_distance_samples(samples: List[DistanceSample]) -> None:
    print("=" * 70)
    print("AVERAGE BEAD DISTANCES AT SAMPLED FRAMES")
    print(f"  {'frame':>8} {'timestep':>12} {'mean dist (Å)':>15} {'min dist (Å)':>14} {'max dist (Å)':>14}")
    for s in samples:
        print(f"  {s.frame_idx:>8} {s.timestep:>12} {s.mean_dist:>15.4f} {s.min_dist:>14.4f} {s.max_dist:>14.4f}")
    # Drift indicator
    if len(samples) >= 2:
        drift = samples[-1].mean_dist - samples[0].mean_dist
        print(f"\n  Mean distance drift (last – first): {drift:+.4f} Å")
    print()


def _print_summary(events: List[ContactEvent], rg_threshold: float) -> None:
    if not events:
        print("No close contacts found.")
        return

    from collections import Counter

    dissoc_events = [e for e in events if (e.rg_increase or 0) >= rg_threshold]
    has_forces = events[0].forces_on_contact is not None

    all_pairs = [(e, pair) for e in events for pair in e.pairs]
    all_dists = [p[4] for _, p in all_pairs]

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Contact frames:        {len(events)}")
    print(f"  Dissociation events:   {len(dissoc_events)}")
    print(f"  Has forces in dump:    {'yes' if has_forces else 'no (add fx fy fz to dump)'}")
    print()

    # ── 1. First frame where cutoff is violated ──────────────────────────────
    first_ev = events[0]
    first_pair = min(first_ev.pairs, key=lambda p: p[4])
    print("1) FIRST CUTOFF VIOLATION")
    print(f"   Timestep {first_ev.timestep}  (frame {first_ev.frame_idx})")
    print(f"   Closest pair: atoms {first_pair[0]} – {first_pair[1]}"
          f"  (types {first_pair[2]}, {first_pair[3]})  dist = {first_pair[4]:.4f} Å")
    if has_forces:
        print(f"   Forces on contact atoms [kcal/(mol·Å)]:")
        _print_forces_block(first_ev)
    print()

    # ── 2. Frame with smallest distance ever ─────────────────────────────────
    min_dist = min(all_dists)
    min_ev, min_pair = min(all_pairs, key=lambda x: x[1][4])
    print("2) SMALLEST DISTANCE")
    print(f"   Timestep {min_ev.timestep}  (frame {min_ev.frame_idx})  dist = {min_dist:.4f} Å")
    print(f"   Pair: atoms {min_pair[0]} – {min_pair[1]}"
          f"  (types {min_pair[2]}, {min_pair[3]})")
    if has_forces:
        print(f"   Forces on contact atoms [kcal/(mol·Å)]:")
        _print_forces_block(min_ev)
    print()

    # ── 3. Frame with highest force magnitude on any contact atom ────────────
    if has_forces:
        best_ev, best_fnorm, best_aid = None, -1.0, -1
        for ev in events:
            if ev.forces_on_contact is None:
                continue
            norms = np.linalg.norm(ev.forces_on_contact, axis=1)
            idx = int(np.argmax(norms))
            if norms[idx] > best_fnorm:
                best_fnorm = float(norms[idx])
                best_ev = ev
                best_aid = ev.contact_atom_ids[idx]
        if best_ev is not None:
            print("3) HIGHEST FORCE ON A CONTACT ATOM")
            print(f"   Timestep {best_ev.timestep}  (frame {best_ev.frame_idx})")
            print(f"   Atom {best_aid}  |f| = {best_fnorm:.4f} kcal/(mol·Å)")
            print(f"   Forces on all contact atoms [kcal/(mol·Å)]:")
            _print_forces_block(best_ev)
            print()
    else:
        print("3) HIGHEST FORCE  — not available (no forces in dump)")
        print()

    # ── Overall statistics ────────────────────────────────────────────────────
    print("-" * 70)
    print("OVERALL STATISTICS")
    print(f"  Closest contact:       {min_dist:.4f} Å")
    print(f"  Mean close-contact:    {np.mean(all_dists):.4f} Å")

    pair_counter: Counter = Counter()
    for e in events:
        for id_i, id_j, _, _, _ in e.pairs:
            pair_counter[(id_i, id_j)] += 1
    print()
    print("  Most frequent contact pairs:")
    print(f"    {'id_i':>6} {'id_j':>6} {'frames':>8}")
    for (i, j), cnt in pair_counter.most_common(10):
        print(f"    {i:>6} {j:>6} {cnt:>8}")

    if dissoc_events:
        print()
        print("  Dissociation events (timestep, Rg_start → Rg_max):")
        for e in dissoc_events:
            rg_max = max(e.rg_context) if e.rg_context else e.rg_at_contact
            print(f"    t={e.timestep:>10d}  Rg {e.rg_at_contact:.2f} → {rg_max:.2f} Å"
                  f"  (+{e.rg_increase:.2f} Å)")


def _write_csv(events: List[ContactEvent], path: Path) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestep", "frame_idx",
                         "n_contacts", "min_dist_A",
                         "rg_at_contact_A", "rg_max_after_A", "rg_increase_A",
                         "contact_pairs"])
        for e in events:
            dists = [d for _, _, _, _, d in e.pairs]
            rg_max = max(e.rg_context) if e.rg_context else e.rg_at_contact
            pair_str = " | ".join(f"{i}-{j}:{d:.3f}"
                                  for i, j, _, _, d in e.pairs)
            writer.writerow([
                e.timestep, e.frame_idx,
                len(e.pairs), f"{min(dists):.4f}",
                f"{e.rg_at_contact:.4f}",
                f"{rg_max:.4f}",
                f"{(e.rg_increase or 0):.4f}",
                pair_str,
            ])
    print(f"CSV written: {path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Find close bead contacts in a LAMMPS dump and assess dissociation.")
    parser.add_argument("dump", help="Path to LAMMPS dump file")
    parser.add_argument("--cutoff", type=float, default=3.0,
                        help="Distance cutoff in Å (default: 3.0)")
    parser.add_argument("--context", type=int, default=10,
                        help="Frames after contact to track Rg (default: 10)")
    parser.add_argument("--rg-threshold", type=float, default=5.0,
                        help="Rg increase in Å to flag as dissociation (default: 5.0)")
    parser.add_argument("--sample-frames", type=int, default=10,
                        help="Number of evenly-spaced frames to report mean bead distance (default: 10)")
    parser.add_argument("--csv", default=None,
                        help="Write summary CSV to this path")
    parser.add_argument("--verbose", action="store_true",
                        help="Print details for every contact frame (off by default)")
    args = parser.parse_args()

    dump_path = Path(args.dump)
    if not dump_path.exists():
        sys.exit(f"Error: dump file not found: {dump_path}")

    print(f"Cutoff:         {args.cutoff} Å")
    print(f"Context frames: {args.context}")
    print(f"Rg threshold:   {args.rg_threshold} Å  (flags possible dissociation)")
    print(f"Sample frames:  {args.sample_frames}")
    print()

    events, distance_samples = analyse(
        dump_path, args.cutoff, args.context, args.rg_threshold,
        quiet=not args.verbose, n_sample=args.sample_frames)

    _print_distance_samples(distance_samples)
    _print_summary(events, args.rg_threshold)

    if args.csv:
        _write_csv(events, Path(args.csv))


if __name__ == "__main__":
    main()
