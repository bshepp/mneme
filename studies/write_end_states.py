"""Write (or check) each GRN study's end_states.csv from the packed run traces.

usage: python studies/write_end_states.py [--data data/raw/betse_grn_2026-11] [--check]

Each column of an end_states.csv is the final frame of one run phase; the
mapping from column to run is COLUMNS below. Without --check the three files
are rewritten. With --check they are compared with the files in the
repository and the script exits 1 on any difference above 1e-4 mV, which is
the precision of the packed traces.

The script also prints the summary of the Dgj = 2e-14 continuation runs
that studies/sweep/RESULTS.md quotes (start-of-phase transient, movement
over the continuation and over its final 3,000 s, the distance between the
two end states, and their depolarised-cell counts and levels), since
studies/sweep/analyze.py reads only the init phases.
"""
from __future__ import annotations

import argparse
import csv
import gzip
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TOLERANCE_MV = 1e-4

COLUMNS = {
    "basins": {
        "A": "settle_A/sim", "B": "settle_B/sim", "C": "cont_pert_A_toward_f50/sim",
        "D": "basin_L100_r1/init", "E": "basin_L100_r2/init", "F": "basin_L100_r4/init",
    },
    "perturbation": {
        "C": "cont_pert_A_toward_f50/sim",
        "D2": "cont_pert_D_erase_f100/sim",
        "E2": "cont_pert_E_erase_f100/sim",
        "dgj2e-14_A": "cont_sweep_dgj2em14_A/sim",
        "dgj2e-14_B": "cont_sweep_dgj2em14_B/sim",
    },
    "sweep": {
        "dgj5e-15_A": "sweep_dgj5em15_A/init", "dgj5e-15_B": "sweep_dgj5em15_B/init",
        "dgj2e-14_A": "sweep_dgj2em14_A/init", "dgj2e-14_B": "sweep_dgj2em14_B/init",
        "dgj1.5e-13_A": "sweep_dgj1p5em13_A/init", "dgj1.5e-13_B": "sweep_dgj1p5em13_B/init",
        "gj1e-9_A": "sweep_gj1em09_A/init", "gj1e-9_B": "sweep_gj1em09_B/init",
    },
}


def load_trace(data: Path, run_phase: str) -> np.ndarray:
    """All frames of one packed run phase, as (frames, cells) in mV."""
    run, phase = run_phase.split("/")
    path = data / f"{run}__{phase}.csv.gz"
    with gzip.open(path, "rt") as f:
        v = np.loadtxt(f, delimiter=",", skiprows=1)
    v = np.atleast_2d(v)[:, 2:]            # drop frame index and time
    if v.size == 0 or not np.all(np.isfinite(v)):
        raise ValueError(f"{path}: empty or non-finite trace")
    return v


def load_cells(data: Path) -> np.ndarray:
    xy = np.loadtxt(data / "cells.csv", delimiter=",", skiprows=1)[:, 1:]
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError("cells.csv must have columns cell, x_um, y_um")
    return xy


def depolarised(v: np.ndarray) -> np.ndarray:
    return v > 0.5 * (v.min() + v.max())


def table(data: Path, columns: dict[str, str], xy: np.ndarray) -> list[list[str]]:
    ends = {}
    for col, run_phase in columns.items():
        v = load_trace(data, run_phase)
        if v.shape[1] != len(xy):
            raise ValueError(f"{run_phase}: {v.shape[1]} cells, cells.csv has {len(xy)}")
        ends[col] = v[-1]
    rows = [["x_um", "y_um", *columns]]
    for i, (x, y) in enumerate(xy):
        rows.append([f"{x:.3f}", f"{y:.3f}", *(f"{ends[c][i]:.4f}" for c in columns)])
    return rows


def check(path: Path, rows: list[list[str]]) -> float:
    with open(path, newline="") as f:
        have = list(csv.reader(f))
    if have[0] != rows[0]:
        raise SystemExit(f"{path}: columns {have[0]} differ from {rows[0]}")
    if len(have) != len(rows):
        raise SystemExit(f"{path}: {len(have) - 1} rows, expected {len(rows) - 1}")
    a = np.array([[float(x) for x in r] for r in have[1:]])
    b = np.array([[float(x) for x in r] for r in rows[1:]])
    return float(np.max(np.abs(a - b)))


def continuation_summary(data: Path) -> None:
    print("\nDgj = 2e-14 continuations (studies/sweep/RESULTS.md):")
    ends = {}
    for start in ("A", "B"):
        init = load_trace(data, f"sweep_dgj2em14_{start}/init")
        sim = load_trace(data, f"cont_sweep_dgj2em14_{start}/sim")
        n_final = 100                       # 100 frames at 30 s = the final 3,000 s
        ends[start] = sim[-1]
        m = depolarised(sim[-1])
        print(f"  {start} start: transient at the start of the continuation "
              f"{np.max(np.abs(init[-1] - sim[0])):.2f} mV; moved "
              f"{np.max(np.abs(init[-1] - sim[-1])):.2f} mV over the continuation, "
              f"{np.max(np.abs(sim[-n_final] - sim[-1])):.4f} mV over its final 3,000 s; "
              f"{int(m.sum())} of {len(m)} cells depolarised, "
              f"{sim[-1][m].mean():.1f} / {sim[-1][~m].mean():.1f} mV")
    print(f"  A start vs B start at the end: {np.max(np.abs(ends['A'] - ends['B'])):.1f} mV")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data", type=Path, default=HERE.parent / "data" / "raw" / "betse_grn_2026-11")
    ap.add_argument("--check", action="store_true", help="compare with the repository files")
    args = ap.parse_args(argv)
    xy = load_cells(args.data)

    worst = 0.0
    for study, columns in COLUMNS.items():
        path = HERE / study / "end_states.csv"
        rows = table(args.data, columns, xy)
        if args.check:
            d = check(path, rows)
            worst = max(worst, d)
            print(f"{path.relative_to(HERE.parent)}: largest difference {d:.6f} mV")
        else:
            with open(path, "w", newline="") as f:
                csv.writer(f).writerows(rows)
            print(f"wrote {path.relative_to(HERE.parent)}: {len(columns)} columns, {len(xy)} cells")

    continuation_summary(args.data)

    if args.check and worst > TOLERANCE_MV:
        print(f"\nFAIL: a difference of {worst:.6f} mV exceeds {TOLERANCE_MV} mV")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
