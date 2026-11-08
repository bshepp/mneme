"""Pack BETSE run output into one gzipped CSV per run phase for the repository.

usage: python studies/pack_runs.py <run_root> <out_dir> [--exclude prefix ...]

For every RESULTS/<run>/<phase>/Vmem2D_TextExport/ under <run_root>, writes
<out_dir>/<run>__<phase>.csv.gz with one row per sampled frame: frame index,
time in seconds (frame + 1 times the sampling interval), then the membrane
voltage of every cell in mV to 4 decimals. Cell positions go once to
<out_dir>/cells.csv, and a manifest.csv lists every file with its frame
count, duration and final mean voltage. Phases whose cells differ from the
first file's are refused: all the GRN studies share one seeded world.

Read a file back with pandas.read_csv(path) or
numpy.loadtxt(path, delimiter=",", skiprows=1).
"""
from __future__ import annotations

import argparse
import csv
import gzip
from pathlib import Path

import numpy as np

from mneme.data.betse_loader import load_betse_cells

SAMPLING_S = {"init": 30.0, "sim": 30.0}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("run_root", type=Path)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--exclude", nargs="*", default=[], help="run-name prefixes to skip")
    ap.add_argument("--sampling", type=float, default=30.0)
    args = ap.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    ref_xy = None
    rows = []
    for run_dir in sorted((args.run_root / "RESULTS").iterdir()):
        if any(run_dir.name.startswith(p) for p in args.exclude):
            continue
        for phase in ("init", "sim"):
            src = run_dir / phase / "Vmem2D_TextExport"
            if not any(src.glob("Vmem2D_*.csv")):
                continue
            v, x, y, frames = load_betse_cells(src)
            xy = np.column_stack([x, y])
            if ref_xy is None:
                ref_xy = xy
                with open(args.out_dir / "cells.csv", "w", newline="") as f:
                    w = csv.writer(f)
                    w.writerow(["cell", "x_um", "y_um"])
                    for i, (cx, cy) in enumerate(xy):
                        w.writerow([i, f"{cx:.3f}", f"{cy:.3f}"])
            elif not np.allclose(xy, ref_xy):
                raise SystemExit(f"{run_dir.name}/{phase}: cells differ from the first file")
            t = (np.asarray(frames, float) + 1.0) * args.sampling
            name = f"{run_dir.name}__{phase}.csv.gz"
            with gzip.open(args.out_dir / name, "wt", newline="") as f:
                w = csv.writer(f)
                w.writerow(["frame", "time_s"] + [f"cell_{i}" for i in range(v.shape[1])])
                for k, (fr, tt) in enumerate(zip(frames, t)):
                    w.writerow([fr, f"{tt:.1f}"] + [f"{val:.4f}" for val in v[k]])
            rows.append([name, run_dir.name, phase, len(frames), f"{t[-1]:.0f}",
                         f"{v[-1].mean():.4f}", f"{np.ptp(v[-1]):.4f}"])
            print(f"{name}: {len(frames)} frames")
    with open(args.out_dir / "manifest.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "run", "phase", "n_frames", "t_end_s", "end_mean_mV", "end_spread_mV"])
        w.writerows(rows)
    print(f"{len(rows)} files, {len(ref_xy)} cells")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
