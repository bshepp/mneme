"""Can persistent homology tell the two gene-network patterns apart better
than the mean voltage can? (project_plan.md, open question 4)

usage: python studies/topology/compare_statistics.py <run_root> [--out results.json]
       python studies/topology/compare_statistics.py --end-states <csv> [--out results.json]

<run_root> is the perturbation study's run directory: the reference states
are the last frames of settle_A's and settle_B's control continuation. With
--end-states, the references are every voltage column of that CSV (x_um,
y_um, then one column per pattern), e.g. studies/basins/end_states.csv for
the six patterns A to F; chance accuracy is then 1 / number of patterns.

Method. Each reference pattern is a voltage per cell. Gaussian measurement
noise of standard deviation sigma is added to every cell, independently, and
the noisy sample is assigned to the nearer reference under one statistic:

  mean      |mean(sample) - mean(reference)|, over cells
  topology  2-Wasserstein distance between the sample's and the reference's
            persistence diagrams (H0 and H1 of the superlevel filtration,
            summed), on the field interpolated to a 64 x 64 grid. Reported
            raw (every feature) and with features of persistence below
            2 mV and below 5 mV dropped, since noise creates many short
            bars that the raw distance counts. 5 mV is well below the
            patterns' depolarised-to-polarised step of about 25 mV.
  cells     root-mean-square per-cell difference (uses the cell identities;
            a ceiling that neither summary statistic can beat)

Accuracy is the fraction of noisy samples assigned to the pattern they came
from. Chance is 0.5. The separation ratio of a statistic is the A-to-B
distance divided by the median distance of a noisy sample from its own
reference; values above 1 mean the patterns are farther apart than the noise
moves a sample.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from mneme.core.topology import PersistentHomology, compute_wasserstein_distance
from mneme.data.betse_loader import interpolate_to_grid, load_betse_cells

SIGMAS_MV = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
N_DRAWS = 40
RESOLUTION = (64, 64)
SEED = 20261105
THRESHOLDS_MV = [0.0, 2.0, 5.0]   # persistence below this is dropped


def topo_label(thr):
    return "topology" if thr == 0 else f"topology{int(thr)}"


def diagrams(ph, x, y, v, resolution=RESOLUTION):
    grid, _, _ = interpolate_to_grid(x, y, v, resolution=resolution, method="linear")
    return ph.compute_persistence(grid)


def topo_distance(d1, d2):
    return sum(compute_wasserstein_distance(a, b, p=2.0) for a, b in zip(d1, d2))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("run_root", type=Path, nargs="?", default=None)
    ap.add_argument("--end-states", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--draws", type=int, default=N_DRAWS)
    ap.add_argument("--resolution", type=int, default=RESOLUTION[0],
                    help="grid side for the interpolated field (default %(default)s, the pre-stated design)")
    args = ap.parse_args(argv)

    refs, coords = {}, None
    if args.end_states is not None:
        table = np.genfromtxt(args.end_states, delimiter=",", names=True)
        coords = (table["x_um"], table["y_um"])
        refs = {name: np.asarray(table[name], float) for name in table.dtype.names[2:]}
    elif args.run_root is not None:
        for label in ("A", "B"):
            v, x, y, _ = load_betse_cells(args.run_root / "RESULTS" / f"settle_{label}" / "sim" / "Vmem2D_TextExport")
            refs[label] = v[-1]
            coords = (x, y)
    else:
        ap.error("give a run_root or --end-states")
    x, y = coords
    labels = list(refs)
    phs = {thr: PersistentHomology(max_dimension=1, filtration="superlevel",
                                   persistence_threshold=thr) for thr in THRESHOLDS_MV}
    resolution = (args.resolution, args.resolution)
    ref_pd = {thr: {k: diagrams(ph, x, y, v, resolution) for k, v in refs.items()}
              for thr, ph in phs.items()}

    # Distances between every pair of references, under each statistic; the
    # separation ratio uses the smallest pair distance (the hardest pair).
    def pair_dist(a, b):
        d = dict(mean=abs(refs[a].mean() - refs[b].mean()),
                 cells=float(np.sqrt(np.mean((refs[a] - refs[b]) ** 2))),
                 max_cell=float(np.max(np.abs(refs[a] - refs[b]))))
        for thr in THRESHOLDS_MV:
            d[topo_label(thr)] = topo_distance(ref_pd[thr][a], ref_pd[thr][b])
        return d

    pairs = {f"{a} vs {b}": pair_dist(a, b) for i, a in enumerate(labels) for b in labels[i + 1:]}
    ab = {k: min(p[k] for p in pairs.values()) for k in next(iter(pairs.values()))}
    for k, ref in refs.items():
        print(f"{k}: mean {ref.mean():.3f} mV; features (H0, H1) at thresholds "
              + ", ".join(f"{thr:g} mV: ({len(ref_pd[thr][k][0].points)}, {len(ref_pd[thr][k][1].points)})"
                          for thr in THRESHOLDS_MV))
    for name, d in pairs.items():
        print(f"{name}: " + "; ".join(f"{k} {v:.3f} mV" for k, v in d.items()))
    if len(pairs) > 1:
        print("smallest pair distance: " + "; ".join(f"{k} {v:.3f} mV" for k, v in ab.items()))
    print()

    rng = np.random.default_rng(SEED)
    stats = ("mean",) + tuple(topo_label(t) for t in THRESHOLDS_MV) + ("cells",)
    print(f"{'sigma[mV]':>9s} " + " ".join(f"{s + ' acc':>13s} {s + ' sep':>13s}" for s in stats))
    print(" " * 10 + f"(acc: fraction assigned to the right pattern, chance {1 / len(labels):.3f}; "
          "sep: smallest pair distance / median own distance)")
    table = {}
    for sigma in SIGMAS_MV:
        correct = {s: 0 for s in stats}
        own = {s: [] for s in stats}
        for truth in labels:
            for _ in range(args.draws):
                sample = refs[truth] + sigma * rng.standard_normal(len(refs[truth]))
                d = {}
                d["mean"] = {k: abs(sample.mean() - refs[k].mean()) for k in refs}
                for thr, ph in phs.items():
                    pd = diagrams(ph, x, y, sample, resolution)
                    d[topo_label(thr)] = {k: topo_distance(pd, ref_pd[thr][k]) for k in refs}
                d["cells"] = {k: float(np.sqrt(np.mean((sample - refs[k]) ** 2))) for k in refs}
                for s in stats:
                    own[s].append(d[s][truth])
                    best = min(d[s].values())
                    winners = [k for k, v in d[s].items() if v == best]
                    if truth in winners:
                        correct[s] += 1 / len(winners)   # a tie splits the credit
        n = len(labels) * args.draws
        row = {}
        for s in stats:
            med = float(np.median(own[s]))
            row[s] = dict(accuracy=correct[s] / n, median_own_distance=med,
                          separation_ratio=(ab[s] / med if med > 0 else float("inf")))
        table[sigma] = row
        print(f"{sigma:9.1f} " + " ".join(
            f"{row[s]['accuracy']:13.3f} {row[s]['separation_ratio']:13.2f}" for s in stats))

    if args.out is not None:
        payload = dict(
            method=dict(sigmas_mV=SIGMAS_MV, draws_per_state=args.draws, resolution=list(resolution),
                        filtration="superlevel", dimensions=[0, 1], wasserstein_p=2, seed=SEED,
                        persistence_thresholds_mV=THRESHOLDS_MV),
            reference={k: dict(mean_mV=float(refs[k].mean()),
                               H0=ref_pd[0.0][k][0].points.tolist(),
                               H1=ref_pd[0.0][k][1].points.tolist()) for k in labels},
            pairs=pairs, smallest_pair=ab, chance=1 / len(labels),
            by_sigma={str(k): v for k, v in table.items()},
        )
        args.out.write_text(json.dumps(payload, indent=1, default=float), encoding="utf-8")
        print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
