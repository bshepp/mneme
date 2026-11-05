"""Can persistent homology tell the two gene-network patterns apart better
than the mean voltage can? (project_plan.md, open question 4)

usage: python studies/topology/compare_statistics.py <run_root> [--out results.json]

<run_root> is the perturbation study's run directory: the reference states
are the last frames of settle_A's and settle_B's control continuation.

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


def diagrams(ph, x, y, v):
    grid, _, _ = interpolate_to_grid(x, y, v, resolution=RESOLUTION, method="linear")
    return ph.compute_persistence(grid)


def topo_distance(d1, d2):
    return sum(compute_wasserstein_distance(a, b, p=2.0) for a, b in zip(d1, d2))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("run_root", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--draws", type=int, default=N_DRAWS)
    args = ap.parse_args(argv)

    refs, coords = {}, None
    for label in ("A", "B"):
        v, x, y, _ = load_betse_cells(args.run_root / "RESULTS" / f"settle_{label}" / "sim" / "Vmem2D_TextExport")
        refs[label] = v[-1]
        coords = (x, y)
    x, y = coords
    phs = {thr: PersistentHomology(max_dimension=1, filtration="superlevel",
                                   persistence_threshold=thr) for thr in THRESHOLDS_MV}
    ref_pd = {thr: {k: diagrams(ph, x, y, v) for k, v in refs.items()} for thr, ph in phs.items()}

    ab = dict(
        mean=abs(refs["A"].mean() - refs["B"].mean()),
        cells=float(np.sqrt(np.mean((refs["A"] - refs["B"]) ** 2))),
        max_cell=float(np.max(np.abs(refs["A"] - refs["B"]))),
    )
    for thr in THRESHOLDS_MV:
        ab[topo_label(thr)] = topo_distance(ref_pd[thr]["A"], ref_pd[thr]["B"])
    for k in refs:
        print(f"{k}: mean {refs[k].mean():.3f} mV; features (H0, H1) at thresholds "
              + ", ".join(f"{thr:g} mV: ({len(ref_pd[thr][k][0].points)}, {len(ref_pd[thr][k][1].points)})"
                          for thr in THRESHOLDS_MV))
    print("A vs B: " + "; ".join(f"{k} {v:.3f} mV" for k, v in ab.items()) + "\n")

    rng = np.random.default_rng(SEED)
    stats = ("mean",) + tuple(topo_label(t) for t in THRESHOLDS_MV) + ("cells",)
    print(f"{'sigma[mV]':>9s} " + " ".join(f"{s + ' acc':>13s} {s + ' sep':>13s}" for s in stats))
    print(" " * 10 + "(acc: fraction assigned to the right pattern; sep: A-B distance / median own distance)")
    table = {}
    for sigma in SIGMAS_MV:
        correct = {s: 0 for s in stats}
        own = {s: [] for s in stats}
        for truth in ("A", "B"):
            other = "B" if truth == "A" else "A"
            for _ in range(args.draws):
                sample = refs[truth] + sigma * rng.standard_normal(len(refs[truth]))
                d = {}
                d["mean"] = {k: abs(sample.mean() - refs[k].mean()) for k in refs}
                for thr, ph in phs.items():
                    pd = diagrams(ph, x, y, sample)
                    d[topo_label(thr)] = {k: topo_distance(pd, ref_pd[thr][k]) for k in refs}
                d["cells"] = {k: float(np.sqrt(np.mean((sample - refs[k]) ** 2))) for k in refs}
                for s in stats:
                    own[s].append(d[s][truth])
                    if d[s][truth] < d[s][other]:
                        correct[s] += 1
                    elif d[s][truth] == d[s][other]:
                        correct[s] += 0.5   # a tie: no decision either way
        n = 2 * args.draws
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
            method=dict(sigmas_mV=SIGMAS_MV, draws_per_state=args.draws, resolution=RESOLUTION,
                        filtration="superlevel", dimensions=[0, 1], wasserstein_p=2, seed=SEED,
                        persistence_thresholds_mV=THRESHOLDS_MV),
            reference=dict(
                A_mean_mV=float(refs["A"].mean()), B_mean_mV=float(refs["B"].mean()),
                A_H0=ref_pd[0.0]["A"][0].points.tolist(), A_H1=ref_pd[0.0]["A"][1].points.tolist(),
                B_H0=ref_pd[0.0]["B"][0].points.tolist(), B_H1=ref_pd[0.0]["B"][1].points.tolist(),
            ),
            A_vs_B=ab, by_sigma={str(k): v for k, v in table.items()},
        )
        args.out.write_text(json.dumps(payload, indent=1, default=float), encoding="utf-8")
        print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
