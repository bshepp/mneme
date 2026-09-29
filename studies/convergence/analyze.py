"""Analyse the convergence experiment: do runs from different starting
conditions settle to the same state?

usage: python studies/convergence/analyze.py <run_root> [--out results.json]

<run_root> holds RESULTS/<run>/{init,sim}/Vmem2D_TextExport/ as written by
run_one.sh. Runs are grouped by the coupling label in their name
(``full_<coupling>_<initial condition>``).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from mneme.analysis.steady_state import (
    assess_steady_state,
    count_distinct_states,
    pairwise_max_difference,
)
from mneme.data.betse_loader import load_betse_cells

# Settling criteria, fixed before the runs were analysed.
RATE_TOLERANCE = 1.0e-4    # mV per second, sustained over the final samples
DRIFT_TOLERANCE = 0.1      # mV still to travel, estimated
SUSTAIN = 20               # final samples over which the rate is checked
# Two end states are the same state when every cell agrees to within this.
# It must sit above the estimated remaining drift of the runs compared, or
# unfinished relaxation would be counted as a difference between states.
SAME_STATE_THRESHOLD = 1.0  # mV


def load_run(run_dir: Path, init_dt: float, sim_dt: float):
    """Load init and sim phases of one run as a single trajectory."""
    init_v, x, y, init_frames = load_betse_cells(run_dir / "init" / "Vmem2D_TextExport")
    sim_v, xs, ys, sim_frames = load_betse_cells(run_dir / "sim" / "Vmem2D_TextExport")
    if init_v.shape[1] != sim_v.shape[1] or not (np.allclose(x, xs) and np.allclose(y, ys)):
        raise ValueError(f"{run_dir.name}: init and sim phases have different cells")
    init_t = np.asarray(init_frames, float) * init_dt
    sim_t = init_t[-1] + init_dt + np.asarray(sim_frames, float) * sim_dt
    return np.vstack([init_v, sim_v]), np.concatenate([init_t, sim_t]), x, y


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--init-sampling", type=float, default=60.0)
    parser.add_argument("--sim-sampling", type=float, default=60.0)
    args = parser.parse_args(argv)

    results_dir = args.run_root / "RESULTS"
    run_dirs = sorted(
        d for d in results_dir.iterdir()
        if (d / "sim" / "Vmem2D_TextExport").is_dir()
        and any((d / "sim" / "Vmem2D_TextExport").glob("Vmem2D_*.csv"))
    )
    if not run_dirs:
        print(f"No completed runs under {results_dir}", file=sys.stderr)
        return 1

    runs = {}
    for d in run_dirs:
        values, times, x, y = load_run(d, args.init_sampling, args.sim_sampling)
        report = assess_steady_state(
            values, times,
            rate_tolerance=RATE_TOLERANCE,
            drift_tolerance=DRIFT_TOLERANCE,
            sustain=SUSTAIN,
        )
        coupling = d.name.split("_ic_")[0]
        runs[d.name] = dict(
            coupling=coupling, values=values, times=times, x=x, y=y, report=report,
        )

    # --- every run must be over the same cells --------------------------
    names = list(runs)
    ref = runs[names[0]]
    geometry_gap = 0.0
    for name in names[1:]:
        r = runs[name]
        if r["x"].shape != ref["x"].shape:
            raise ValueError(f"{name} has {len(r['x'])} cells; {names[0]} has {len(ref['x'])}")
        geometry_gap = max(
            geometry_gap,
            float(np.max(np.abs(r["x"] - ref["x"]))),
            float(np.max(np.abs(r["y"] - ref["y"]))),
        )

    print(f"Runs: {len(runs)}   cells: {len(ref['x'])}   "
          f"largest coordinate difference between runs: {geometry_gap:.3g} um")
    print(f"Settling criteria: rate < {RATE_TOLERANCE} mV/s over final {SUSTAIN} samples, "
          f"estimated remaining drift < {DRIFT_TOLERANCE} mV")
    print()
    header = (f"{'run':32s} {'frames':>6s} {'t_end[s]':>9s} {'start mean':>10s} "
              f"{'end mean':>9s} {'end spread':>10s} {'rate[mV/s]':>11s} {'tau[s]':>8s} "
              f"{'to go[mV]':>10s} {'settled':>8s}")
    print(header)
    summary = {}
    for name, r in runs.items():
        v, rep = r["values"], r["report"]
        print(f"{name:32s} {len(v):6d} {r['times'][-1]:9.0f} {v[0].mean():10.3f} "
              f"{v[-1].mean():9.3f} {np.ptp(v[-1]):10.3f} {rep.final_rate:11.2e} "
              f"{rep.time_constant:8.0f} {rep.remaining_drift:10.4f} {str(rep.settled):>8s}")
        summary[name] = dict(
            coupling=r["coupling"], n_frames=int(len(v)), t_end=float(r["times"][-1]),
            start_mean=float(v[0].mean()), end_mean=float(v[-1].mean()),
            end_min=float(v[-1].min()), end_max=float(v[-1].max()),
            final_rate=rep.final_rate, time_constant=rep.time_constant,
            remaining_drift=rep.remaining_drift, settled=rep.settled,
        )

    # --- replicate noise -------------------------------------------------
    replicate_noise = None
    for name in names:
        if name.endswith("_rep") and name[: -len("_rep")] in runs:
            twin = name[: -len("_rep")]
            replicate_noise = float(
                np.max(np.abs(runs[name]["values"][-1] - runs[twin]["values"][-1]))
            )
            print(f"\nReplicate noise ({twin} vs its exact repeat): "
                  f"{replicate_noise:.3g} mV largest per-cell difference")

    # --- distinct end states, per coupling -------------------------------
    print(f"\nSame-state threshold: {SAME_STATE_THRESHOLD} mV")
    groups = {}
    for coupling in sorted({r["coupling"] for r in runs.values()}):
        members = [n for n in names if runs[n]["coupling"] == coupling and not n.endswith("_rep")]
        ends = [runs[n]["values"][-1] for n in members]
        starts = [runs[n]["values"][0] for n in members]
        found = count_distinct_states(ends, SAME_STATE_THRESHOLD)
        start_gap = pairwise_max_difference(starts)
        worst_drift = max(runs[n]["report"].remaining_drift for n in members)
        all_settled = all(runs[n]["report"].settled for n in members)
        print(f"\n[{coupling}] {len(members)} starting conditions")
        print(f"  largest difference between starting states: {start_gap.max():8.3f} mV")
        print(f"  largest difference between end states:      {found.distances.max():8.3f} mV")
        print(f"  distinct end states:                        {found.n_states}")
        print(f"  largest estimated remaining drift:          {worst_drift:8.3f} mV")
        print(f"  all runs settled:                           {all_settled}")
        print("  end-state distance matrix [mV]:")
        for n, row in zip(members, found.distances):
            print("   ", f"{n.split('_ic_')[1]:10s}", " ".join(f"{d:7.3f}" for d in row))
        groups[coupling] = dict(
            members=members, n_states=found.n_states, labels=found.labels.tolist(),
            largest_start_difference=float(start_gap.max()),
            largest_end_difference=float(found.distances.max()),
            largest_remaining_drift=float(worst_drift), all_settled=all_settled,
            distances=found.distances.tolist(),
        )

    if args.out is not None:
        payload = dict(
            criteria=dict(rate_tolerance=RATE_TOLERANCE, drift_tolerance=DRIFT_TOLERANCE,
                          sustain=SUSTAIN, same_state_threshold=SAME_STATE_THRESHOLD),
            n_cells=int(len(ref["x"])), geometry_gap_um=geometry_gap,
            replicate_noise_mV=replicate_noise, runs=summary, groups=groups,
        )
        args.out.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
        print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
