"""Analyse the perturbation study: does a settled pattern return after a
stated displacement, flip to the other pattern, or end somewhere else?

usage: python studies/perturbation/analyze.py <run_root> [--out results.json]

<run_root> holds RESULTS/<run>/{init,sim}/Vmem2D_TextExport/ as written by
run_one.sh, and perturbations.json from make_configs.py perturb.

Reference states: the last frame of each settling run's unperturbed sim
phase (settle_A, settle_B). Each perturbed run pert_<P>_<label> starts from
the state saved at the end of settle_<P>'s init phase, displaced as recorded
in perturbations.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from mneme.analysis.steady_state import assess_steady_state
from mneme.data.betse_loader import load_betse_cells

# Settling criteria and same-state threshold, as in studies/convergence.
RATE_TOLERANCE = 1.0e-4    # mV per second, sustained over the final samples
DRIFT_TOLERANCE = 0.1      # mV still to travel, estimated
SUSTAIN = 20               # final samples over which the rate is checked
SAME_STATE_THRESHOLD = 1.0  # mV, largest per-cell difference


def load_phase(run_dir: Path, phase: str, dt: float):
    v, x, y, frames = load_betse_cells(run_dir / phase / "Vmem2D_TextExport")
    # BETSE's first sample is one sampling interval after the phase starts.
    t = (np.asarray(frames, float) + 1.0) * dt
    return v, t, x, y


def max_diff(a, b):
    return float(np.max(np.abs(a - b)))


def first_sustained_below(d, t, threshold):
    """Earliest time after which d stays below threshold, or None."""
    below = d < threshold
    if not below[-1]:
        return None
    k = len(below)
    while k > 0 and below[k - 1]:
        k -= 1
    return float(t[k])


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--sampling", type=float, default=30.0)
    args = parser.parse_args(argv)
    root = args.run_root
    results = root / "RESULTS"
    dt = args.sampling

    meta = json.loads((root / "perturbations.json").read_text(encoding="utf-8")) \
        if (root / "perturbations.json").exists() else {}

    # --- reference states ------------------------------------------------
    ref, settle = {}, {}
    for label in ("A", "B"):
        d = results / f"settle_{label}"
        if not (d / "sim" / "Vmem2D_TextExport").is_dir():
            continue
        vi, ti, x, y = load_phase(d, "init", dt)
        vs, ts, xs, ys = load_phase(d, "sim", dt)
        assert np.allclose(x, xs) and np.allclose(y, ys)
        init_rep = assess_steady_state(vi, ti, rate_tolerance=RATE_TOLERANCE,
                                       drift_tolerance=DRIFT_TOLERANCE, sustain=SUSTAIN)
        sim_rep = assess_steady_state(vs, ts, rate_tolerance=RATE_TOLERANCE,
                                      drift_tolerance=DRIFT_TOLERANCE, sustain=SUSTAIN)
        ref[label] = vs[-1]
        settle[label] = dict(
            init_frames=int(len(vi)), init_settled=init_rep.settled,
            init_remaining_drift=init_rep.remaining_drift, init_time_constant=init_rep.time_constant,
            sim_settled=sim_rep.settled, sim_remaining_drift=sim_rep.remaining_drift,
            init_end_to_sim_start_mV=max_diff(vi[-1], vs[0]),
            init_end_to_sim_end_mV=max_diff(vi[-1], vs[-1]),
            end_mean_mV=float(vs[-1].mean()), end_spread_mV=float(np.ptp(vs[-1])),
            n_cells=int(vs.shape[1]),
        )
        print(f"settle_{label}: init {len(vi)} frames, settled={init_rep.settled} "
              f"(drift {init_rep.remaining_drift:.4f} mV, tau {init_rep.time_constant:.0f} s); "
              f"sim continuation moved {settle[label]['init_end_to_sim_end_mV']:.4f} mV, "
              f"settled={sim_rep.settled}; end mean {vs[-1].mean():.2f} mV, spread {np.ptp(vs[-1]):.2f} mV")
    if len(ref) < 2:
        print("Need both settle_A and settle_B", file=sys.stderr)
        return 1
    ab = max_diff(ref["A"], ref["B"])
    print(f"A vs B: {ab:.2f} mV largest per-cell difference; "
          f"same-state threshold {SAME_STATE_THRESHOLD} mV\n")

    # --- perturbed runs --------------------------------------------------
    header = (f"{'run':30s} {'anion max':>9s} {'anion mean':>10s} {'disp[mV]':>8s} {'d_own end':>9s} "
              f"{'d_other end':>11s} {'settled':>7s} {'outcome':>8s} {'t_return[s]':>11s}")
    print(header)
    runs = {}
    run_dirs = sorted(d for d in results.iterdir() if d.name.startswith("pert_")
                      and any((d / "sim" / "Vmem2D_TextExport").glob("Vmem2D_*.csv")))
    for d in run_dirs:
        own = d.name.split("_")[1]
        other = "B" if own == "A" else "A"
        v, t, _, _ = load_phase(d, "sim", dt)
        d_own = np.array([max_diff(f, ref[own]) for f in v])
        d_other = np.array([max_diff(f, ref[other]) for f in v])
        rep = assess_steady_state(v, t, rate_tolerance=RATE_TOLERANCE,
                                  drift_tolerance=DRIFT_TOLERANCE, sustain=SUSTAIN)
        if d_own[-1] < SAME_STATE_THRESHOLD:
            outcome = "returned"
        elif d_other[-1] < SAME_STATE_THRESHOLD:
            outcome = "flipped"
        else:
            outcome = "other"
        t_return = first_sustained_below(d_own, t, SAME_STATE_THRESHOLD)
        m = meta.get(own, {}).get("runs", {}).get(d.name, {})
        runs[d.name] = dict(
            pattern=own, kind=m.get("kind"), parameter=m.get("parameter"),
            anion_max_relative_change=m.get("anion_max_relative_change"),
            anion_mean_relative_change=m.get("anion_mean_relative_change"),
            n_cells_changed=m.get("n_cells_changed"),
            n_frames=int(len(v)), t_end=float(t[-1]),
            displacement_first_frame_mV=float(d_own[0]),
            mean_abs_displacement_first_frame_mV=float(np.mean(np.abs(v[0] - ref[own]))),
            d_own_end_mV=float(d_own[-1]), d_other_end_mV=float(d_other[-1]),
            d_own_min_mV=float(d_own.min()), d_other_min_mV=float(d_other.min()),
            settled=rep.settled, remaining_drift=rep.remaining_drift,
            time_constant=rep.time_constant, outcome=outcome, t_return=t_return,
            d_own_trace=d_own.round(4).tolist(), d_other_trace=d_other.round(4).tolist(),
            times=t.tolist(),
        )
        amax = m.get("anion_max_relative_change")
        amean = m.get("anion_mean_relative_change")
        print(f"{d.name:30s} {amax if amax is None else f'{amax:9.3f}':>9} "
              f"{amean if amean is None else f'{amean:10.3f}':>10} {d_own[0]:8.2f} {d_own[-1]:9.3f} "
              f"{d_other[-1]:11.3f} {str(rep.settled):>7s} {outcome:>8s} "
              f"{'-' if t_return is None else f'{t_return:.0f}':>11s}")

    if args.out is not None:
        payload = dict(
            criteria=dict(rate_tolerance=RATE_TOLERANCE, drift_tolerance=DRIFT_TOLERANCE,
                          sustain=SUSTAIN, same_state_threshold=SAME_STATE_THRESHOLD),
            reference=settle, A_vs_B_mV=ab,
            targets={k: {kk: vv for kk, vv in meta[k].items() if kk != "runs"} for k in meta},
            runs=runs,
        )
        args.out.write_text(json.dumps(payload, indent=1, default=float), encoding="utf-8")
        print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
