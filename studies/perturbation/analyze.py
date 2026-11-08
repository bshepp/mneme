"""Analyse the perturbation study: does a settled pattern return after a
stated displacement, flip to the other pattern, or end somewhere else?
A run cont_pert_<P>_<label> is a continuation of pert_<P>_<label> from its
final state; its times restart at zero.

usage: python studies/perturbation/analyze.py <run_root> [--out results.json]

<run_root> holds RESULTS/<run>/{init,sim}/Vmem2D_TextExport/ as written by
run_one.sh, and perturbations.json from make_configs.py perturb.

Reference states: A and B are the last frame of each settling run's
unperturbed sim phase (settle_A, settle_B); C is the end of the midpoint
run's continuation; D, E and F are the ends of the basin study's coarse
random starts (see REFERENCES). Each perturbed run pert_<P>_<label> starts
from pattern P's saved state, displaced as recorded in perturbations.json.
"other" means within the threshold of no reference; "flipped" means within
the threshold of a reference other than the pattern's own.
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

# Pattern label -> RESULTS/<run>/<phase> whose last frame is the reference.
REFERENCES = {
    "A": "settle_A/sim", "B": "settle_B/sim", "C": "cont_pert_A_toward_f50/sim",
    "D": "basin_L100_r1/init", "E": "basin_L100_r2/init", "F": "basin_L100_r4/init",
}


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
    for label, rel in REFERENCES.items():
        if label in ref:
            continue
        run, phase = rel.split("/")
        d = results / run / phase / "Vmem2D_TextExport"
        if d.is_dir():
            v, t, _, _ = load_phase(results / run, phase, dt)
            rep = assess_steady_state(v, t, rate_tolerance=RATE_TOLERANCE,
                                      drift_tolerance=DRIFT_TOLERANCE, sustain=SUSTAIN)
            ref[label] = v[-1]
            settle[label] = dict(source=rel, settled=rep.settled, remaining_drift=rep.remaining_drift,
                                 end_mean_mV=float(v[-1].mean()), end_spread_mV=float(np.ptp(v[-1])))
            print(f"{label}: from {rel}, settled={rep.settled}, mean {v[-1].mean():.2f} mV")
    if len(ref) < 2:
        print("Need at least two reference states", file=sys.stderr)
        return 1
    ref_pairs = {f"{a} vs {b}": max_diff(ref[a], ref[b])
                 for i, a in enumerate(ref) for b in list(ref)[i + 1:]}
    ab = min(ref_pairs.values())
    print("Reference distances [mV]: " + "; ".join(f"{k} {v:.2f}" for k, v in ref_pairs.items()))
    print(f"Smallest: {ab:.2f} mV; same-state threshold {SAME_STATE_THRESHOLD} mV\n")

    # --- perturbed runs --------------------------------------------------
    header = (f"{'run':30s} {'anion max':>9s} {'anion mean':>10s} {'disp[mV]':>8s} {'d_own end':>9s} "
              f"{'d_other end':>11s} {'settled':>7s} {'outcome':>8s} {'t_return[s]':>11s}")
    print(header)
    runs = {}
    run_dirs = sorted(d for d in results.iterdir()
                      if (d.name.startswith("pert_") or d.name.startswith("cont_pert_"))
                      and any((d / "sim" / "Vmem2D_TextExport").glob("Vmem2D_*.csv")))
    for d in run_dirs:
        # pert_<P>_<label>, or cont_pert_<P>_<label> for a continuation of it
        own = d.name.split("_")[2 if d.name.startswith("cont_") else 1]
        if own not in ref:
            print(f"{d.name}: no reference for pattern {own}; skipped")
            continue
        v, t, _, _ = load_phase(d, "sim", dt)
        d_own = np.array([max_diff(f, ref[own]) for f in v])
        # Distance to the nearest other reference at the end; for A and B
        # that is each other, as in the first round.
        others = {k: max_diff(v[-1], r) for k, r in ref.items() if k != own}
        other = min(others, key=others.get)
        d_other = np.array([max_diff(f, ref[other]) for f in v])
        rep = assess_steady_state(v, t, rate_tolerance=RATE_TOLERANCE,
                                  drift_tolerance=DRIFT_TOLERANCE, sustain=SUSTAIN)
        if d_own[-1] < SAME_STATE_THRESHOLD:
            outcome = "returned"
        elif d_other[-1] < SAME_STATE_THRESHOLD:
            outcome = f"flipped:{other}"
        else:
            outcome = "other"
        t_return = first_sustained_below(d_own, t, SAME_STATE_THRESHOLD)
        m = meta.get(own, {}).get("runs", {}).get(d.name.removeprefix("cont_"), {})
        runs[d.name] = dict(
            pattern=own, kind=m.get("kind"), parameter=m.get("parameter"),
            anion_max_relative_change=m.get("anion_max_relative_change"),
            anion_mean_relative_change=m.get("anion_mean_relative_change"),
            n_cells_changed=m.get("n_cells_changed"),
            n_frames=int(len(v)), t_end=float(t[-1]),
            displacement_first_frame_mV=float(d_own[0]),
            mean_abs_displacement_first_frame_mV=float(np.mean(np.abs(v[0] - ref[own]))),
            nearest_other=other,
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

    # --- describe end states that are neither A nor B ----------------------
    def depolarised(v):
        return v > 0.5 * (v.min() + v.max())

    masks = {k: depolarised(ref[k]) for k in ref}
    others = {}
    for d in run_dirs:
        r = runs.get(d.name)
        if r is None or r["outcome"] != "other":
            continue
        v = load_phase(d, "sim", dt)[0][-1]
        m = depolarised(v)
        others[d.name] = dict(
            n_depolarised=int(m.sum()),
            shared_with={k: int((m & mk).sum()) for k, mk in masks.items()},
            mean_mV=float(v.mean()), spread_mV=float(np.ptp(v)),
            depolarised_mV=float(v[m].mean()), polarised_mV=float(v[~m].mean()),
        )
        runs[d.name]["end_state"] = others[d.name]
    if others:
        print("\nEnd states matching no reference (depolarised cells: "
              + ", ".join(f"{k} {int(m.sum())}" for k, m in masks.items()) + "):")
        for n, o in others.items():
            print(f"  {n}: {o['n_depolarised']} depolarised cells, shared "
                  + ", ".join(f"{k} {c}" for k, c in o["shared_with"].items())
                  + f"; mean {o['mean_mV']:.2f} mV, {o['depolarised_mV']:.1f} / {o['polarised_mV']:.1f} mV")
    pairs = {}
    other_names = list(others)
    for i, a in enumerate(other_names):
        for b in other_names[i + 1:]:
            va = load_phase(results / a, "sim", dt)[0][-1]
            vb = load_phase(results / b, "sim", dt)[0][-1]
            pairs[f"{a} vs {b}"] = max_diff(va, vb)
    if pairs:
        print("  distances between them [mV]: " + "; ".join(f"{k}: {v:.3f}" for k, v in pairs.items()))

    if args.out is not None:
        payload = dict(
            criteria=dict(rate_tolerance=RATE_TOLERANCE, drift_tolerance=DRIFT_TOLERANCE,
                          sustain=SUSTAIN, same_state_threshold=SAME_STATE_THRESHOLD),
            reference=settle, reference_distances_mV=ref_pairs, smallest_reference_distance_mV=ab,
            targets={k: {kk: vv for kk, vv in meta[k].items() if kk != "runs"} for k in meta},
            runs=runs, other_end_states=others, other_pairwise_mV=pairs,
        )
        args.out.write_text(json.dumps(payload, indent=1, default=float), encoding="utf-8")
        print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
