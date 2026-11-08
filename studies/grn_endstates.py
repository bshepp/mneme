"""End-state analysis shared by the basins and sweep studies.

Each run's init phase (6,000 s) is loaded, judged settled or not by the
criteria of studies/convergence, and its final frame is compared with the
reference patterns A and B from studies/perturbation (the last frame of
settle_A's and settle_B's control continuation, in the same world).

usage, from a study's analyze.py::

    from grn_endstates import analyze
    analyze(run_root, prefix="basin_", group=lambda name: "all", out=path)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from mneme.analysis.steady_state import assess_steady_state, count_distinct_states
from mneme.data.betse_loader import load_betse_cells

RATE_TOLERANCE = 1.0e-4
DRIFT_TOLERANCE = 0.1
SUSTAIN = 20
SAME_STATE_THRESHOLD = 1.0  # mV


def load_phase(run_dir: Path, phase: str, dt: float):
    v, x, y, frames = load_betse_cells(run_dir / phase / "Vmem2D_TextExport")
    return v, (np.asarray(frames, float) + 1.0) * dt, x, y


def max_diff(a, b):
    return float(np.max(np.abs(a - b)))


# The third state of studies/perturbation, reached from the midpoint of the
# A-B line and held for a further 6,000 s, is a reference too when present.
EXTRA_REFERENCES = {"C": "cont_pert_A_toward_f50/sim"}


def depolarised(v):
    return v > 0.5 * (v.min() + v.max())


def analyze(run_root: Path, prefix: str, group, out: Path | None = None, dt: float = 30.0,
            extra=None):
    results = run_root / "RESULTS"
    refs = {}
    ref_xy = None
    for label in ("A", "B"):
        d = results / f"settle_{label}" / "sim" / "Vmem2D_TextExport"
        if d.is_dir():
            v, x, y, _ = load_betse_cells(d)
            refs[label] = v[-1]
            ref_xy = np.column_stack([x, y]) if ref_xy is None else ref_xy
    for label, rel in EXTRA_REFERENCES.items():
        d = results / rel / "Vmem2D_TextExport"
        if d.is_dir():
            refs[label] = load_betse_cells(d)[0][-1]
    ab = max_diff(refs["A"], refs["B"]) if {"A", "B"} <= set(refs) else None
    print(f"References: {sorted(refs)}; A vs B {ab if ab is None else f'{ab:.2f}'} mV; "
          f"same-state threshold {SAME_STATE_THRESHOLD} mV")
    masks = {k: depolarised(r) for k, r in refs.items()}
    print("Depolarised cells: " + ", ".join(f"{k} {int(m.sum())}" for k, m in masks.items()))

    runs = {}
    for d in sorted(results.iterdir()):
        if not d.name.startswith(prefix):
            continue
        if not any((d / "init" / "Vmem2D_TextExport").glob("Vmem2D_*.csv")):
            continue
        v, t, x, y = load_phase(d, "init", dt)
        if ref_xy is not None and not np.allclose(np.column_stack([x, y]), ref_xy):
            raise ValueError(f"{d.name}: cell positions differ from the reference world; "
                             "per-cell distances would compare unrelated cells")
        rep = assess_steady_state(v, t, rate_tolerance=RATE_TOLERANCE,
                                  drift_tolerance=DRIFT_TOLERANCE, sustain=SUSTAIN)
        dist = {k: max_diff(v[-1], r) for k, r in refs.items()}
        nearest = min(dist, key=dist.get) if dist else None
        if nearest and dist[nearest] < SAME_STATE_THRESHOLD:
            state = nearest
        elif np.ptp(v[-1]) < SAME_STATE_THRESHOLD:
            state = "uniform"      # every cell at the same voltage: no pattern
        else:
            state = "other"
        runs[d.name] = dict(
            group=group(d.name), n_frames=int(len(v)), t_end=float(t[-1]),
            start_mean=float(v[0].mean()), start_spread=float(np.ptp(v[0])),
            end_mean=float(v[-1].mean()), end_spread=float(np.ptp(v[-1])),
            settled=rep.settled, remaining_drift=rep.remaining_drift,
            time_constant=rep.time_constant,
            distance_to=dist, state=state, end=v[-1],
        )
        if extra is not None:
            runs[d.name].update(extra(d.name))

    ref_cols = sorted(refs)
    print(f"\n{'run':28s} {'frames':>6s} {'start mean':>10s} {'end mean':>8s} {'spread':>7s} "
          f"{'to go':>7s} {'settled':>7s} " + " ".join(f"{'d(' + k + ')':>7s}" for k in ref_cols)
          + f" {'state':>6s}")
    for n, r in runs.items():
        print(f"{n:28s} {r['n_frames']:6d} {r['start_mean']:10.2f} {r['end_mean']:8.2f} "
              f"{r['end_spread']:7.2f} {r['remaining_drift']:7.3f} {str(r['settled']):>7s} "
              + " ".join(f"{r['distance_to'][k]:7.2f}" for k in ref_cols) + f" {r['state']:>6s}")

    # Describe end states that match no reference, and compare them with each other.
    others = [n for n, r in runs.items() if r["state"] in ("other", "uniform")]
    for n in others:
        v = runs[n]["end"]
        m = depolarised(v)
        runs[n]["end_state"] = dict(
            n_depolarised=int(m.sum()),
            shared_with={k: int((m & mk).sum()) for k, mk in masks.items()},
            depolarised_mV=float(v[m].mean()), polarised_mV=float(v[~m].mean()),
        )
    if others:
        print("\nEnd states matching no reference:")
        for n in others:
            e = runs[n]["end_state"]
            print(f"  {n}: {e['n_depolarised']} depolarised cells, shared "
                  + ", ".join(f"{k} {c}" for k, c in e["shared_with"].items())
                  + f"; {e['depolarised_mV']:.1f} / {e['polarised_mV']:.1f} mV")
        pairs = {f"{a} vs {b}": max_diff(runs[a]["end"], runs[b]["end"])
                 for i, a in enumerate(others) for b in others[i + 1:]}
        if pairs:
            print("  between them [mV]: " + "; ".join(f"{k}: {v:.2f}" for k, v in pairs.items()))

    groups = {}
    for g in sorted({r["group"] for r in runs.values()}):
        members = [n for n, r in runs.items() if r["group"] == g]
        ends = [runs[n]["end"] for n in members]
        # Count distinct states among the runs plus the two references.
        found = count_distinct_states(ends + [refs[k] for k in sorted(refs)], SAME_STATE_THRESHOLD)
        labels = found.labels.tolist()
        new_states = len(set(labels[: len(members)]) - set(labels[len(members):]))
        n_uniform = sum(runs[n]["state"] == "uniform" for n in members)
        groups[g] = dict(
            members=members, n_distinct_with_references=found.n_states,
            n_states_not_A_or_B=new_states, n_uniform=n_uniform,
            states=[runs[n]["state"] for n in members],
            all_settled=all(runs[n]["settled"] for n in members),
            labels=labels,
        )
        print(f"\n[{g}] {len(members)} runs: states {groups[g]['states']}; "
              f"{found.n_states} distinct among runs + references, "
              f"{new_states} not A or B; all settled: {groups[g]['all_settled']}")

    if out is not None:
        payload = dict(
            criteria=dict(rate_tolerance=RATE_TOLERANCE, drift_tolerance=DRIFT_TOLERANCE,
                          sustain=SUSTAIN, same_state_threshold=SAME_STATE_THRESHOLD),
            A_vs_B_mV=ab,
            runs={n: {k: v for k, v in r.items() if k != "end"} for n, r in runs.items()},
            groups=groups,
        )
        out.write_text(json.dumps(payload, indent=1, default=float), encoding="utf-8")
        print(f"\nWrote {out}")
    return runs, groups
