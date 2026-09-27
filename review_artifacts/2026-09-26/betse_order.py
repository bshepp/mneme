"""Side quest: what do the BETSE runs look like in TRUE frame order? (raw cells, no interpolation)"""
import re
from pathlib import Path

import numpy as np

ROOT = Path("data/betse-results")
RUNS = {
    "sim_1": ROOT / "attractors_1_RESULTS/sim_1/Vmem2D_TextExport",
    "sim_2": ROOT / "attractors_1_RESULTS/sim_2/Vmem2D_TextExport",
    "physiology": ROOT / "physiology_RESULTS/sim_Feb_1/Vmem2D_TextExport",
    "patterns": ROOT / "patterns_RESULTS/RESULTS/sim_ellipse_m/Vmem2D_TextExport",
}


def loader_key(p):  # exactly what betse_loader.py:154 does
    return int(re.search(r"(\d+)", p.stem).group(1))


def true_key(p):
    return int(re.search(r"_(\d+)$", p.stem).group(1))


def stats(X, label):
    steps = np.linalg.norm(np.diff(X, axis=0), axis=1)
    path = steps.sum()
    net = np.linalg.norm(X[-1] - X[0])
    m = X.mean(axis=1)
    dm = np.diff(m)
    mono = max((dm > 0).mean(), (dm < 0).mean())
    Xc = X - X.mean(axis=0)
    s = np.linalg.svd(Xc, compute_uv=False)
    ev = s**2 / np.sum(s**2)
    tail = steps[-max(5, len(steps) // 10):].mean() / (steps[: max(5, len(steps) // 10)].mean() + 1e-30)
    print(f"  {label:12s} max_step={steps.max():8.3f} median_step={np.median(steps):7.4f} path={path:9.2f} net={net:8.2f} "
          f"path/net={path/net:7.2f} mean_monotone={mono:5.2f} late/early_step={tail:6.3f}")
    print(f"  {'':12s} mean Vmem start={m[0]:7.2f} end={m[-1]:7.2f} min={m.min():7.2f} max={m.max():7.2f} | PCA var: "
          + " ".join(f"{v*100:5.2f}%" for v in ev[:5]))


for name, d in RUNS.items():
    files = list(d.glob("Vmem2D_*.csv"))
    lo = sorted(files, key=loader_key)
    tr = sorted(files, key=true_key)
    data = {p: np.loadtxt(p, delimiter=",", skiprows=1)[:, 2] for p in files}
    ncell = {len(v) for v in data.values()}
    print(f"\n{name}: frames={len(files)} cells={ncell}")
    print("  loader order, first 8 frame ids:", [true_key(p) for p in lo[:8]])
    stats(np.array([data[p] for p in lo]), "LOADER order")
    stats(np.array([data[p] for p in tr]), "TRUE order")
