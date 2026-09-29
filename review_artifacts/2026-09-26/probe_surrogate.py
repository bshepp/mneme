"""Does the surrogate gate + classify_attractor emit STRANGE for non-chaotic inputs? (review only)

n=39 surrogates: smallest two-sided p is 2/40 = 0.05 <= alpha, so significance is attainable.
"""
import sys
import time
import warnings

import numpy as np

from probe_common import ar1, ar2_narrowband, lorenz, noisy_vdp, rk4, rossler
from mneme.core import classify_attractor, surrogate_test

warnings.simplefilter("ignore")
N = 3000
NS = 39
rng = np.random.RandomState(777)
t = np.arange(N)


def tar(n, rng):
    """Threshold AR: NONLINEAR but purely stochastic, non-chaotic."""
    x = np.zeros(n + 500)
    e = rng.standard_normal(n + 500)
    for i in range(1, n + 500):
        x[i] = (0.9 if x[i - 1] < 0 else -0.4) * x[i - 1] + e[i]
    return x[500:]


def nonstationary(n, rng):
    """Linear AR(1) whose variance drifts: violates the stationarity the null assumes."""
    return ar1(n, 0.9, rng) * np.linspace(0.3, 3.0, n)


cases = [
    ("Lorenz x dt=0.01", lambda: rk4(lorenz, [1, 1, 1], 0.01, N, sub=2)[:, 0], 0.01, "chaotic"),
    ("Rossler x dt=0.1", lambda: rk4(rossler, [1, 1, 1], 0.1, N, sub=10)[:, 0], 0.1, "chaotic"),
    ("white noise", lambda: rng.standard_normal(N), 1.0, "NOT chaotic"),
    ("AR(1) phi=0.95", lambda: ar1(N, 0.95, rng), 1.0, "NOT chaotic"),
    ("AR(2) narrowband", lambda: ar2_narrowband(N, rng), 1.0, "NOT chaotic"),
    ("sine + 5% noise", lambda: np.sin(2 * np.pi * t / 50.0) + 0.05 * rng.standard_normal(N), 1.0, "NOT chaotic"),
    ("noisy VdP (dynamical noise)", lambda: noisy_vdp(N, rng), 0.05, "NOT chaotic"),
    ("threshold-AR (nonlinear stoch.)", lambda: tar(N, rng), 1.0, "NOT chaotic"),
    ("nonstationary AR(1)", lambda: nonstationary(N, rng), 1.0, "NOT chaotic"),
]
only = sys.argv[1:]
print(f"{'case':34s} {'truth':>12s} {'lambda1':>9s} {'null_mean':>9s} {'null_sd':>8s} {'effect':>7s} {'p':>6s} {'signif':>6s} {'LABEL':>13s} {'sec':>5s}")
for name, gen, dt, truth in cases:
    if only and not any(o.lower() in name.lower() for o in only):
        continue
    t0 = time.time()
    try:
        x = gen()
        s = surrogate_test(x, statistic="lambda1", n=NS, seed=1, dt=dt)
        lab = classify_attractor(s.statistic_value, surrogate=s, oscillatory=True)
        lab = getattr(lab, "value", str(lab))
        print(f"{name:34s} {truth:>12s} {s.statistic_value:9.4f} {np.mean(s.null_distribution):9.4f} {np.std(s.null_distribution):8.4f} {s.effect_size:7.2f} {s.p_value:6.3f} {str(s.significant):>6s} {lab:>13s} {time.time()-t0:5.0f}", flush=True)
    except Exception as e:
        print(f"{name:34s} ERROR {type(e).__name__}: {e}", flush=True)
