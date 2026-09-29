"""Held-out probes of largest_lyapunov + classify_attractor (review only)."""
import time
import warnings

import numpy as np

from probe_common import ar1, ar2_narrowband, chen, lorenz, noisy_vdp, rk4, rossler, vdp
from mneme.core import classify_attractor, largest_lyapunov

warnings.simplefilter("ignore")
N = 6000
rng = np.random.RandomState(12345)
t = np.arange(N)

cases = []  # (name, series, dt, truth)
for dt in (0.005, 0.01, 0.02, 0.05):
    cases.append((f"Lorenz x dt={dt}", rk4(lorenz, [1, 1, 1], dt, N, sub=max(1, int(dt / 0.005)))[:, 0], dt, "0.906"))
cases.append(("Lorenz z dt=0.01", rk4(lorenz, [1, 1, 1], 0.01, N, sub=2)[:, 2], 0.01, "0.906"))
cases.append(("Lorenz 3-D dt=0.01", rk4(lorenz, [1, 1, 1], 0.01, N, sub=2), 0.01, "0.906"))
for dt in (0.05, 0.1, 0.2):
    cases.append((f"Rossler x dt={dt}", rk4(rossler, [1, 1, 1], dt, N, sub=max(1, int(dt / 0.01)))[:, 0], dt, "0.071"))
for dt in (0.002, 0.005):
    cases.append((f"Chen x dt={dt} (held-out)", rk4(chen, [-3, 2, 20], dt, N, sub=max(1, int(dt / 0.001)))[:, 0], dt, "~2.0"))
cases.append(("Lorenz x dt=0.01 + 5% obs noise", cases[1][1] + 0.05 * cases[1][1].std() * rng.standard_normal(N), 0.01, "0.906"))

cases.append(("sine (period 50)", np.sin(2 * np.pi * t / 50.0), 1.0, "0"))
cases.append(("sine + 1% noise", np.sin(2 * np.pi * t / 50.0) + 0.01 * rng.standard_normal(N), 1.0, "0"))
cases.append(("sine + 20% noise", np.sin(2 * np.pi * t / 50.0) + 0.2 * rng.standard_normal(N), 1.0, "0"))
cases.append(("quasi-periodic 2-torus", np.sin(2 * np.pi * t / 50.0) + np.sin(2 * np.pi * t / (50.0 * np.sqrt(2))), 1.0, "0"))
cases.append(("Van der Pol x dt=0.05", rk4(vdp, [1, 0], 0.05, N, sub=5)[:, 0], 0.05, "0"))
cases.append(("noisy VdP (dynamical noise)", noisy_vdp(N, rng), 0.05, "0 (stochastic)"))
cases.append(("white noise", rng.standard_normal(N), 1.0, "n/a (stochastic)"))
cases.append(("AR(1) phi=0.95", ar1(N, 0.95, rng), 1.0, "n/a (stochastic)"))
cases.append(("AR(2) narrowband", ar2_narrowband(N, rng), 1.0, "n/a (stochastic)"))
cases.append(("random walk", np.cumsum(rng.standard_normal(N)), 1.0, "n/a (stochastic)"))

print(f"{'case':36s} {'truth':>16s} {'lambda1':>9s} {'R2':>6s} {'m':>3s} {'tau':>4s} {'thl':>4s} {'fit':>11s} {'label(no surrogate)':>20s} {'sec':>5s}")
for name, x, dt, truth in cases:
    t0 = time.time()
    try:
        r = largest_lyapunov(x, dt=dt)
        lab = classify_attractor(r.lambda1).value if hasattr(classify_attractor(r.lambda1), "value") else str(classify_attractor(r.lambda1))
        print(f"{name:36s} {truth:>16s} {r.lambda1:9.4f} {r.fit_r2:6.3f} {r.emb_dim:3d} {r.delay:4d} {r.theiler:4d} {str(r.fit_region):>11s} {lab:>20s} {time.time()-t0:5.1f}", flush=True)
    except Exception as e:  # report, don't hide
        print(f"{name:36s} ERROR {type(e).__name__}: {e}", flush=True)
