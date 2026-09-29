"""Power of the surrogate gate on genuine chaos vs series length / initial condition (review only).
usage: probe_power.py <system> <N> <ic_index>
"""
import sys
import time
import warnings

import numpy as np

from probe_common import lorenz, rk4, rossler
from mneme.core import classify_attractor, surrogate_test

warnings.simplefilter("ignore")
system, N, ic = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
ICS = [[1, 1, 1], [-5.0, 7.0, 22.0], [3.0, -4.0, 30.0]]
if system == "lorenz":
    x, dt = rk4(lorenz, ICS[ic], 0.01, N, sub=2)[:, 0], 0.01
else:
    x, dt = rk4(rossler, ICS[ic][:2] + [0.5], 0.1, N, sub=10)[:, 0], 0.1
t0 = time.time()
s = surrogate_test(x, statistic="lambda1", n=39, seed=1, dt=dt)
lab = classify_attractor(s.statistic_value, surrogate=s, oscillatory=True)
print(f"{system:8s} N={N:6d} ic={ic} lambda1={s.statistic_value:7.4f} null={np.mean(s.null_distribution):7.4f}+-{np.std(s.null_distribution):6.4f} "
      f"effect={s.effect_size:7.2f} p={s.p_value:5.3f} signif={s.significant} label={getattr(lab,'value',lab)} emb={s.embedding} sec={time.time()-t0:.0f}", flush=True)
