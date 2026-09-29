# Module 7: Attractor Detection (Recurrence, Lyapunov, Clustering)

> **Experimental.** The detectors in `mneme.core.attractors` are experimental and emit `mneme.ExperimentalWarning` when constructed; see [SCOPE.md](../SCOPE.md). They locate dense or recurrent regions of a trajectory but cannot say what kind of attractor a region is: every `Attractor` they return has `type == AttractorType.UNDETERMINED`. Do not use their output as evidence for a scientific claim. Claims about chaos go through the frozen tools in 7.5.

- Objectives
  - Locate candidate attractor regions in temporal field trajectories
  - Tune thresholds and method-specific parameters
- Time: 60–90 minutes

## 7.1 Recurrence (default)
```python
import numpy as np
from mneme.core.attractors import AttractorDetector

# Create a simple 2D oscillation
t = np.linspace(0, 10, 200)
traj = np.column_stack([np.sin(t), np.cos(t)])

ad = AttractorDetector(method='recurrence', threshold=0.2, min_persistence=0.1, embedding_dimension=3, time_delay=1)
attractors = ad.detect(traj)
```

## 7.2 Lyapunov (basic MVP)
```python
ad = AttractorDetector(method='lyapunov', threshold=0.05, n_neighbors=10, evolution_time=5)
attractors = ad.detect(traj)
```

## 7.3 Clustering (DBSCAN/KMeans)
```python
ad = AttractorDetector(method='clustering', threshold=0.2, min_samples=20, clustering_method='dbscan')
attractors = ad.detect(traj)
```

## 7.4 Exercises
1) Vary `threshold` and observe recurrence matrix density and the number of regions found
2) For Lyapunov, change `evolution_time` and observe how the local exponent averages attached to each region change (these are not a classification; see 7.5)
3) For clustering, compare DBSCAN vs KMeans (set `n_clusters` via code edit if needed)

Run log (MVP)
- Recurrence: Returned several regions on a simple circular trajectory (expected, due to recurrence clustering). Adjust thresholds for control.
- Lyapunov: Returned 0 regions on the simple sinusoid with default params (expected — near-neutral exponents). Increase `evolution_time` or apply to longer, more structured trajectories.
- Clustering: Returned 0 for the toy sinusoid with `min_samples=20` (expected). Reduce `min_samples` or use denser/recurrent trajectories to see clusters.

Solutions (outline)
- Lower threshold → denser recurrences → more/merged regions; higher → sparser
- Longer evolution windows smooth the local exponent estimates
- DBSCAN finds dense clusters; KMeans partitions more uniformly but may miss irregular ones

## 7.5 Classifying an attractor (frozen Lyapunov tools)
The detectors above never assign fixed-point, limit-cycle or strange labels. For that, use the frozen tools in `mneme.core`:

```python
from mneme.core import largest_lyapunov, surrogate_test, classify_attractor

res = largest_lyapunov(series, dt=0.01)                       # Rosenstein 1993
sur = surrogate_test(series, statistic="lambda1", n=39, dt=0.01)
label = classify_attractor(res.lambda1, surrogate=sur)        # STRANGE only if sur.significant
```

- `classify_attractor` returns `UNDETERMINED` unless there is surrogate evidence (`STRANGE`) or you assert `oscillatory=True/False` yourself
- The surrogate test needs at least 39 surrogates at α = 0.05 (it raises below that) and about 4,000 points to have power (it warns below that)
- `STRANGE` means "consistent with chaos", not a proof of it
- Read [Lyapunov Operating Range](../LYAPUNOV_OPERATING_RANGE.md) before using any number from these tools; λ₁ was off by tens of percent away from the conditions it was tuned on