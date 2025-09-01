# Module 7: Attractor Detection (Recurrence, Lyapunov, Clustering)

- Objectives
  - Detect attractors from temporal field trajectories
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
1) Vary `threshold` and observe recurrence matrix density and attractor counts
2) For Lyapunov, change `evolution_time`; infer stability from mean exponents
3) For clustering, compare DBSCAN vs KMeans (set `n_clusters` via code edit if needed)

Solutions (outline)
- Lower threshold → denser recurrences → more/merged attractors; higher → sparser
- Longer evolution windows smooth estimates; negative averages suggest attracting regions
- DBSCAN finds dense basins; KMeans partitions more uniformly but may miss irregular basins