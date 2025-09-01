# Module 6: Topology (Cubical, Rips, Alpha) and Features

- Objectives
  - Compute persistence via cubical, Rips, Alpha backends
  - Extract feature vectors; understand persistence images/landscapes
- Time: 60–90 minutes

## 6.1 Cubical persistence
```python
import numpy as np
from mneme.core.topology import PersistentHomology

field = np.random.randn(64,64)
ph = PersistentHomology(max_dimension=2, filtration='sublevel', persistence_threshold=0.05)
diagrams = ph.compute_persistence(field)
features = ph.extract_features(diagrams)
```

## 6.2 Rips/Alpha via adapter
```python
from mneme.core.topology import RipsComplex, AlphaComplex, field_to_point_cloud

pc = field_to_point_cloud(field, method='peaks', percentile=95.0, max_points=2000)
rc = RipsComplex(max_dimension=1)
rd = rc.compute_persistence(pc)
rf = rc.extract_features(rd)
```

## 6.3 Persistence images/landscapes (MVP)
- Use `compute_persistence_image`/`compute_persistence_landscape` helpers for derived representations

## Exercises
1) Compare feature vectors across backends on the same field
2) Increase `percentile` in `field_to_point_cloud` and see how point density affects Rips results
3) Plot a persistence image for H1 and interpret visually

Solutions (outline)
- Different backends produce different sensitivities; features reflect scale and sampling
- Higher threshold → fewer points → sparser complexes
- Bright areas in persistence image correspond to long-lived features