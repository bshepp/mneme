# Module 5: Field Reconstruction (IFT and GP)

- Objectives
  - Understand and use IFT and GP reconstructors
  - Provide sparse observations and positions to reconstruct fields
- Time: 60–90 minutes

## 5.1 API overview
- `mneme.core.field_theory.FieldReconstructor(method='ift'|'gaussian_process'|'neural_field')`
- `fit(observations, positions)` then `reconstruct()` and `uncertainty()`

## 5.2 Minimal example (Python)
```python
import numpy as np
from mneme.core.field_theory import FieldReconstructor

# Suppose we have 40 electrode observations on a 32x32 field
rng = np.random.default_rng(0)
positions = rng.uniform(0, 1, size=(40, 2))  # in [0,1]^2
observations = np.sin(4*np.pi*positions[:,0]) * np.cos(4*np.pi*positions[:,1]) + 0.1*rng.normal(size=40)

recon = FieldReconstructor(method='gaussian_process', resolution=(32, 32), length_scale=0.2, noise_level=0.05)
result = recon.fit_reconstruct(observations, positions)
field = result.field.data
unc = result.uncertainty
```

## 5.3 IFT notes
- Prior covariance controlled by `correlation_length`, `power_spectrum_model`
- Identity fallback is used when the pipeline has no sparse obs; supply `observations` and `positions` to enable real reconstruction

## 5.4 Exercises
1) Reconstruct with GP over multiple `length_scale` values; inspect smoothness and uncertainty
2) Switch to IFT; tune `correlation_length`; compare visualizations
3) Stress test: add noise and see how uncertainty reflects confidence

Solutions (outline)
- Larger `length_scale` → smoother fields; uncertainty smaller where observations dense
- IFT yields similar behavior with different prior formulation; visual inspection + metrics
- Noise increases posterior variance; visualize uncertainty heatmap