# Module 11 (Optional): Symbolic Regression with PySR

- Objectives
  - Understand where symbolic regression may fit (post-feature extraction)
  - Run a small regression to recover simple dynamics
- Time: 45–60 minutes (optional)

## 11.1 Caveats
- PySR is optional and may install Julia on first use
- MVP integrates lightly; treat this as exploratory

## 11.2 Minimal example (toy data)
```python
import numpy as np
from pysr import PySRRegressor

rng = np.random.default_rng(0)
X = rng.uniform(-1,1,(200,2))
y = np.sin(X[:,0]) + 0.5*X[:,1]**2

model = PySRRegressor(niterations=20, binary_operators=["+","*","-"], unary_operators=["sin","cos"])
model.fit(X, y)
print(model.get_best())
```

## Exercises
1) Use topological features as `X` and a known morphodynamic target as `y`; see if simple expressions emerge
2) Inspect equation complexity vs error

Solutions (outline)
- Topological summaries may correlate with morphological regimes; use as features for interpretable rules
- Complexity penalties steer PySR toward simpler, more generalizable equations