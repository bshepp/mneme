# Module 11 (Optional): Symbolic Regression with PySR

- Objectives
  - Understand where symbolic regression may fit (post-feature extraction)
  - Run a small regression to recover simple dynamics
- Time: 45–60 minutes (optional)

## 11.1 Caveats
- PySR is optional and may install Julia on first use
- Recommended install: `pip install -e .[pysr]` (pins scikit-learn and juliacall compatibly)
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

Run log (MVP)
- With `pip install -e .[pysr]`, PySR should work with compatible scikit-learn/juliacall versions. If Julia setup is deferred, the first call may download/build Julia artifacts.

## 11.3 Using topological features as inputs

In this section, we treat persistence-derived features as explanatory variables and try to learn an interpretable mapping to a target quantity. We’ll synthesize a target first, then show how to use features from a pipeline run.

### 11.3.1 Direct feature construction from fields

```python
import numpy as np
from mneme.core.topology import PersistentHomology
from pysr import PySRRegressor

# Create K random fields and compute PH features
rng = np.random.default_rng(0)
K = 100
ph = PersistentHomology(max_dimension=2, filtration='sublevel', persistence_threshold=0.03)
X = []
for _ in range(K):
    field = rng.normal(size=(64, 64))
    diags = ph.compute_persistence(field)
    feats = ph.extract_features(diags)  # shape ~ (6 * (max_dimension+1),)
    X.append(feats)
X = np.vstack(X)

# Synthesize a target as a simple function of features
# (e.g., y = 0.5 * sum of top 3 feature components + sin of the next)
y = 0.5 * X[:, 0] + 0.3 * X[:, 1] + np.sin(X[:, 2])

# Fit PySR
model = PySRRegressor(
    niterations=40,
    binary_operators=["+", "*", "-"],
    unary_operators=["sin", "cos", "exp"],
    model_selection="best",
    maxsize=20,
    progress=False,
)
model.fit(X, y)

print("Best equation:")
print(model.get_best())

# Evaluate R^2 on a held-out mini-set
X_test = []
for _ in range(20):
    field = rng.normal(size=(64, 64))
    diags = ph.compute_persistence(field)
    feats = ph.extract_features(diags)
    X_test.append(feats)
X_test = np.vstack(X_test)
y_test = 0.5 * X_test[:, 0] + 0.3 * X_test[:, 1] + np.sin(X_test[:, 2])
from sklearn.metrics import r2_score
print("R2:", r2_score(y_test, model.predict(X_test)))
```

Expected outcome: PySR recovers a simple, low-complexity expression approximating the synthetic rule with high R^2.

### 11.3.2 Using features from a pipeline result (HDF5)

You can also run the pipeline once (Module 3/4), then use the saved features under `topology/features` as inputs:

```python
from mneme.utils.io import load_results
from pysr import PySRRegressor
import numpy as np

ar = load_results("results_cli/analysis_results.hdf5")  # returns AnalysisResult
if ar.topology is None or ar.topology.features is None:
    raise RuntimeError("No topology features found in results. Re-run with topology enabled.")

# For demonstration, treat the feature vector as a single sample; in practice,
# assemble a matrix across many samples (multiple fields or timepoints) before fitting.
X = ar.topology.features.reshape(1, -1)
y = np.array([np.sum(X)])  # dummy target; replace with a true label or metric

model = PySRRegressor(niterations=10, binary_operators=["+","*","-"], unary_operators=["sin","cos"], progress=False)
model.fit(X, y)
print(model.get_best())
```

Note: For real learning, collect features from many samples (e.g., multiple runs, timepoints, or fields) and build a proper dataset `(X, y)`.

## 11.4 Tips and best practices
- Normalize/standardize features (z-score) before symbolic regression when magnitudes vary widely
- Keep operator sets small initially (e.g., `+ - *`, plus one or two unary ops)
- Limit expression size (`maxsize`) to encourage interpretable formulas
- Cross-validate by splitting fields/timepoints to gauge generalization
- Use domain knowledge to craft targets (e.g., known morphometric metrics) and constrain operator sets

## 11.5 Exercises
1) Build a dataset of (features, target) from 50–100 synthetic fields; recover a simple target like `sin(f0) + 0.2*f1` and report R^2 on a held-out set
2) Restrict operators to `+ - *` and observe how accuracy vs simplicity trades off
3) Use features from Rips/Alpha backends (via `field_to_point_cloud`) and compare the discovered forms
4) If you have labels from experiments, try replacing synthetic `y` with your real target; report the simplest model that achieves R^2 > 0.7

Run log (MVP)
- With `pip install -e .[pysr]`, the synthetic feature regression converged quickly to a low-complexity formula with high R^2. Pipeline-derived features worked once multiple samples were aggregated.

## Exercises
1) Use topological features as `X` and a known morphodynamic target as `y`; see if simple expressions emerge
2) Inspect equation complexity vs error

Solutions (outline)
- Topological summaries may correlate with morphological regimes; use as features for interpretable rules
- Complexity penalties steer PySR toward simpler, more generalizable equations