# Mneme

An exploratory research toolkit for studying field-like memory in biological tissue, starting with simulated bioelectric data.

> **Validation status (2026-09-27):** No scientific result produced with Mneme is currently asserted. The BETSE analysis report and the earlier PhysioNet Lyapunov numbers were both withdrawn after a review found defects in the code that produced them. The defects are fixed; the analyses have not yet been re-run.

## What it does

Mneme loads spatial voltage data, reconstructs fields from sparse observations, and measures their topology. Components are sorted into three tiers by how far their output can be relied on. See [docs/SCOPE.md](docs/SCOPE.md).

| Tier | Components |
|---|---|
| **Core** (tested against known answers) | BETSE loading, cubical persistent homology, Wasserstein and bottleneck distances, Gaussian-process and Wiener-filter reconstruction, preprocessing, I/O |
| **Frozen** (documented operating range) | Largest Lyapunov exponent, surrogate significance test, gated attractor classification |
| **Experimental** (not validated) | Attractor detectors, symbolic regression, variational autoencoder, neural field reconstruction |

Experimental components emit `mneme.ExperimentalWarning`. The default pipelines run core stages only.

## Installation

```bash
git clone https://github.com/bshepp/mneme.git
cd mneme

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

pip install -e ".[tda]"   # core, plus GUDHI and POT for topology

# Optional: symbolic regression (needs Julia)
pip install -e ".[pysr]"
```

Python 3.12 or later. For detailed setup see [docs/DEVELOPMENT_SETUP.md](docs/DEVELOPMENT_SETUP.md).

Without GUDHI, persistence is computed for H0 only and a `RuntimeWarning` says so.

## Quick Start

### Topology of a field

```python
import numpy as np
from mneme.core.topology import PersistentHomology, compute_wasserstein_distance

field_a = np.random.default_rng(0).normal(size=(64, 64))
field_b = np.random.default_rng(1).normal(size=(64, 64))

ph = PersistentHomology(max_dimension=1, filtration="sublevel", persistence_threshold=0.0)
h0_a, h1_a = ph.compute_persistence(field_a)
h0_b, h1_b = ph.compute_persistence(field_b)

print(len(h0_a.points), "components,", len(h1_a.points), "loops")
print("H1 distance:", compute_wasserstein_distance(h1_a, h1_b))
```

`sublevel` tracks pits as the threshold rises. `superlevel` tracks peaks, with diagrams expressed in units of the negated field.

### Reconstruction from sparse observations

```python
import numpy as np
from mneme.core import create_reconstructor

rng = np.random.default_rng(0)
positions = rng.uniform(0, 1, (300, 2))
observations = np.sin(2 * np.pi * positions[:, 0])

rec = create_reconstructor("gp_subset", resolution=(64, 64))
rec.fit(observations, positions)
field = rec.reconstruct()
uncertainty = rec.uncertainty()
```

### BETSE simulation output

[BETSE](https://github.com/betsee/betse) is a 2D bioelectric tissue simulator.

```python
from mneme.data.betse_loader import load_betse_cells, betse_to_field

# Voltages at the cells, in time order. No interpolation.
vmem, x, y, frames = load_betse_cells("path/to/Vmem2D_TextExport/")
# vmem: shape (n_timesteps, n_cells), in mV

# Or interpolated to a regular grid, for topology
field = betse_to_field("path/to/Vmem2D_TextExport/", resolution=(64, 64))
inside = field.metadata["inside_hull"]   # False where values are fill
```

Prefer `load_betse_cells()` unless you need a grid. Grid values outside the convex hull of the cells are nearest-neighbour fill, not simulation output.

### Lyapunov analysis (frozen)

```python
from mneme.core import largest_lyapunov, surrogate_test, classify_attractor

res = largest_lyapunov(series, dt=0.01)
sur = surrogate_test(series, statistic="lambda1", n=200, dt=0.01)
label = classify_attractor(res.lambda1, surrogate=sur)
```

Read [docs/LYAPUNOV_OPERATING_RANGE.md](docs/LYAPUNOV_OPERATING_RANGE.md) before using a number from these. In brief:

- The surrogate test needs about 4,000 points to detect chaos.
- `STRANGE` means "consistent with chaos", and is returned only with a passed surrogate test.
- λ₁ was off by 14% to 81% away from the conditions it was tuned on.

### Command line

```bash
mneme generate -o sample_data.npz
mneme analyze sample_data.npz --pipeline bioelectric -o results
mneme analyze sample_data.npz --topology-backend rips -o results
```

`mneme analyze` prints the status of each stage and exits non-zero if a stage fails.

Attractor detection is experimental and off by default. Opt in with `--attractor-method {recurrence,lyapunov,clustering}`.

## Reconstruction Methods

| Method | Name | Cost | Notes |
|---|---|---|---|
| Subset GP (default) | `gp_subset` | O(m³), m = subset size | Fits a GP to a random subset of the observations and discards the rest |
| Wiener filter | `wiener_filter` | O(n³), n = grid points | Small grids only |
| Standard GP | `gaussian_process` | O(n³), n = observations | Uses every observation |
| Neural field | `neural_field` | per epoch | Experimental; no uncertainty estimate |

The names `ift`, `sparse_gp` and `dense_ift`, and the classes `SparseGPReconstructor`, `IFTReconstructor` and `DenseIFTReconstructor`, still work and emit a `DeprecationWarning`. The methods behind them are unchanged; the old names described them inaccurately.

## Project Structure

```
mneme/
├── src/mneme/
│   ├── core/           # Reconstruction, topology, Lyapunov tools, attractor detectors
│   ├── analysis/       # Pipeline, visualization, metrics
│   ├── data/           # Generators, loaders, preprocessors, BETSE loader
│   ├── models/         # VAE, symbolic regression (experimental)
│   └── utils/          # Config, logging, I/O
├── scripts/            # Analysis scripts
├── notebooks/          # Demo notebooks
├── tests/              # Test suite
└── docs/               # Documentation
```

## Documentation

- [Scope and Support Status](docs/SCOPE.md) — what is core, frozen and experimental
- [Lyapunov Operating Range](docs/LYAPUNOV_OPERATING_RANGE.md) — measured accuracy and limits
- [Project Structure](docs/PROJECT_STRUCTURE.md) — code organization
- [Development Setup](docs/DEVELOPMENT_SETUP.md) — environment setup
- [Data Pipeline](docs/DATA_PIPELINE.md) — pipeline stages
- [Course](docs/course/README.md) — 11-module learning course

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT. See [LICENSE](LICENSE).

## Acknowledgments

- Inspired by work on bioelectric patterns in regeneration (Levin Lab)
- BETSE, GUDHI, PySR and scikit-learn
