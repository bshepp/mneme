# CLAUDE.md

Developer context for the Mneme project. Useful for both AI assistants and human contributors working on this codebase.

## Project Overview

Mneme is an exploratory research toolkit for studying field-like memory in biological tissue, starting with simulated bioelectric data (BETSE).

## Current Status (2026-09-27)

**No scientific result produced with Mneme is currently asserted.** A review in September 2026 found that the BETSE analysis report and the PhysioNet Lyapunov numbers were produced by defective code. Both are withdrawn. The defects are fixed; the analyses have not been re-run.

The plan is in [project_plan.md](project_plan.md). Two studies under [docs/MULTISTABILITY_PROTOCOL.md](docs/MULTISTABILITY_PROTOCOL.md) are done: the published 2016 configuration has one stable state ([studies/convergence](studies/convergence/RESULTS.md)); the 2018 gene-network configuration has two stable voltage patterns under one parameter set, while Kir2.1 plus a Na⁺ leak gives one ([studies/multistability](studies/multistability/RESULTS.md)). These are simulation results with one geometry and one parameter set each.

### Component tiers

Full detail in [docs/SCOPE.md](docs/SCOPE.md).

| Tier | Components |
|---|---|
| **Core** (tested against known answers) | BETSE loading, cubical persistent homology, Wasserstein and bottleneck distances, subset-GP / standard-GP / Wiener-filter reconstruction, preprocessing, I/O |
| **Frozen** (documented operating range, no development) | `largest_lyapunov`, `surrogate_test`, `classify_attractor`, `lyapunov_spectrum`, `kaplan_yorke_dimension`, `mneme.core.embedding` |
| **Experimental** (not validated) | `mneme.core.attractors` detectors, symbolic regression, VAE, neural field reconstruction, quality checker |

Experimental components emit `mneme.ExperimentalWarning`. Default pipelines run core stages only.

## Rules for claims

1. Report a result only with a null model or baseline beside it.
2. Only core-tier output supports a claim.
3. Use a method only on data inside its measured operating range.
4. Replicate conditions. One run per condition demonstrates nothing.

## Development Environment

- **Python**: 3.12+ required
- **Core Dependencies**: numpy, scipy, pandas, scikit-learn, torch, matplotlib
- **Optional Dependencies**: gudhi and POT (`.[tda]`), pysr (`.[pysr]`)
- The `venv/` directory in the repo root is a stale Linux environment from an earlier checkout path. Create a fresh one.

## Key Commands

```bash
pip install -e ".[dev,tda]"

# Run tests (about 5 minutes)
pytest tests

# Without installing the package
PYTHONPATH=src python -m pytest tests

# CLI usage
mneme generate -o sample_data.npz
mneme analyze sample_data.npz --pipeline bioelectric -o results
mneme info
```

## Module Structure

```
src/mneme/
├── _status.py             # ExperimentalWarning, warn_experimental()
├── core/
│   ├── field_theory.py    # SubsetGPReconstructor, WienerFilterReconstructor, etc.
│   ├── topology.py        # PersistentHomology, RipsComplex, AlphaComplex, distances
│   ├── embedding.py       # delay embedding, MI delay, Cao dimension (frozen)
│   ├── lyapunov.py        # largest_lyapunov, lyapunov_spectrum (frozen)
│   ├── surrogates.py      # IAAFT surrogates, surrogate_test (frozen)
│   ├── classify.py        # classify_attractor, kaplan_yorke_dimension (frozen)
│   └── attractors.py      # RecurrenceAnalysis, ClusteringDetector (experimental)
├── analysis/
│   ├── pipeline.py        # MnemePipeline, default_config(), merge_config()
│   ├── steady_state.py    # assess_steady_state(), count_distinct_states()
│   └── visualization.py   # FieldVisualizer, dashboards
├── data/
│   ├── generators.py      # SyntheticFieldGenerator
│   ├── preprocessors.py   # Denoiser, Normalizer, Interpolator
│   └── betse_loader.py    # load_betse_cells(), load_betse_timeseries(), betse_to_field()
├── models/                # experimental
│   ├── autoencoders.py    # FieldAutoencoder (Conv VAE)
│   └── symbolic.py        # SymbolicRegressor, discover_field_dynamics()
└── utils/
    ├── config.py
    └── io.py

scripts/                   # not re-run since the fixes; see Known Issues
studies/convergence/       # BETSE convergence study: configs, runner, analysis, results
studies/multistability/    # gene-network and channel studies: configs, results
review_artifacts/          # probe scripts and outputs behind the measured numbers
```

## Important Implementation Notes

### Field Reconstruction
```python
from mneme.core import create_reconstructor

# Default: GP fitted to a random subset of the observations
rec = create_reconstructor('gp_subset', resolution=(256, 256), n_subset=500)

# Dense Wiener filter (small fields only)
rec = create_reconstructor('wiener_filter', resolution=(32, 32))
```

The names `ift`, `sparse_gp`, `dense_ift`, `SparseGPReconstructor`, `IFTReconstructor`, `DenseIFTReconstructor` and `n_inducing` are deprecated aliases. They still work and emit `DeprecationWarning`.

### Topology
```python
from mneme.core.topology import PersistentHomology

ph = PersistentHomology(max_dimension=1, filtration="sublevel")
h0, h1 = ph.compute_persistence(field)
```

- `sublevel` tracks pits. `superlevel` tracks peaks, in units of the negated field.
- GUDHI reads cells with the first axis fastest. Pass `values.flatten(order="F")`.
- Without GUDHI only H0 is computed, by union-find, with a `RuntimeWarning`.

### BETSE
```python
from mneme.data.betse_loader import load_betse_cells, betse_to_field

vmem, x, y, frames = load_betse_cells("path/to/Vmem2D_TextExport/")   # preferred
field = betse_to_field("path/to/Vmem2D_TextExport/", resolution=(64, 64))
```

The frame index is the trailing integer of the file name. An unanchored digit search matches the "2" in "Vmem2D".

### Lyapunov tools (frozen)
```python
from mneme.core import largest_lyapunov, surrogate_test, classify_attractor

res = largest_lyapunov(series, dt=0.01)
sur = surrogate_test(series, statistic="lambda1", n=200, dt=0.01)
label = classify_attractor(res.lambda1, surrogate=sur)
```

Read [docs/LYAPUNOV_OPERATING_RANGE.md](docs/LYAPUNOV_OPERATING_RANGE.md) first. Do not retune the detector constants to make a new case pass: they were already tuned on the test fixtures, which is why accuracy falls off elsewhere.

## Contributor Guidance

1. **Test against a known answer.** A test that checks only shape, type or absence of an exception does not count toward moving a component into the core tier.
2. **Do not tune a constant and its test in the same change** without a held-out case.
3. **Fail loudly.** A stage that cannot produce a result raises or reports failure. It does not return zeros, empties or its input.
4. **Fallbacks warn.** Any path that substitutes a weaker method emits a warning.
5. **Preserve backwards compatibility** for names, with `DeprecationWarning`.
6. **Update docs**: keep README.md, CLAUDE.md, docs/SCOPE.md and CHANGELOG.md in sync.

## Known Issues / TODOs

- The scripts in `scripts/` have not been re-run since the fixes. They compute λ₁ on multi-dimensional trajectories while testing only the first column, and they still save the exploratory spectrum and D_KY.
- The BETSE runs on hand are 119 to 635 frames and are relaxations toward rest. They are outside the operating range of the Lyapunov tools.
- BETSE's default gap-junction coupling is numerically unstable at a time step of 0.01 s. The convergence study used a coupling 50 times weaker.
- Import order warning: import juliacall before torch to avoid a potential segfault. On Windows, PySR prints "access violation" traces during tests that still pass.
- `mypy` runs in CI but cannot fail it.
- `compute_basin_of_attraction()` was removed; design notes are in [docs/FUTURE_IDEAS.md](docs/FUTURE_IDEAS.md).
