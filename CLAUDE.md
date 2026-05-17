# CLAUDE.md

Developer context for the Mneme project. Useful for both AI assistants and human contributors working on this codebase.

## Project Overview

Mneme is an exploratory research system designed to detect field-like, emergent memory structures embedded in biological systems, beginning with planarian regeneration and bioelectric data. The project seeks to uncover attractor states, regulatory logic, and latent architectures not captured by sequence-based models alone.

## Current Status (2026-02-14)

The core system is in **active development** with all major components implemented:

### Implemented Features
- **Field Reconstruction**: Sparse GP (default, scalable), Dense IFT, Standard GP, Neural Fields
- **Topology Analysis**: Full GUDHI integration (cubical, Rips, Alpha complexes)
- **Attractor Detection**: Recurrence, Lyapunov, and clustering methods
- **Lyapunov Spectrum**: Rosenstein-1993 `largest_lyapunov()`, exploratory `lyapunov_spectrum()`, `surrogate_test()`, surrogate-gated `classify_attractor()`, `kaplan_yorke_dimension()`
- **Symbolic Regression**: Full PySR integration with `discover_field_dynamics()`
- **Latent Space Analysis**: Convolutional VAE with training, encoding, interpolation
- **Pipeline**: End-to-end analysis with visualization and HDF5 export
- **Real Data Validation**: PhysioNet ECG/HRV validation pending re-run under Tier 0 corrected estimators
- **BETSE Integration**: Loader for BETSE bioelectric tissue simulation output (`betse_loader.py`)
- **Test Suite**: Unit and integration tests with CI via GitHub Actions

### Key Architecture Decisions
- Sparse GP is the default reconstruction method (scalable to 256×256 fields)
- Dense IFT preserved as `method='dense_ift'` for exact computation on small fields
- PySR falls back to linear regression if Julia not available
- GUDHI falls back to scipy-based persistence if not installed

## Development Environment

- **Python**: 3.12+ required
- **Virtual Environment**: `venv/` directory
- **Core Dependencies**: numpy, scipy, pandas, scikit-learn, torch, matplotlib
- **Optional Dependencies**: gudhi (TDA), pysr (symbolic regression)

## Key Commands

```bash
# Environment setup
source venv/bin/activate  # Linux/Mac
.\venv\Scripts\Activate.ps1  # Windows PowerShell

pip install -r requirements.txt
pip install -e .

# Install optional dependencies
pip install gudhi pysr

# Run tests
python -c "from mneme.models import SymbolicRegressor, create_field_vae; print('OK')"

# CLI usage
mneme generate -o sample_data.npz  # Generate synthetic data first
mneme analyze sample_data.npz --pipeline bioelectric -o results
mneme info  # Show system information
```

## Module Structure

```
src/mneme/
├── core/
│   ├── field_theory.py    # SparseGPReconstructor, DenseIFTReconstructor, etc.
│   ├── topology.py        # PersistentHomology, RipsComplex, AlphaComplex
│   └── attractors.py      # RecurrenceAnalysis, LyapunovAnalysis, ClusteringDetector
├── analysis/
│   ├── pipeline.py        # MnemePipeline, create_bioelectric_pipeline()
│   └── visualization.py   # FieldVisualizer, dashboards
├── data/
│   ├── generators.py      # SyntheticFieldGenerator, generate_planarian_bioelectric_sequence()
│   ├── preprocessors.py   # Denoiser, Normalizer, Interpolator
│   └── betse_loader.py    # BETSE simulation CSV → Mneme Field objects
├── models/
│   ├── autoencoders.py    # FieldAutoencoder (Conv VAE), create_field_vae()
│   └── symbolic.py        # SymbolicRegressor, discover_field_dynamics()
└── utils/
    ├── config.py          # Configuration management
    └── io.py              # save_results(), load_results()

scripts/
├── analyze_betse.py       # Run Mneme pipeline on BETSE simulation output
├── deep_analysis.py       # PCA, Wasserstein matrix, VAE, symbolic regression
├── analyze_physionet.py   # PhysioNet ECG/HRV analysis
└── validate_installation.py
```

## Important Implementation Notes

### Field Reconstruction
```python
from mneme.core import create_reconstructor

# Default: Sparse GP (scalable)
rec = create_reconstructor('ift', resolution=(256, 256), n_inducing=500)

# Dense IFT (exact, for small fields only)
rec = create_reconstructor('dense_ift', resolution=(32, 32))
```

### Symbolic Regression
```python
from mneme.models import discover_field_dynamics

# Discovers PDEs from field time series
result = discover_field_dynamics(field_sequence, dt=1.0, niterations=100)
# Returns: equations, best_equation, r2_score, features_used
```

### VAE Training
```python
from mneme.models import create_field_vae

vae = create_field_vae((64, 64), latent_dim=16)
result = vae.fit(train_data, val_data, epochs=100, early_stopping_patience=10)
latent = vae.encode_fields(fields)
interpolation = vae.interpolate(field_a, field_b, n_steps=10)
```

## Contributor Guidance

1. **Sparse GP is default**: Don't change IFT to use dense matrices unless explicitly requested
2. **Preserve backwards compatibility**: Old code using `method='ift'` should continue to work
3. **Optional dependencies**: Always provide fallbacks for gudhi and pysr
4. **Test thoroughly**: Run full integration test before committing
5. **Update docs**: Keep README.md, CLAUDE.md, and CHANGELOG.md in sync

## BETSE Integration

```python
from mneme.data.betse_loader import betse_to_field, load_betse_timeseries

# Load BETSE Vmem2D CSV exports into a Mneme Field object
field = betse_to_field("path/to/Vmem2D_TextExport/", resolution=(64, 64))

# Or load raw time series for custom analysis
field_sequence, metadata = load_betse_timeseries("path/to/Vmem2D_TextExport/")
# field_sequence: shape (n_timesteps, rows, cols), values in mV
```

There is also a standalone analysis script:
```bash
python scripts/analyze_betse.py path/to/Vmem2D_TextExport/ --resolution 64 --output results/betse
```

## Known Issues / TODOs

- Import order warning: import juliacall before torch to avoid potential segfault
- `lyapunov_spectrum()` (exploratory) requires >100 timesteps (hard error) and emits a `RuntimeWarning` below 1000 (recommended); short BETSE simulations (e.g. `betse try`) will fail this check
- GUDHI Wasserstein distance requires the `POT` package (`pip install POT`); bottleneck distance works without it
- Test coverage is 63.92%; target is 70%+ for JOSS submission
- `compute_basin_of_attraction()` was removed in this revision; design notes preserved in [docs/FUTURE_IDEAS.md](docs/FUTURE_IDEAS.md) for future re-implementation

## Lyapunov Spectrum Usage

```python
from mneme.core import (
    largest_lyapunov, surrogate_test, classify_attractor,
    lyapunov_spectrum, kaplan_yorke_dimension,
)

res = largest_lyapunov(trajectory, dt=0.01)            # robust λ₁ (Rosenstein 1993)
sur = surrogate_test(trajectory, statistic="lambda1", n=200, dt=0.01)
attractor_type = classify_attractor(res.lambda1, surrogate=sur)  # STRANGE only if sur.significant
spectrum = lyapunov_spectrum(trajectory, dt=0.01)      # EXPLORATORY full spectrum (RuntimeWarning)
d_ky = kaplan_yorke_dimension(spectrum)
```

> **Validation status:** Lyapunov/attractor results are **pending re-validation** under the Tier 0 corrected estimators. The previous PhysioNet headline numbers were produced by the now-removed estimator and are **not asserted**. Chaos is never reported without a passed surrogate-significance test.
