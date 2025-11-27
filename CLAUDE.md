# CLAUDE.md

**Name:** Mneme - Biofield Memory Researcher
**Role:** Biological field memory detection specialist

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Mneme is an exploratory research system designed to detect field-like, emergent memory structures embedded in biological systems, beginning with planarian regeneration and bioelectric data. The project seeks to uncover attractor states, regulatory logic, and latent architectures not captured by sequence-based models alone.

## Current Status (2025-11-27)

The core system is **production-ready** with all major components implemented:

### Implemented Features
- **Field Reconstruction**: Sparse GP (default, scalable), Dense IFT, Standard GP, Neural Fields
- **Topology Analysis**: Full GUDHI integration (cubical, Rips, Alpha complexes)
- **Attractor Detection**: Recurrence, Lyapunov, and clustering methods
- **Symbolic Regression**: Full PySR integration with `discover_field_dynamics()`
- **Latent Space Analysis**: Convolutional VAE with training, encoding, interpolation
- **Pipeline**: End-to-end analysis with visualization and HDF5 export

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
mneme analyze data/synthetic/test_small.npz --pipeline bioelectric -o results
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
│   └── preprocessors.py   # Denoiser, Normalizer, Interpolator
├── models/
│   ├── autoencoders.py    # FieldAutoencoder (Conv VAE), create_field_vae()
│   └── symbolic.py        # SymbolicRegressor, discover_field_dynamics()
└── utils/
    ├── config.py          # Configuration management
    └── io.py              # save_results(), load_results()
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

## Known Issues / TODOs

- `compute_lyapunov_spectrum()` raises NotImplementedError (needs Wolf algorithm)
- `compute_basin_of_attraction()` raises NotImplementedError
- Attractor classification uses heuristic variance thresholds (could use Lyapunov signs)
- Import order warning: import juliacall before torch to avoid potential segfault
