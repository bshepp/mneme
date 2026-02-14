# Mneme

An exploratory research system for detecting field-like, emergent memory structures in biological systems, with initial focus on planarian regeneration and bioelectric data.

## Overview

Mneme seeks to uncover attractor states, regulatory logic, and latent architectures not captured by sequence-based models alone. The project employs Information Field Theory (IFT), Topological Data Analysis (TDA), and machine learning to identify and model distributed memory encoding via fields in biological tissue.

## Key Features

- **Field Reconstruction**: Scalable Sparse GP reconstruction (default), with dense IFT, standard GP, and neural field backends available. Handles 256×256 fields in sub-second time.
- **Topology Analysis**: Full GUDHI integration for cubical, Rips, and Alpha complexes. Computes persistence diagrams, landscapes, and images with Wasserstein/bottleneck distances.
- **Attractor Detection**: Recurrence-based, Lyapunov, and clustering detectors for identifying stable states in temporal field data.
- **Lyapunov Spectrum**: Full Wolf algorithm implementation for computing Lyapunov exponents from trajectory data. Includes `kaplan_yorke_dimension()` for fractal dimension and automatic attractor classification.
- **Symbolic Regression**: Full PySR integration for discovering governing equations from field dynamics. Includes `discover_field_dynamics()` for automatic PDE discovery.
- **Latent Space Analysis**: Convolutional VAE (`FieldAutoencoder`) for learning compressed field representations, with training loop, interpolation, and sampling capabilities.

## Installation

```bash
# Clone repository
git clone https://github.com/bshepp/mneme.git
cd mneme

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -e .

# Install optional dependencies (recommended)
pip install gudhi pysr
```

For detailed setup instructions, see [docs/DEVELOPMENT_SETUP.md](docs/DEVELOPMENT_SETUP.md).

## Project Status: Active Development (Core Implemented)

**Recent Updates (2026-02-14):**
- ✅ Deep analysis pipeline: PCA mode extraction, cross-frame Wasserstein matrix, symbolic regression on PCA dynamics
- ✅ GUDHI integration for H0+H1 persistence computation; PySR/Julia for symbolic regression
- ✅ BETSE integration: load bioelectric tissue simulation data directly into Mneme
- ✅ 113 unit/integration tests with GitHub Actions CI (38.4% coverage)

**Previous milestones (2025-11-27):**
- ✅ Sparse GP reconstruction as scalable default (O(nm²) instead of O(n³))
- ✅ Full Lyapunov spectrum computation (Wolf algorithm) with real data validation
- ✅ Full PySR integration for symbolic regression with Julia backend
- ✅ Convolutional VAE with proper training loop and latent space utilities
- ✅ GUDHI integration for Rips, Alpha, and cubical complexes
- ✅ Dense IFT preserved as option for exact computation on small fields

### Validated on Real Biological Data

The Lyapunov spectrum implementation has been tested on real ECG data from PhysioNet:

```
Heart Rate Variability Analysis (MIT-BIH Record 100):
  λ₁ = +0.123 /s  (chaos - healthy!)
  λ₂ = -0.007 /s  (near-zero)
  λ₃ = -0.330 /s  (contraction)
  λ₄ = -0.953 /s  (contraction)
  
  Kaplan-Yorke Dimension: 2.35
  Predictability Horizon: ~8 seconds
```

This matches published literature on HRV chaos and validates the algorithm for biological time series.

## Quick Start

```python
import numpy as np
from mneme.core import FieldReconstructor, create_reconstructor
from mneme.analysis.pipeline import create_bioelectric_pipeline
from mneme.data.generators import generate_planarian_bioelectric_sequence
from mneme.models import create_field_vae, SymbolicRegressor

# Generate synthetic bioelectric data
data = generate_planarian_bioelectric_sequence(shape=(64, 64), timesteps=30, seed=42)

# Run analysis pipeline
pipe = create_bioelectric_pipeline()
result = pipe.run({'field': data})
print(f"Pipeline completed in {result.execution_time:.2f}s")

# Reconstruct field from sparse observations
positions = np.random.rand(100, 2)
observations = np.sin(4 * np.pi * positions[:, 0])
rec = create_reconstructor('ift', resolution=(128, 128))  # Uses Sparse GP
rec.fit(observations, positions)
field = rec.reconstruct()

# Train VAE on field data
vae = create_field_vae((64, 64), latent_dim=16)
vae.fit(data, epochs=50, verbose=True)  # Accepts numpy arrays directly
latent = vae.encode_fields(data)  # Shape: (30, 16)

# Discover governing equations
from mneme.models import discover_field_dynamics
result = discover_field_dynamics(data, dt=1.0, niterations=50)
print(f"Discovered equation: {result['best_equation']}")

# Compute Lyapunov spectrum (chaos analysis)
from mneme.core import compute_lyapunov_spectrum, kaplan_yorke_dimension
trajectory = latent  # Use VAE latent space as phase space
spectrum = compute_lyapunov_spectrum(trajectory, dt=1.0)
print(f"Lyapunov spectrum: {spectrum}")
print(f"Kaplan-Yorke dimension: {kaplan_yorke_dimension(spectrum):.2f}")
```

### CLI Usage

```bash
# Generate synthetic data first
mneme generate -o sample_data.npz

# Basic analysis
mneme analyze sample_data.npz --pipeline bioelectric -o results

# With Rips topology backend
mneme analyze sample_data.npz --topology-backend rips -o results

# With clustering attractor detection
mneme analyze sample_data.npz \
  --attractor-method clustering \
  --attractor-threshold 0.2 \
  -o results
```

#### Attractor CLI Flags

| Flag | Description |
|------|-------------|
| --attractor-method {none,recurrence,lyapunov,clustering} | Choose attractor detector |
| --attractor-threshold FLOAT | Detection threshold |
| --attractor-min-persistence FLOAT | Recurrence: minimum persistence fraction |
| --attractor-embedding-dim INT | Embedding dimension for 1D series |
| --attractor-time-delay INT | Time delay for embedding |
| --attractor-n-neighbors INT | Lyapunov: number of neighbors |
| --attractor-min-samples INT | Clustering: minimum samples per cluster |

## BETSE Integration

Mneme can directly ingest output from [BETSE](https://github.com/betsee/betse) (BioElectric Tissue Simulation Engine), the 2D bioelectric simulator used in Levin Lab research:

```python
from mneme.data.betse_loader import betse_to_field

# Load BETSE CSV exports into a Mneme Field object
field = betse_to_field("path/to/Vmem2D_TextExport/", resolution=(64, 64))

# Or run the standalone analysis script
# python scripts/analyze_betse.py path/to/Vmem2D_TextExport/ --resolution 64 --output results/betse
```

The loader handles:
- Irregular cell-center data → regular grid interpolation
- Multi-frame time series stacking
- Metadata extraction (spatial bounds, units, cell count)
- Single-cell `ExportedData.csv` time series

## Project Structure

```
mneme/
├── src/mneme/
│   ├── core/           # Field theory, topology, attractors
│   ├── analysis/       # Pipeline, visualization, metrics
│   ├── data/           # Generators, loaders, preprocessors, BETSE loader
│   ├── models/         # VAE, symbolic regression
│   └── utils/          # Config, logging, I/O
├── scripts/            # Analysis scripts (BETSE, PhysioNet, deep analysis)
├── notebooks/          # Demo notebooks
├── tests/              # Test suite (113 tests)
├── docs/               # Documentation
└── results/            # Analysis output (JSON, NPZ, reports)
```

## Documentation

- [Project Structure](docs/PROJECT_STRUCTURE.md) — Code organization and architecture
- [Development Setup](docs/DEVELOPMENT_SETUP.md) — Environment setup and dependencies
- [API Design](docs/API_DESIGN.md) — Module interfaces and usage
- [Data Pipeline](docs/DATA_PIPELINE.md) — Pipeline architecture and stages
- [Course](docs/course/README.md) — 11-module learning course

## Reconstruction Methods

| Method | Command | Complexity | Best For |
|--------|---------|------------|----------|
| Sparse GP | `method='ift'` (default) | O(nm²) | Large fields, production use |
| Dense IFT | `method='dense_ift'` | O(n³) | Small fields, exact computation |
| Standard GP | `method='gaussian_process'` | O(n³) | Moderate datasets |
| Neural Field | `method='neural_field'` | O(epochs) | Complex patterns |

## Core Technologies

- **Python 3.12+**: Primary development language
- **NumPy/SciPy**: Numerical computing
- **PyTorch**: Deep learning (VAE, neural fields)
- **GUDHI**: Topological data analysis
- **PySR**: Symbolic regression (Julia backend)
- **scikit-learn**: Sparse GP, clustering

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Citation

If you use Mneme in your research, please cite:
```bibtex
@software{mneme2024,
  title = {Mneme: Detecting Field-Like Memory Structures in Biological Systems},
  year = {2024},
  url = {https://github.com/bshepp/mneme}
}
```

## License

This project is licensed under the MIT License - see LICENSE file for details.

## Acknowledgments

- Inspired by work on bioelectric patterns in regeneration (Levin Lab)
- Built on theoretical foundations from Information Field Theory
- Leverages topological methods for biological data analysis
