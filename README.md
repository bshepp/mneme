# Mneme

An exploratory research system for detecting field-like, emergent memory structures in biological systems, with initial focus on planarian regeneration and bioelectric data.

## Overview

Mneme seeks to uncover attractor states, regulatory logic, and latent architectures not captured by sequence-based models alone. The project employs Information Field Theory (IFT), Topological Data Analysis (TDA), and machine learning to identify and model distributed memory encoding via fields in biological tissue.

## Key Features

- **Field Reconstruction (MVP-ready)**: Basic IFT and GP reconstruction APIs; identity fallback when sparse observations are not provided
- **Topology Analysis (MVP-ready)**: Cubical persistence via GUDHI when installed; simple fallback otherwise
- **Attractor Detection (experimental)**: Recurrence-based detector usable on temporal data; Lyapunov and clustering detectors have basic MVP implementations (some advanced methods like full spectra/basin estimation remain TODO)
- **Symbolic Regression (placeholder)**: PySR is installed optionally; shipped `SymbolicRegressor` is a placeholder. Integrations are roadmap
- **Latent Space Analysis (placeholder)**: `FieldAutoencoder` class is a minimal placeholder; not a production model

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

# Optional extras
# Symbolic regression (PySR) with compatible pins
pip install -e .[pysr]
```

For detailed setup instructions, see [docs/DEVELOPMENT_SETUP.md](docs/DEVELOPMENT_SETUP.md).

## Project Status: Phase 1 Development Ready ✅

**Recent Updates (2025-01-04):**
- ✅ Fixed critical import issues preventing package usage
- ✅ Updated Pydantic v2 compatibility in type system
- ✅ Resolved syntax errors in analysis pipeline
- ✅ Core modules now import successfully for development

## Quick Start (MVP)

```python
import mneme  # Package imports successfully
from mneme.core import field_theory, topology
from mneme.analysis import pipeline
from mneme.data import generators

# Generate synthetic field data (CPU-friendly)
generator = generators.SyntheticFieldGenerator(seed=42)
field = generator.generate_dynamic(shape=(64, 64), timesteps=10, parameters={'noise_level': 0.1})

# Create analysis pipeline (lightweight defaults)
pipe = pipeline.create_bioelectric_pipeline()
results = pipe.run({'field': field})

# Access results
print("Pipeline executed successfully!")
```

### CLI usage

Run analysis on a saved array and choose a topology backend:

```bash
# Topology backend options: cubical (default), rips, alpha
mneme analyze data/synthetic/test_small.npz \
  --pipeline bioelectric \
  --topology-backend rips \
  -o results
```

Run analysis with clustering-based attractor detection (example):

```bash
mneme analyze data/synthetic/test_small.npz \
  --pipeline bioelectric \
  --attractor-method clustering \
  --attractor-threshold 0.2 \
  --attractor-min-samples 20 \
  -o results_clustering
```

#### Attractor CLI flags

| Flag | Description |
|------|-------------|
| --attractor-method {none,recurrence,lyapunov,clustering} | Choose attractor detector (use none to disable) |
| --attractor-threshold FLOAT | Detection threshold (method-specific) |
| --attractor-min-persistence FLOAT | Recurrence: minimum persistence fraction |
| --attractor-embedding-dim INT | Recurrence/Clustering: embedding dimension for 1D series |
| --attractor-time-delay INT | Recurrence/Clustering: time delay for embedding |
| --attractor-n-neighbors INT | Lyapunov: number of neighbors |
| --attractor-evolution-time INT | Lyapunov: evolution time steps |
| --attractor-min-samples INT | Clustering: minimum samples per cluster |
| --attractor-clustering-method {dbscan,kmeans} | Clustering: algorithm selection |

## Project Structure (MVP)

```
mneme/
├── src/mneme/         # Core library code
├── notebooks/         # One demo notebook
├── tests/             # Minimal smoke tests
├── docs/             # Documentation
├── data/             # Data directory (gitignored)
└── experiments/      # Experiment tracking
```

See [docs/PROJECT_STRUCTURE.md](docs/PROJECT_STRUCTURE.md) for detailed structure.

## Documentation

- [Project Structure](docs/PROJECT_STRUCTURE.md) — Code organization and architecture
- [Development Setup](docs/DEVELOPMENT_SETUP.md) — Environment setup and dependencies
- [API Design](docs/API_DESIGN.md) — Module interfaces and usage
- [Data Pipeline](docs/DATA_PIPELINE.md) — MVP note: several sections are illustrative/roadmap and not yet implemented (quality, features, parallel, monitoring, recovery)
- [Testing Strategy](docs/TESTING_STRATEGY.md) — Current suite is a smoke test; more tests are roadmap

### Capabilities vs Roadmap

- MVP capabilities: importable package, CLI (`mneme info`, `mneme analyze`), lightweight preprocessing, identity or basic reconstruction, cubical persistence (with GUDHI), simple recurrence attractor detection on temporal inputs, plotting utilities
- Roadmap (not fully implemented): rich loaders/quality/feature extractors; Lyapunov/clustering attractors; real autoencoders; symbolic regression integration; parallel/distributed pipeline; monitoring/recovery utilities
- [Contributing](CONTRIBUTING.md) - How to contribute

## Roadmap

- Add robust reconstruction methods and validation
- Expand topology features and distances
- Attractor characterization beyond recurrence
- Optional models (autoencoders, symbolic regression)

## Core Technologies

- **Python 3.12**: Primary development language (tested)
- **NumPy/SciPy**: Numerical computing
- **PyTorch**: Deep learning models
- **GUDHI**: Topological data analysis
- **PySR**: Symbolic regression
- **Jupyter**: Interactive analysis

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

- Inspired by work on bioelectric patterns in regeneration
- Built on theoretical foundations from Information Field Theory
- Leverages topological methods for biological data analysis# mneme
