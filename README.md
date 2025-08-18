# Mneme

An exploratory research system for detecting field-like, emergent memory structures in biological systems, with initial focus on planarian regeneration and bioelectric data.

## Overview

Mneme seeks to uncover attractor states, regulatory logic, and latent architectures not captured by sequence-based models alone. The project employs Information Field Theory (IFT), Topological Data Analysis (TDA), and machine learning to identify and model distributed memory encoding via fields in biological tissue.

## Key Features

- **Field Reconstruction**: Information Field Theory implementations for continuous field interpolation
- **Topology Analysis**: Persistent homology for identifying stable structures
- **Attractor Detection**: Methods for finding and characterizing dynamical attractors
- **Symbolic Regression**: Discovery of mathematical rules governing field behavior
- **Latent Space Analysis**: Autoencoders for dimensionality reduction and pattern discovery

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
```

For detailed setup instructions, see [docs/DEVELOPMENT_SETUP.md](docs/DEVELOPMENT_SETUP.md).

## Project Status: Phase 1 Development Ready ✅

**Recent Updates (2025-01-04):**
- ✅ Fixed critical import issues preventing package usage
- ✅ Updated Pydantic v2 compatibility in type system
- ✅ Resolved syntax errors in analysis pipeline
- ✅ Core modules now import successfully for development

## Quick Start

```python
import mneme  # Package imports successfully
from mneme.core import field_theory, topology
from mneme.analysis import pipeline
from mneme.data import generators

# Generate synthetic field data
generator = generators.SyntheticFieldGenerator(seed=42)
field = generator.generate_dynamic(shape=(64, 64), timesteps=10)

# Create analysis pipeline
config = pipeline.create_standard_pipeline()
results = config.run({'field': field})

# Access results
print("Pipeline executed successfully!")
```

## Project Structure

```
mneme/
├── src/mneme/         # Core library code
├── notebooks/         # Jupyter notebooks for exploration
├── tests/            # Test suite
├── docs/             # Documentation
├── data/             # Data directory (gitignored)
└── experiments/      # Experiment tracking
```

See [docs/PROJECT_STRUCTURE.md](docs/PROJECT_STRUCTURE.md) for detailed structure.

## Documentation

- [Project Structure](docs/PROJECT_STRUCTURE.md) - Code organization and architecture
- [Development Setup](docs/DEVELOPMENT_SETUP.md) - Environment setup and dependencies
- [API Design](docs/API_DESIGN.md) - Module interfaces and usage
- [Data Pipeline](docs/DATA_PIPELINE.md) - Data processing workflows
- [Testing Strategy](docs/TESTING_STRATEGY.md) - Testing approach and guidelines
- [Contributing](CONTRIBUTING.md) - How to contribute

## Research Phases

1. **Phase 1**: Synthetic data prototyping with planarian-inspired fields
2. **Phase 2**: Real bioelectric and gene expression data analysis
3. **Phase 3**: Theory development and cross-organism validation

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
