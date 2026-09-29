# Mneme Development Setup Guide

## Prerequisites

- Python 3.12 or later
- Git
- Virtual environment tool (venv recommended)
- CUDA-capable GPU (optional, for deep learning models)

Windows, Linux and macOS all work natively. WSL2 is not required.

## Initial Setup

### 1. Clone the Repository

```bash
git clone https://github.com/bshepp/mneme.git
cd mneme
```

### 2. Create Virtual Environment

The `venv/` directory that may already be present in a checkout is a stale Linux environment. Do not activate it; create a fresh one.

```bash
# Using venv
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Or using conda
conda create -n mneme python=3.12
conda activate mneme
```

### 3. Install Dependencies

```bash
# Package in development mode, with test/lint tools and the TDA extras
# (GUDHI and POT). This is what CI installs.
pip install -e ".[dev,tda]"

# Optional: symbolic regression (needs Julia)
pip install -e ".[pysr]"

# Optional: notebooks, docs
pip install -e ".[notebooks]"
pip install -e ".[docs]"
```

Core dependencies are declared in `pyproject.toml`. `requirements.txt` pins the same core set for reproducible installs and is not required when installing with `pip install -e`.

## Dependencies Overview

### Core (from `pyproject.toml`)
```txt
numpy>=1.24.0
scipy>=1.10.0
pandas>=2.0.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
plotly>=5.0.0
torch>=2.0.0
h5py>=3.0.0
scikit-image>=0.20.0
pyyaml>=6.0
tqdm>=4.60.0
click>=8.0.0
pydantic>=2.0.0
```

### Optional Extras
```txt
tda:    gudhi>=3.4.0, POT>=0.9.0      # persistent homology above H0, Wasserstein distance
pysr:   pysr, juliacall               # symbolic regression (experimental; needs Julia)
dev:    pytest, pytest-cov, pytest-mock, black, flake8, isort, mypy, pre-commit
docs:   mkdocs, mkdocs-material, mkdocstrings[python]
notebooks: jupyter, jupyterlab, ipykernel
```

Without GUDHI, persistence is computed for H0 only and a `RuntimeWarning` says so.

## Environment Configuration

### 1. Configuration Files

The pipeline's configuration keys are the ones returned by `mneme.analysis.pipeline.default_config('standard' | 'bioelectric')`: top-level `preprocessing`, `reconstruction`, `topology` and, to opt in to the experimental detectors, `attractors`. A YAML file passed as `mneme --config file.yaml analyze ...` is overlaid on those defaults with `merge_config()`.

```yaml
# Example override file
reconstruction:
  method: gp_subset
  resolution: [128, 128]

topology:
  max_dimension: 1
  filtration: sublevel
  persistence_threshold: 0.05
```

`config/default.yaml` and `config/experiment_example.yaml` are older, broader files; keys in them that the pipeline does not read are ignored.

### 2. Environment Variables

`mneme.utils.config.Config.from_env()` reads variables with the `MNEME_` prefix. No environment variable is required for a normal install: once the package is installed with `pip install -e .`, `PYTHONPATH` does not need to include `src/`.

## Verify Installation

### 1. Run Test Suite

```bash
# Run all tests (372 tests, about 5 minutes)
pytest

# Run with coverage (about 70%; CI fails below 60%)
pytest --cov=src/mneme --cov-report=html

# Run specific test module
pytest tests/test_field_theory.py
```

### 2. Check Imports

```python
# In Python interpreter or notebook
import mneme
from mneme.core import field_theory
from mneme.data import generators
from mneme.models import autoencoders

print(f"Mneme version: {mneme.__version__}")
```

Or run `python scripts/validate_installation.py`.

### 3. Run Example Commands

```bash
# Generate synthetic data
mneme generate -o sample_data.npz

# Run the default pipeline (prints each stage's status; exits non-zero if a stage fails)
mneme analyze sample_data.npz --pipeline bioelectric -o results

# Show system information
mneme info
```

## Development Tools Setup

### 1. Code Formatting

```bash
# Format code with black
black src/ tests/

# Check without modifying
black --check src/ tests/
```

### 2. Linting

```bash
# Run flake8
flake8 src/ tests/

# Run mypy for type checking
mypy src/
```

### 3. Pre-commit Hooks

The repository ships a `.pre-commit-config.yaml` (trailing-whitespace and end-of-file fixers, YAML and merge-conflict checks, black, isort, flake8). Install the hooks with:

```bash
pip install pre-commit
pre-commit install
```

## Jupyter Notebook Setup

```bash
# Install kernel for virtual environment
python -m ipykernel install --user --name mneme --display-name "Mneme"

# Start Jupyter
jupyter notebook

# Or JupyterLab
jupyter lab
```

## GPU Setup (Optional)

### For NVIDIA GPUs:

1. Install a CUDA Toolkit supported by your PyTorch version
2. Install cuDNN
3. Install PyTorch with CUDA support following the selector at https://pytorch.org/get-started/locally/

### Verify GPU:

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
```

## Troubleshooting

### Common Issues:

1. **Import errors**: Ensure the package is installed (`pip install -e .`) into the active environment, not the stale `venv/`
2. **GUDHI installation**: `pip install -e ".[tda]"`; may require a C++ compiler on some systems
3. **PySR installation**: Requires Julia, follow [PySR docs](https://github.com/MilesCranmer/PySR); import `juliacall` before `torch` to avoid a possible segfault
4. **Memory issues**: Reduce the reconstruction resolution or `n_subset` in the configuration

### Getting Help:

- Check existing issues on GitHub
- Consult documentation in `docs/`
- Run tests to identify specific problems
