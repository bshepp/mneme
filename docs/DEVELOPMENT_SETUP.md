# Mneme Development Setup Guide

## Prerequisites

- Python 3.12.3 (tested and working)
- Git
- Virtual environment tool (venv recommended)
- CUDA-capable GPU (optional, for deep learning models)
- WSL2 environment (if on Windows)

## Initial Setup

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/mneme.git
cd mneme
```

### 2. Create Virtual Environment

```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or using conda
conda create -n mneme python=3.9
conda activate mneme
```

### 3. Install Dependencies

```bash
# Core dependencies
pip install -r requirements.txt

# Development dependencies (includes testing and linting tools)
pip install -r requirements-dev.txt

# Install package in development mode
pip install -e .
```

## Dependencies Overview

### Core Scientific Libraries
```txt
numpy>=1.21.0
scipy>=1.7.0
pandas>=1.3.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=0.24.0
```

### Deep Learning
```txt
torch>=2.0.0
torchvision>=0.15.0
```

### Specialized Tools
```txt
pysr>=0.6.0              # Symbolic regression
gudhi>=3.4.0             # Topological data analysis
nifty>=0.1.0             # Information field theory
scikit-image>=0.18.0     # Image processing
```

### Development Tools
```txt
pytest>=6.2.0
pytest-cov>=2.12.0
black>=21.6b0
flake8>=3.9.0
mypy>=0.910
jupyter>=1.0.0
ipykernel>=6.0.0
```

## Environment Configuration

### 1. Create Configuration File

Create `config/development.yaml`:

```yaml
# Development configuration
data:
  raw_path: ./data/raw
  processed_path: ./data/processed
  synthetic_path: ./data/synthetic

experiments:
  output_dir: ./experiments/results
  log_level: DEBUG
  random_seed: 42

compute:
  device: auto  # 'cuda', 'cpu', or 'auto'
  num_workers: 4
  batch_size: 32

visualization:
  backend: matplotlib
  dpi: 300
  save_format: png
```

### 2. Set Environment Variables

Create `.env` file in project root:

```bash
# Environment variables
MNEME_CONFIG_PATH=./config/development.yaml
MNEME_LOG_LEVEL=DEBUG
PYTHONPATH="${PYTHONPATH}:${PWD}/src"
```

## Verify Installation

### 1. Run Test Suite

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=mneme --cov-report=html

# Run specific test module
pytest tests/unit/test_field_theory.py
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

### 3. Run Example Script

```bash
# Generate synthetic data
python src/scripts/generate_synthetic.py --size 100 --noise 0.1

# Run basic pipeline
python src/scripts/run_pipeline.py --config config/development.yaml
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

Create `.pre-commit-config.yaml`:

```yaml
repos:
  - repo: https://github.com/psf/black
    rev: 21.6b0
    hooks:
      - id: black
  - repo: https://github.com/pycqa/flake8
    rev: 3.9.2
    hooks:
      - id: flake8
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v0.910
    hooks:
      - id: mypy
```

Install hooks:
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

1. Install CUDA Toolkit (11.3 or higher)
2. Install cuDNN
3. Install PyTorch with CUDA support:

```bash
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu113
```

### Verify GPU:

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
```

## Troubleshooting

### Common Issues:

1. **Import errors**: Ensure `PYTHONPATH` includes `src/` directory
2. **GUDHI installation**: May require C++ compiler on some systems
3. **PySR installation**: Requires Julia, follow [PySR docs](https://github.com/MilesCranmer/PySR)
4. **Memory issues**: Reduce batch size in configuration

### Getting Help:

- Check existing issues on GitHub
- Consult documentation in `docs/`
- Run tests to identify specific problems