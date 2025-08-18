# CLAUDE.md

**Name:** Mneme - Biofield Memory Researcher
**Role:** Biological field memory detection specialist

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Mneme is an exploratory research system designed to detect field-like, emergent memory structures embedded in biological systems, beginning with planarian regeneration and bioelectric data. The project seeks to uncover attractor states, regulatory logic, and latent architectures not captured by sequence-based models alone.

## Direction and Scope (MVP-first)

We will ship a minimal, reproducible end-to-end pipeline before expanding features. The MVP analyzes 2D bioelectric-like fields (synthetic or small real samples), reconstructs a continuous field, computes cubical persistence (via GUDHI with a simple fallback), detects simple attractor signals, and saves plots/metrics. Everything else is roadmap.

### MVP Deliverables
- CLI: `mneme analyze data/synthetic/planarian_demo.npz -o results/` produces figures and metrics
- Notebook: one demo notebook that runs in <5 minutes CPU-only
- Docs: README Quickstart that reproduces results; docs truthfully reflect implemented modules
- CI: lint + tests green on Python 3.12 (mypy non-blocking initially)

### Roadmap (post-MVP)
- Field models (neural fields), Lyapunov metrics, richer topology (Rips/Alpha), symbolic regression (PySR), and 3D support
- Type hygiene pass; re-enable strict mypy

## Development Environment

- **Python Environment**: Uses a virtual environment in `venv/`
- **Primary Tools**: Python, Jupyter/Colab, NumPy, SciPy, PyTorch/Keras, PySR, GUDHI (for TDA)
- **Focus Areas**: Information Field Theory (IFT), dimensionality reduction, Topological Data Analysis (TDA), symbolic regression

## Key Technical Approaches (current)

Current MVP includes: simple reconstruction (IFT or interpolation), cubical persistence (GUDHI or fallback), recurrence-based attractor detection, and visualization.

## Development Commands

**Environment Setup:**
```bash
# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

**Testing and Validation:**
```bash
# Test core imports
python -c "import mneme; print('Package working!')"

# Run basic functionality tests
python -c "import numpy as np; import pandas as pd; import matplotlib.pyplot as plt; print('Core packages working!')"
```

**Package Management:**
- Python 3.12.3 in virtual environment
- Core packages: numpy, scipy, pandas, scikit-learn, matplotlib, seaborn
- Deep learning: torch, torchvision (for neural network components)
- Specialized tools: pysr, gudhi (when fully installed)

## Current Status (2025-08-18)

- Package imports cleanly on Python 3.12
- CI: lint passes; mypy runs non-blocking; docs deploy to `gh-pages`
- `create_bioelectric_pipeline()` stub available; minimal model placeholders added

## Contributor Guidance

1) Prioritize MVP features that make the end-to-end run better before adding new modules
2) Keep CI green (add tests with each feature). Mypy is non-blocking until type hygiene pass, but don’t add new untyped public APIs
3) If adding “future” features, hide them behind flags with clear errors and update docs accordingly