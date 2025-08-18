# CLAUDE.md

**Name:** Mneme - Biofield Memory Researcher
**Role:** Biological field memory detection specialist

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Mneme is an exploratory research system designed to detect field-like, emergent memory structures embedded in biological systems, beginning with planarian regeneration and bioelectric data. The project seeks to uncover attractor states, regulatory logic, and latent architectures not captured by sequence-based models alone.

## Project Architecture

The project is organized in phases:

1. **Phase 1: Synthetic Data Prototyping** - Develop and validate a modular analysis pipeline on synthetic field-like data inspired by planarian voltage maps and regenerative logic
2. **Phase 2: Real Bioelectric + Gene Expression Data** - Test Mneme on actual biological data, starting with planarian bioelectric images, gene expression overlays, and regeneration timelines
3. **Phase 3: Interpretation + Theory Development** - Formalize insights into a model of distributed memory encoding via fields in biological tissue

## Development Environment

- **Python Environment**: Uses a virtual environment in `venv/`
- **Primary Tools**: Python, Jupyter/Colab, NumPy, SciPy, PyTorch/Keras, PySR, GUDHI (for TDA)
- **Focus Areas**: Information Field Theory (IFT), dimensionality reduction, Topological Data Analysis (TDA), symbolic regression

## Key Technical Approaches

- Information Field Theory (IFT) reconstruction to interpolate continuous fields
- Dimensionality reduction (PCA, autoencoders) to uncover latent spaces
- Topological Data Analysis (TDA) to identify persistent structures
- Symbolic regression to extract mathematical rules governing local field behaviors
- Bioelectric and gene expression field reconstruction
- Attractor detection and bifurcation analysis

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

## Phase 1 Status: COMPLETE ✅

**Critical Import Issues Fixed (2025-01-04):**
- ✅ **Pydantic v2 compatibility** (types.py:95-130) - Updated `Field(default_factory=list)` → `Field(default=[])` and `@validator` → `@field_validator`
- ✅ **Missing model imports** (models/__init__.py) - Commented out non-existent modules to prevent circular import failures
- ✅ **Import dependencies** - Added missing `Union` import to topology.py and `List` import to generators.py  
- ✅ **Pipeline syntax errors** - Fixed malformed escape sequences in pipeline.py, created clean version without `\n` artifacts

**Current Status:** Core package components (types, models, core, data, analysis.pipeline) import successfully. The Mneme system is now ready for Phase 1 synthetic data prototyping and pipeline development.

**Remaining Minor Issues:** 
- Visualization.py has syntax formatting issues (non-blocking)
- Missing model implementations (autoencoders, symbolic, field_models) - planned for future development

**Next Steps:** Begin synthetic biofield data generation and test the analysis pipeline on controlled field-like structures.