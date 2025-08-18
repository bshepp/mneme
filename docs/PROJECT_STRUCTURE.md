# Mneme Project Structure

## Directory Layout

```
mneme/
├── docs/                      # Project documentation
│   ├── PROJECT_STRUCTURE.md   # This file
│   ├── DEVELOPMENT_SETUP.md   # Setup and installation guide
│   ├── API_DESIGN.md          # Module and API documentation
│   ├── DATA_PIPELINE.md       # Data processing pipeline docs
│   └── TESTING_STRATEGY.md    # Testing approach and guidelines
│
├── src/                       # Source code
│   ├── mneme/                 # Main package
│   │   ├── __init__.py
│   │   ├── core/              # Core functionality
│   │   │   ├── __init__.py
│   │   │   ├── field_theory.py    # IFT implementations
│   │   │   ├── topology.py        # TDA algorithms
│   │   │   └── attractors.py      # Attractor detection
│   │   │
│   │   ├── models/            # ML models
│   │   │   ├── __init__.py
│   │   │   ├── autoencoders.py    # Dimensionality reduction
│   │   │   ├── symbolic.py         # Symbolic regression
│   │   │   └── field_models.py    # Field reconstruction models
│   │   │
│   │   ├── data/              # Data handling
│   │   │   ├── __init__.py
│   │   │   ├── loaders.py         # Data loading utilities
│   │   │   ├── generators.py      # Synthetic data generation
│   │   │   ├── preprocessors.py   # Data preprocessing
│   │   │   └── bioelectric.py     # Bioelectric data handling
│   │   │
│   │   ├── analysis/          # Analysis modules
│   │   │   ├── __init__.py
│   │   │   ├── pipeline.py        # Main analysis pipeline
│   │   │   ├── visualization.py   # Plotting and visualization
│   │   │   └── metrics.py         # Evaluation metrics
│   │   │
│   │   └── utils/             # Utilities
│   │       ├── __init__.py
│   │       ├── config.py          # Configuration management
│   │       ├── logging.py         # Logging setup
│   │       └── io.py              # I/O utilities
│   │
│   └── scripts/               # Executable scripts
│       ├── generate_synthetic.py   # Generate synthetic data
│       ├── run_pipeline.py         # Run full analysis pipeline
│       └── visualize_results.py    # Visualize results
│
├── notebooks/                 # Jupyter notebooks
│   ├── 01_synthetic_data_exploration.ipynb
│   ├── 02_ift_reconstruction.ipynb
│   ├── 03_topology_analysis.ipynb
│   ├── 04_symbolic_regression.ipynb
│   └── 05_bioelectric_analysis.ipynb
│
├── tests/                     # Test suite
│   ├── unit/                  # Unit tests
│   │   ├── test_field_theory.py
│   │   ├── test_topology.py
│   │   └── test_data_loaders.py
│   │
│   ├── integration/           # Integration tests
│   │   └── test_pipeline.py
│   │
│   └── fixtures/              # Test data
│       └── synthetic_test_data.npz
│
├── data/                      # Data directory
│   ├── raw/                   # Raw data (gitignored)
│   ├── processed/             # Processed data (gitignored)
│   └── synthetic/             # Generated synthetic data
│
├── experiments/               # Experiment tracking
│   ├── configs/               # Experiment configurations
│   └── results/               # Experiment results (gitignored)
│
├── requirements.txt           # Python dependencies
├── requirements-dev.txt       # Development dependencies
├── setup.py                   # Package setup
├── .gitignore                # Git ignore file
├── README.md                  # Project README
├── CLAUDE.md                  # Claude Code guidance
└── LICENSE                    # License file
```

## Module Responsibilities

### Core Modules (`src/mneme/core/`)
- **field_theory.py**: Information Field Theory implementations for continuous field reconstruction
- **topology.py**: Topological Data Analysis algorithms for persistent structure identification
- **attractors.py**: Attractor detection and characterization methods

### Model Modules (`src/mneme/models/`)
- **autoencoders.py**: VAE and autoencoder architectures for latent space analysis
- **symbolic.py**: Symbolic regression interfaces (PySR integration)
- **field_models.py**: Neural field models for spatiotemporal reconstruction

### Data Modules (`src/mneme/data/`)
- **loaders.py**: Unified data loading interfaces for different data sources
- **generators.py**: Synthetic data generation for testing and validation
- **preprocessors.py**: Normalization, filtering, and data preparation
- **bioelectric.py**: Specialized handlers for bioelectric imaging data

### Analysis Modules (`src/mneme/analysis/`)
- **pipeline.py**: Orchestrates the complete analysis workflow
- **visualization.py**: Publication-quality plotting and interactive visualizations
- **metrics.py**: Coherence metrics, validation measures, and evaluation tools

## Development Workflow

1. **Feature Development**: Create feature branches from `main`
2. **Testing**: Write tests alongside new features
3. **Documentation**: Update relevant docs with changes
4. **Experiments**: Track experiments in `experiments/` directory
5. **Notebooks**: Use notebooks for exploration, move stable code to modules

## Code Organization Principles

1. **Separation of Concerns**: Keep data, models, and analysis logic separate
2. **Modularity**: Each module should have a single, well-defined purpose
3. **Testability**: Design for easy unit and integration testing
4. **Configuration**: Use config files for experiment parameters
5. **Reproducibility**: Track random seeds, versions, and parameters