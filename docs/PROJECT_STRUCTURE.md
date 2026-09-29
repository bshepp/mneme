# Mneme Project Structure

## Directory Layout

```
mneme/
├── docs/                      # Project documentation
│   ├── SCOPE.md               # Component tiers: core, frozen, experimental
│   ├── LYAPUNOV_OPERATING_RANGE.md  # Measured accuracy and limits of the frozen Lyapunov tools
│   ├── MULTISTABILITY_PROTOCOL.md   # Protocol for the multistability question
│   ├── PROJECT_STRUCTURE.md   # This file
│   ├── DEVELOPMENT_SETUP.md   # Setup and installation guide
│   ├── API_DESIGN.md          # Module and API documentation
│   ├── DATA_PIPELINE.md       # Data processing pipeline docs
│   ├── TESTING_STRATEGY.md    # Testing approach and guidelines
│   ├── FUTURE_IDEAS.md        # Deferred implementations
│   ├── api/                   # mkdocstrings reference pages
│   ├── course/                # 11-module learning course
│   ├── BETSE_ANALYSIS_REPORT.md        # Withdrawn (see CHANGELOG.md)
│   ├── REPO_AUDIT_2025-08-18.md        # Historical record
│   ├── mneme_project_plan_v1_original.md  # Historical record
│   └── superpowers/           # Historical plan and spec records
│
├── src/mneme/                 # Main package
│   ├── __init__.py
│   ├── _status.py             # ExperimentalWarning, warn_experimental()
│   ├── cli.py                 # `mneme` command line (generate, analyze, info, ...)
│   ├── types.py               # Field, PersistenceDiagram, Attractor, AttractorType, ...
│   │
│   ├── core/                  # Core functionality
│   │   ├── field_theory.py    # SubsetGPReconstructor, WienerFilterReconstructor,
│   │   │                      #   GaussianProcessReconstructor, NeuralFieldReconstructor
│   │   ├── topology.py        # PersistentHomology, RipsComplex, AlphaComplex, distances
│   │   ├── attractors.py      # Attractor detectors (experimental)
│   │   ├── embedding.py       # Delay/dimension selection (frozen)
│   │   ├── lyapunov.py        # largest_lyapunov, lyapunov_spectrum (frozen)
│   │   ├── surrogates.py      # iaaft_surrogates, surrogate_test (frozen)
│   │   └── classify.py        # classify_attractor, kaplan_yorke_dimension (frozen)
│   │
│   ├── models/                # ML models (experimental)
│   │   ├── autoencoders.py    # FieldAutoencoder (convolutional VAE)
│   │   └── symbolic.py        # SymbolicRegressor (PySR, linear fallback)
│   │
│   ├── data/                  # Data handling
│   │   ├── loaders.py         # Data loading utilities
│   │   ├── generators.py      # Synthetic data generation
│   │   ├── preprocessors.py   # Data preprocessing
│   │   ├── bioelectric.py     # Bioelectric data handling
│   │   ├── betse_loader.py    # BETSE simulation output
│   │   ├── validation.py      # Schema validation, QualityChecker (experimental)
│   │   ├── quality.py         # Thin wrapper over validation.QualityChecker
│   │   └── parallel.py        # ParallelPipeline helper
│   │
│   ├── analysis/              # Analysis modules
│   │   ├── pipeline.py        # MnemePipeline, default_config(), merge_config()
│   │   ├── steady_state.py    # assess_steady_state(), count_distinct_states()
│   │   ├── visualization.py   # Plotting and visualization
│   │   ├── features.py        # Basic field feature extraction
│   │   ├── metrics.py         # Evaluation metrics
│   │   ├── results.py         # ResultManager, reports
│   │   └── results_generator.py  # ResultGenerator facade
│   │
│   └── utils/                 # Utilities
│       ├── config.py          # Configuration management
│       ├── logging.py         # Logging setup
│       ├── io.py              # I/O utilities (HDF5, JSON, pickle)
│       └── monitoring.py      # PipelineMonitor
│
├── scripts/                   # Analysis scripts
│   ├── analyze_betse.py
│   ├── analyze_physionet.py
│   ├── deep_analysis.py
│   ├── validate_installation.py
│   └── setup_dev_env.sh
│
├── studies/                   # Recorded studies
│   └── convergence/           # BETSE convergence study (README.md, RESULTS.md, analyze.py)
│
├── review_artifacts/          # Probe scripts and outputs from the correctness review
│   └── 2026-09-26/
│
├── notebooks/                 # Jupyter notebooks
│   └── 01_synthetic_data_exploration.ipynb
│
├── tests/                     # Test suite (flat layout, one file per module)
│   ├── conftest.py            # Shared fixtures (fields, Lorenz/Rössler trajectories, ...)
│   ├── test_field_theory.py
│   ├── test_topology.py
│   ├── test_topology_correctness.py
│   ├── test_lyapunov.py
│   ├── test_surrogates.py
│   ├── test_steady_state.py
│   ├── test_pipeline.py
│   └── ...
│
├── config/                    # Example configuration files
├── data/                      # Data directory (raw/processed gitignored)
├── experiments/               # Experiment tracking
│
├── pyproject.toml             # Package metadata, dependencies, extras, tool config
├── requirements.txt           # Pinned core dependencies (mirror of pyproject.toml)
├── mkdocs.yml                 # Documentation site
├── .pre-commit-config.yaml    # Pre-commit hooks
├── CHANGELOG.md
├── CONTRIBUTING.md
├── CODE_OF_CONDUCT.md
├── README.md
├── CLAUDE.md                  # Developer context for AI assistants and contributors
└── LICENSE
```

A `venv/` directory may be present in a checkout. It is a stale Linux environment; create a fresh one (see [DEVELOPMENT_SETUP.md](DEVELOPMENT_SETUP.md)).

## Module Responsibilities

Every component belongs to a tier (core, frozen or experimental); see [SCOPE.md](SCOPE.md).

### Core Modules (`src/mneme/core/`)
- **field_theory.py**: Field reconstruction from sparse observations. `SubsetGPReconstructor` (`gp_subset`, default: a GP on a random subset of the observations), `WienerFilterReconstructor` (`wiener_filter`, dense, small grids), `GaussianProcessReconstructor` (`gaussian_process`) and the experimental `NeuralFieldReconstructor`. The old names `ift`, `sparse_gp`, `dense_ift` and their classes are deprecated aliases.
- **topology.py**: Cubical persistent homology (`sublevel` tracks pits, `superlevel` tracks peaks), Rips and Alpha complexes, Wasserstein and bottleneck distances. H1 and above need GUDHI.
- **attractors.py**: Detectors that locate dense or recurrent regions of a trajectory. They cannot classify what they find and report `AttractorType.UNDETERMINED`. Experimental.
- **embedding.py, lyapunov.py, surrogates.py, classify.py**: Largest Lyapunov exponent, surrogate significance test and gated classification. Frozen; see [LYAPUNOV_OPERATING_RANGE.md](LYAPUNOV_OPERATING_RANGE.md).

### Model Modules (`src/mneme/models/`)
- **autoencoders.py**: `FieldAutoencoder`, a convolutional VAE. Experimental.
- **symbolic.py**: `SymbolicRegressor` and `discover_field_dynamics()` on PySR, with a linear-regression fallback. Experimental.

### Data Modules (`src/mneme/data/`)
- **loaders.py**: Unified data loading interfaces for different data sources
- **generators.py**: Synthetic data generation for testing
- **preprocessors.py**: Normalization, filtering, and data preparation
- **bioelectric.py**: Specialized handlers for bioelectric imaging data
- **betse_loader.py**: BETSE output; `load_betse_cells()` is preferred, `betse_to_field()` interpolates to a grid
- **validation.py**: Schema validation and the `QualityChecker` (experimental: thresholds not calibrated)

### Analysis Modules (`src/mneme/analysis/`)
- **pipeline.py**: Orchestrates the workflow. Default pipelines run preprocessing, reconstruction and topology; attractor detection is opt-in. `PipelineResult.success` is False when any stage fails.
- **steady_state.py**: Has a relaxing run settled, and how many distinct end states do several runs reach
- **visualization.py**: Plotting utilities
- **features.py**: Basic feature extractor
- **metrics.py**: Evaluation utilities

## Development Workflow

1. **Feature Development**: Create feature branches from `main`
2. **Testing**: Write tests alongside new features; a component is core only when a test checks it against an independently known answer
3. **Documentation**: Update relevant docs with changes
4. **Experiments**: Track experiments in `experiments/`; recorded studies go in `studies/`
5. **Notebooks**: Use notebooks for exploration, move stable code to modules

## Code Organization Principles

1. **Separation of Concerns**: Keep data, models, and analysis logic separate
2. **Modularity**: Each module should have a single, well-defined purpose
3. **Testability**: Design for easy unit and integration testing
4. **Configuration**: Use config files for experiment parameters
5. **Reproducibility**: Track random seeds, versions, and parameters
