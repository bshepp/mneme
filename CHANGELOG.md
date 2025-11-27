### Changelog

All notable changes to this project will be documented in this file.

- 2025-11-27
  - **feat(reconstruction)**: Add `SparseGPReconstructor` as scalable default for IFT
    - O(nm²) complexity instead of O(n³), handles 256×256 fields in <1s
    - Automatic inducing point selection with configurable count
    - Full uncertainty quantification preserved
  - **feat(reconstruction)**: Preserve dense IFT as `DenseIFTReconstructor` / `method='dense_ift'`
    - Warns when resolution > 64×64 due to memory requirements
    - Useful for exact computation on small fields
  - **feat(reconstruction)**: Add `create_reconstructor()` factory function
  - **feat(models)**: Full PySR integration in `SymbolicRegressor`
    - Proper `fit()`, `predict()`, `score()` interface
    - `get_equations()`, `get_best_equation()`, `latex()` for equation extraction
    - `get_sympy_expression()` for symbolic manipulation
    - Falls back to linear regression when PySR not installed
  - **feat(models)**: Add `discover_field_dynamics()` for automatic PDE discovery
    - Extracts features (u, ∇²u, ∂u/∂x, ∂u/∂y) from field time series
    - Returns best equation, all equations, R² score
  - **feat(models)**: Full convolutional VAE implementation in `FieldAutoencoder`
    - 4-layer encoder/decoder with BatchNorm and LeakyReLU
    - Proper reparameterization trick and β-VAE support
    - `fit()` method with early stopping and LR scheduling
    - `encode_fields()`, `decode_latent()`, `reconstruct()`, `sample()`, `interpolate()`
  - **feat(models)**: Add `create_field_vae()` convenience function
  - **refactor(core)**: Update `FieldReconstructor` to dispatch to appropriate backend
  - **refactor(core)**: `IFTReconstructor` is now alias for `SparseGPReconstructor`
  - **docs**: Update README.md with new features and examples
  - **docs**: Update CLAUDE.md with current architecture and guidelines
  - **test**: Full integration test suite (10/10 passing)

- 2025-08-30
  - feat(io): HDF5 loader returns `AnalysisResult`; support `.h5` and `.hdf5`
  - fix(pipeline): preprocessing accepts `numpy.ndarray` fields gracefully
  - chore(cli): import `juliacall` before `torch` in `mneme info`; clean PySR status
  - feat(extras): add `[pysr]` optional extra (pins scikit-learn/juliacall)
  - docs(course): add 11-module course; expand Module 11; update Modules 3/8 to new loader
  - docs: various syncs and run logs across modules

- 2025-08-18
  - Align docs to MVP-first; add repo audit
  - Fix package import (visualization, metrics datetime)
  - Update CI (Ubuntu, py3.11/3.12; docs to gh-pages)
  - Update pyproject (py3.12, URLs, scripts; mypy 3.12)
  - Add `create_bioelectric_pipeline()` stub
  - Add placeholder models (`FieldAutoencoder`, `SymbolicRegressor`)
  - Add minimal tests and docs deploy
