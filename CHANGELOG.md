### Changelog

All notable changes to this project will be documented in this file.

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
