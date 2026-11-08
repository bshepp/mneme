### Changelog

All notable changes to this project will be documented in this file.

- Unreleased
  - ### Added (2026-11)
  - `studies/betse_configs/`: the BETSE paper configurations
    (`patterns_2018.yaml`, `attractors_2016_1.yaml`, with their geometry and
    network files) that the studies start from, copied from BETSE 1.5.0 with
    its BSD-2 license, so the studies reproduce without a BETSE checkout.
  - `studies/perturbation/`: protocol step 4 on the gene-network patterns.
    Both patterns returned, to within 0.01 mV, from eleven displacements of
    up to 20 mV (Anion erasure, multiplicative Anion noise); displacement
    along the line between them flips the pattern, and the midpoint is a
    third stable state. BETSE's scheduled interventions do nothing under the
    fast solver, so the saved state is displaced directly.
  - `studies/topology/`: persistent homology against the mean voltage for
    telling the two patterns apart under noise. Topology wins at 2 to 5 mV;
    both fail at 10 mV and above.
  - `studies/basins/`: eight random Anion starts reached six distinct stable
    patterns; fine-grained starts went to A or B, coarse starts mostly to
    patterns not seen before.
  - `studies/sweep/` (protocol step 5): the pattern count depends on the
    Anion's gap-junction diffusion constant (none at a tenth of the paper's
    value, one at three times it); ionic coupling a tenth of the paper's
    kept two patterns; ten times the paper's crashes BETSE.
  - `end_states.csv` in both: the final per-cell voltages of every end state.
  - `data/raw/betse_grn_2026-11/`: the full per-cell voltage trace of every
    study run as gzipped CSV (79 files, 5 MB), with `studies/pack_runs.py`.
  - Second perturbation round on patterns C to F: all survive random
    disturbances; erasure separates A, B, C (regrow) from D, E, F (move to
    new patterns D2, E2, or fall into A). The 2e-14 coupling value settled
    after a further 12,000 s into two shallow patterns. Topology separates
    all six patterns at up to 1 mV of noise where the mean voltage cannot.
  - `studies/grn_common.py`, `studies/grn_endstates.py`, `studies/run_queue.sh`.
  - The docs site publishes the studies: a `docs/hooks/studies.py` MkDocs
    hook serves every study's README, RESULTS and data files at their
    repository paths, a Studies section in the navigation lists them, and
    `docs/studies.md` summarises the six studies and how to read them.
  - ### Fixed (2026-11-08)
  - `studies/perturbation/RESULTS.md` had the outcome of the "B moved 25 %
    toward A" run inverted: it returns to B (7.3 mV away at 3,000 s, within
    0.000 mV after a further 6,000 s), it does not reach A. The basin
    boundary along the A-B line lies between 25 % and 50 % on each side.
    `project_plan.md` and `CLAUDE.md` said the same and are corrected.
    Also: D2 is 30.8 mV from E and 35.1 mV from B (not 30.8 mV from B);
    the A-B mean-voltage difference is 0.56 mV (was rounded to 0.57); the
    sweep table's count at the paper's value is now eight.
  - `studies/multistability/RESULTS.md`: the September description of
    patterns A and B (93 and 96 depolarised cells, 28 shared, about −34 and
    −59 mV) came from an unrecorded threshold and did not reproduce from the
    saved runs. Restated with the midpoint definition the later studies
    use: 118 and 122 cells, 48 shared, −38.5 against −57.5 mV. The
    distances and every tabulated number reproduced exactly.
  - `data/raw/betse_grn_2026-09/`: the per-cell voltage trace of every
    September GRN run (14 files, 0.8 MB), packed with `studies/pack_runs.py`,
    so that study can be re-checked without the run directory.
  - `studies/write_end_states.py`: writes, or with `--check` verifies, each
    study's `end_states.csv` from the packed traces, and prints the 2e-14
    continuation numbers the sweep write-up quotes; until now neither had a
    script behind it. `grn_endstates.py` and `perturbation/analyze.py` now
    refuse a run whose cell positions differ from the reference world.
  - `studies/topology/compare_statistics.py` takes `--resolution`. The
    six-pattern test rerun at 32 and 128 grid points (`results_six_res32.json`,
    `results_six_res128.json`) and the two-pattern test at 128
    (`results_two_res128.json`) show the diagrams are not converged at the
    pre-stated 64: at 128, topology separates all six patterns to 2 mV of
    noise and is well above chance at 5 mV. Reported in RESULTS.md beside the
    64-grid numbers, which stand as the pre-registered result. The design
    now states the interpolation fill, ground metric and essential-bar
    handling.
  - `assess_steady_state` warns when the run is shorter than the `sustain`
    window instead of silently checking fewer intervals.
  - `studies/perturbation/RESULTS.md`: only the runs that left their pattern
    were continued, not every unsettled run; three returning runs were
    0.02 to 0.14 mV short at 3,000 s, now said so. "Moved" figures for the
    continuations are now the state's own movement.
  - `studies/convergence/run_one.sh` takes an optional run directory and a
    `PHASES` list, and only uses a backslash log path on Windows.
  - ### Changed
  - `ExperimentConfig` uses pydantic's `ConfigDict` (was a deprecated inner
    `Config` class).
  - `project_plan.md`: the JOSS section notes the January 2026 change to the
    effort criterion.
  - ### Withdrawn (2026-09-27)
  - The BETSE analysis report is withdrawn. Its time-dependent results were
    computed on frames loaded out of time order.
  - Withdrawn PhysioNet result files are removed from `data/raw/physionet/`.
  - ### Fixed (correctness review)
  - **betse_loader**: frames are read in numeric time order. The sort key
    matched the "2" in "Vmem2D", so runs of ten or more frames loaded as
    0, 1, 10, 100, ...
  - **topology**: `sublevel` and `superlevel` were swapped. Non-square fields
    were passed to GUDHI in the wrong cell order. NaN fields now raise.
  - **topology**: the no-GUDHI fallback is now an exact union-find H0
    computation. The Wasserstein and bottleneck fallbacks are rewritten; the
    Wasserstein fallback crashed on infinite bars.
  - **attractors**: detectors no longer assign fixed-point, limit-cycle or
    strange labels from variance thresholds. They report `UNDETERMINED`.
  - **classify_attractor**: near-zero or negative estimates return
    `UNDETERMINED` unless the caller passes `oscillatory`.
  - **surrogate_test**: raises when the surrogate count cannot reach `alpha`;
    warns on multi-dimensional input and below 4000 points.
  - **pipeline**: a failed stage gives `success=False`, with the stage in
    `failed_stages`. Reconstruction is skipped, not faked, when there are no
    sparse observations.
  - **cli**: `mneme analyze` without `--config` uses the pipeline defaults
    (it previously disabled every stage) and exits non-zero on failure.
  - **field_theory**: Wiener filter response rows are normalised and
    `correlation_length` is converted from pixels.
    `optimize_hyperparameters=False` no longer optimises.
  - ### Changed
  - **BREAKING**: `classify_attractor(..., oscillatory=...)` defaults to
    `None`. `PersistentHomology(compute_cycles=True)` and
    `NeuralFieldReconstructor.uncertainty()` raise `NotImplementedError`.
    Default pipelines no longer run attractor detection.
  - Reconstructors renamed: `SubsetGPReconstructor` (`gp_subset`) and
    `WienerFilterReconstructor` (`wiener_filter`). Old names are deprecated
    aliases.
  - BETSE interpolation defaults to linear. Cubic overshoots the data range.
  - `largest_lyapunov` divergence loop is vectorised: same results to 1e-13,
    about ten times faster.
  - ### Added
  - `mneme.analysis.steady_state`: `assess_steady_state()` and
    `count_distinct_states()`.
  - `studies/convergence`: a BETSE study of one tissue started from six ion
    concentrations. Every run reached the same state.
  - `studies/multistability`: two mechanisms added to BETSE tissues. A
    voltage-coupled gene network gave two stable voltage patterns; Kir2.1
    plus a Na+ leak gave one.
  - `load_betse_cells()` for analysis without interpolation.
  - `mneme.ExperimentalWarning`, and component tiers in `docs/SCOPE.md`.
  - `docs/LYAPUNOV_OPERATING_RANGE.md`, `docs/MULTISTABILITY_PROTOCOL.md`.
  - CI installs GUDHI and POT; coverage floor raised from 35% to 60%.
  - ### Changed (Tier 0 — scientific core repair)
  - **BREAKING:** removed `compute_lyapunov_spectrum` and `classify_attractor_by_lyapunov`.
  - Added `largest_lyapunov` (Rosenstein 1993), exploratory `lyapunov_spectrum`
    (corrected Sano-Sawada), `surrogate_test` (IAAFT, two-sided rank + effect-size
    gate, shared embedding), and surrogate-gated `classify_attractor` with the new
    `AttractorType.UNDETERMINED`.
  - Added `mneme.core.embedding` (true-MI delay, Cao-1997 dimension, Theiler window).
  - Chaos / strange-attractor labels now require a passed surrogate test.
  - Discrete maps are out of scope for `largest_lyapunov` (continuous flows only).
  - Previous PhysioNet headline numbers (λ₁≈0.12, D_KY≈2.35) are withdrawn pending
    re-validation under the corrected estimators.
  - **fix(data.loaders)**: `BaseDataLoader.get_info()` now reads dtype from `Field.data` (numpy ndarray) instead of attempting `Field.dtype` (which does not exist).
  - **fix(analysis.results)**: `_create_html_report()` switched to f-string with escaped CSS braces, fixing `KeyError` on report generation; previously skipped test re-enabled.
  - **refactor(core.attractors)**: Removed unused `compute_basin_of_attraction()` placeholder. Design notes and a suggested signature for a future re-implementation are preserved in [docs/FUTURE_IDEAS.md](docs/FUTURE_IDEAS.md).
  - **feat(core.attractors)**: `compute_lyapunov_spectrum()` now emits a `RuntimeWarning` when called on trajectories shorter than `RECOMMENDED_TRAJECTORY_LENGTH` (1000 points). The hard `MIN_TRAJECTORY_LENGTH` (100 points) `ValueError` is unchanged.
  - **chore(typing)**: Re-enabled strict mypy on a curated, clean subset (`mneme.utils.logging`, `mneme.utils.config`) via `[[tool.mypy.overrides]]`. Remaining modules will be staged in as they are hardened.
  - **docs**: Consolidated duplicated root/docs files (`CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`); `docs/contributing.md` and `docs/code_of_conduct.md` are now thin pointers to the canonical root files.
  - **docs**: Refreshed CLAUDE.md (test coverage 38.4 → 63.92%, removed basin TODO, added FUTURE_IDEAS pointer).

- 2026-02-14
  - **feat(analysis)**: Deep analysis pipeline (`scripts/deep_analysis.py`)
    - PCA mode extraction with effective rank estimation
    - Cross-frame NxN Wasserstein distance matrix (H0 + H1 via GUDHI)
    - PySR symbolic regression on PCA mode dynamics (ODE discovery)
    - Spatial PDE discovery on raw field sequences via `discover_field_dynamics()`
    - Convolutional VAE latent-space embedding across all datasets
  - **feat(topology)**: Full GUDHI Wasserstein/Bottleneck distance support
    - Import from `gudhi.wasserstein` submodule (not top-level API)
    - Requires `POT` package for Wasserstein; bottleneck works without it
    - Scipy fallback preserved when GUDHI not installed
  - **test**: Expanded test suite to 113 tests (38.4% coverage)
    - New: `test_betse_loader.py` (12 tests, 91% module coverage)
    - New: Wasserstein/Bottleneck distance tests (9 tests)
    - New: `discover_field_dynamics` and `SymbolicRegressor` edge-case tests (10 tests)
  - **data**: Analyzed 4 BETSE simulation configs (attractors x2, physiology, patterns)
    - Patterns config: 824-cell elliptical tissue with GRN-driven pattern formation
    - Discovered 5-mode PCA structure (vs rank-2 for attractor/physiology configs)
    - Wasserstein total drift 84.1 mV (2.6x nearest comparator)
  - **docs**: Updated BETSE analysis report with patterns findings

- 2026-02-12
  - **feat(data)**: BETSE bioelectric simulation loader (`betse_loader.py`)
    - `load_betse_vmem_csv()`: parse single Vmem2D CSV files
    - `load_betse_timeseries()`: load multi-frame time series onto regular grid
    - `betse_to_field()`: full pipeline from BETSE CSV directory to Mneme `Field` object
    - `load_betse_exported_data()`: parse single-cell ExportedData.csv time series
    - Handles irregular cell-center data → cubic interpolation → regular grid
  - **feat(scripts)**: `scripts/analyze_betse.py` for end-to-end BETSE analysis
    - Topology analysis (persistent homology) on individual frames
    - Temporal dynamics: time-delay embedding, Lyapunov spectrum, recurrence analysis
    - Topology evolution tracking across simulation timesteps
    - JSON result export and console summary
  - **infra**: Consolidated `.gitignore` and `.dockerignore`
    - Added BETSE output patterns, Windows venv, AWS temp files
  - **infra**: Updated documentation (README, CLAUDE.md, CHANGELOG)

- 2025-11-27 (continued)
  - **feat(attractors)**: Full Lyapunov spectrum computation (Wolf algorithm)
    - `compute_lyapunov_spectrum()`: Data-driven spectrum from trajectory
    - Local Jacobian estimation via least-squares on neighbors
    - QR decomposition for orthogonalization
    - Validated against Lorenz (D_KY=2.09 vs literature 2.05)
  - **feat(attractors)**: Add `classify_attractor_by_lyapunov()` 
    - Automatic classification: fixed point, limit cycle, quasi-periodic, strange
  - **feat(attractors)**: Add `kaplan_yorke_dimension()` for fractal dimension
  - **feat(data)**: Real biological data analysis capability
    - Downloaded PhysioNet MIT-BIH ECG data for validation
    - Successfully analyzed heart rate variability (HRV)
    - Found chaotic dynamics (λ₁=+0.12/s, D_KY=2.35)
    - Predictability horizon ~8 seconds matches literature
  - **docs**: Added plain-English explanation of Lyapunov analysis
  - **deps**: Added `wfdb` for PhysioNet data access

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
