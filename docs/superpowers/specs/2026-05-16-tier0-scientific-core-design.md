# Tier 0 — Scientific Core Repair: Design Spec

**Date:** 2026-05-16
**Status:** Approved for implementation planning
**Author:** Brian Sheppard (with Claude Code)

## 1. Problem

The flagship dynamical-systems instrument in Mneme is scientifically unreliable. Independent verification on this codebase established:

- `compute_lyapunov_spectrum` is quantitatively inaccurate on a known system: Lorenz spectrum sum returned ≈ −6.64 (true −13.67), λ₁ ≈ 0.587 (true ≈ 0.906).
- The pipeline labels **white Gaussian noise** as `STRANGE` (λ₁ ≈ +0.56, D_KY ≈ 3.0) and a **pure sine wave** as `STRANGE`. It cannot discriminate structure from noise.
- There is **no statistical-significance machinery anywhere in `src/`** (zero matches for surrogate / IAAFT / Theiler / bootstrap / p_value). Every "chaos / strange attractor" claim is unguarded.
- `_estimate_time_delay_mutual_info` uses linear autocorrelation, not mutual information (despite name and docstring); `_estimate_dimension_fnn` uses a non-standard FNN variant; 1-D input is hard-wired to `embedding_dimension=3, time_delay=1`; no Theiler window exists.

Until this is repaired, every downstream attractor/memory claim produced by the tool is suspect.

## 2. Goals

1. Replace the Lyapunov machinery with literature-standard, correctly-implemented estimators.
2. Make statistical significance (surrogate-data testing) a **mandatory gate**: the tool must never label a signal `STRANGE`/chaotic without passed surrogate evidence.
3. Provide correct embedding-parameter selection (true mutual-information delay, Cao-1997 dimension, Theiler window).
4. Prove correctness with validation tests against systems with known answers, written before the implementation (TDD).

## 3. Non-Goals (explicitly out of scope for this spec)

- Pipeline fail-soft semantics (`success=True` on swallowed-stage failure; `np.zeros` returns). Separate follow-up change.
- Renaming `SparseGPReconstructor` / "Information Field Theory".
- Topology, symbolic regression, VAE modules.
- Regenerating or numerically reconciling the committed PhysioNet `.npz` result files against real data (Tier 1 validation work).

## 4. Key Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Estimator source | **Hand-rolled, pure numpy.** No new dependencies. | Full control; no install-footprint growth; correctness proven by validation tests rather than borrowed from a library. |
| API compatibility | **Clean break.** Old public names deleted, not shimmed. | User directive: no backwards compatibility required. Removes deprecation cruft. |
| Chaos-without-evidence | New `AttractorType.UNDETERMINED`; `classify_attractor` refuses `STRANGE` without a passed `SurrogateResult`. | Scientific correctness — the central credibility fix. Independent of the compat decision. |
| Module organization | New focused modules; `attractors.py` reduced to recurrence/clustering. | Executes the engineering report's recommended split of the 1285-LOC `attractors.py`; keeps units small and independently testable. |

## 5. Module Layout

```
src/mneme/core/
  embedding.py    NEW   true-MI delay, Cao-1997 dimension, Theiler window, embed_trajectory (moved here)
  lyapunov.py     NEW   largest_lyapunov (Rosenstein 1993), lyapunov_spectrum (corrected Sano-Sawada)
  surrogates.py   NEW   iaaft_surrogates, surrogate_test
  classify.py     NEW   classify_attractor, kaplan_yorke_dimension (moved here, formula unchanged)
  attractors.py   KEEP  RecurrenceAnalysis, ClusteringDetector, AttractorDetector,
                        and LyapunovAnalysis (its method delegates to lyapunov.py).
                        Module-level compute_lyapunov_spectrum /
                        classify_attractor_by_lyapunov / the private _estimate_* /
                        _estimate_local_jacobian helpers DELETED.
```

`mneme/core/__init__.py` re-exports only the new public names. No old-name aliases.

## 6. Component Design

### 6.1 `embedding.py`

- **`mutual_information_delay(x, max_delay, *, n_bins=None) -> int`** — time delay = first local minimum of time-delayed mutual information, estimated by histogram binning (Fraser–Swinney). Default bin count via Freedman–Diaconis. Falls back to `max_delay`-bounded search; documented behavior when no local minimum exists (returns delay at global MI minimum within range).
- **`cao_embedding_dimension(x, delay, *, max_dim=10) -> int`** — Cao 1997 E1/E2 statistics (robust for short/noisy series; no arbitrary FNN ratio threshold). Returns the dimension where E1 saturates.
- **`theiler_window(x) -> int`** — default Theiler window = first zero crossing (or 1/e decay) of the autocorrelation function; used to exclude temporally-correlated neighbors in all neighbor-based estimators.
- **`embed_trajectory(series, embedding_dimension, time_delay) -> np.ndarray`** — moved verbatim from `attractors.py` (logic correct as-is); `attractors.py` imports it from here for its recurrence/clustering use.
- **`estimate_embedding_parameters(series, max_dimension=10, max_delay=100) -> (int, int)`** — moved here; internals rebuilt on the two correct estimators above.

### 6.2 `lyapunov.py`

- **`LyapunovResult`** dataclass: `lambda1: float`, `divergence_curve: np.ndarray`, `fit_region: tuple[int, int]`, `emb_dim: int`, `delay: int`, `theiler: int`, `dt: float`.
- **`largest_lyapunov(trajectory, dt=1.0, *, emb_dim=None, delay=None, theiler=None, min_separation=None) -> LyapunovResult`** — Rosenstein 1993:
  1. If 1-D, embed using `delay`/`emb_dim` (estimated via §6.1 when `None` — **no hard-coded 3/1**).
  2. For each point, nearest neighbor excluding indices within the Theiler window.
  3. Track mean log Euclidean divergence ⟨ln d(i)⟩ vs. step i.
  4. λ₁ = slope of the least-squares line over an automatically-detected linear scaling region (longest contiguous near-constant-slope segment), divided by `dt`.
  5. Returns the curve and fitted region so callers can audit the linear region.
- **`lyapunov_spectrum(trajectory, dt=1.0, *, emb_dim=None, delay=None, theiler=None) -> np.ndarray`** — corrected Sano-Sawada local-Jacobian QR method: Theiler-excluded conditioned neighbor sets, regularized least-squares Jacobian, and **consistent** growth-log accumulation vs. time normalization (the per-step-multiply / interval-log mismatch in the old code is removed). **Docstring and a `RuntimeWarning` mark this exploratory**; `largest_lyapunov` is the headline method.

### 6.3 `surrogates.py`

- **`iaaft_surrogates(x, n=200, *, max_iter=1000, seed=None) -> np.ndarray`** — Schreiber–Schmitz iterative amplitude-adjusted Fourier transform. Preserves the amplitude distribution and power spectrum (linear structure) while randomizing nonlinear structure. Returns shape `(n, len(x))`.
- **`SurrogateResult`** dataclass: `statistic_value: float`, `null_distribution: np.ndarray`, `p_value: float`, `n_surrogates: int`, `alpha: float`, `significant: bool`, `statistic_name: str`.
- **`surrogate_test(trajectory, statistic="lambda1", n=200, *, alpha=0.05, seed=None, **stat_kwargs) -> SurrogateResult`** — computes the statistic on the original and on `n` IAAFT surrogates; **one-sided rank-based p-value** = (1 + #{surrogate ≥ original}) / (n + 1); `significant = p_value < alpha`. `statistic="lambda1"` uses `largest_lyapunov`. Pluggable statistic registry so other discriminating statistics can be added later.

### 6.4 `classify.py`

- **`kaplan_yorke_dimension(spectrum) -> float`** — moved unchanged (formula was correct).
- **`classify_attractor(lambda1, *, spectrum=None, surrogate=None, zero_tol=None) -> AttractorType`**:
  - If `surrogate` is `None` **or** `not surrogate.significant`: positive λ₁ → `UNDETERMINED` (never `STRANGE`).
  - Only with a passed, significant `SurrogateResult` and λ₁ meaningfully > 0 → `STRANGE`.
  - `zero_tol` defaults to a band scaled by the surrogate spread (std of null distribution) when available, else a fixed small constant; |λ₁| within band → `LIMIT_CYCLE` (when oscillatory) / `FIXED_POINT`.
- **`AttractorType`** (in `mneme/types.py`) gains member **`UNDETERMINED`**.

### 6.5 Deletions / migrations (clean break)

- Delete `compute_lyapunov_spectrum`, `classify_attractor_by_lyapunov`, `_estimate_time_delay_mutual_info`, `_estimate_dimension_fnn`, `_estimate_local_jacobian` from `attractors.py`.
- `LyapunovAnalysis.compute_lyapunov_spectrum` (the class method at `attractors.py:512`) re-implemented to delegate to the new `lyapunov_spectrum` / `largest_lyapunov`.
- Update `mneme/core/__init__.py` exports.
- Migrate the 3 scripts to the new API: `scripts/analyze_physionet.py`, `scripts/deep_analysis.py`, `scripts/analyze_betse.py`. They must import and run without error against the new functions. (Numeric re-validation of their outputs vs. literature is Tier 1, not here.)
- Update existing `tests/test_attractors.py` to the new API (and add the new validation tests, §7).
- Update API usage snippets in `CLAUDE.md` and `README.md` to the new names. Replace the asserted headline numbers (λ₁=+0.12/s, D_KY=2.35 "matching literature") with a "pending re-validation under corrected estimators" note. Add a CHANGELOG entry.

## 7. Validation Plan (TDD — tests written before implementation)

Acceptance gates, with explicit tolerances. Lorenz and Rössler RK4 integrators added as seeded test fixtures (the Euler Lorenz fixture in `conftest.py` is replaced).

| System | Assertion |
|---|---|
| Lorenz (RK4, then 1-D x(t) delay-embedded) | `largest_lyapunov` λ₁ ∈ [0.85, 0.97]; **scale invariance**: spectrum/λ₁ for `x` and `1000*x` agree within 1% |
| Lorenz spectrum (3-D) | `lyapunov_spectrum` sum ∈ [−15, −12] (true −13.67); λ₁ ∈ [0.85, 0.97] |
| Rössler (RK4) | λ₁ ∈ [0.05, 0.10] |
| Pure sine | λ₁ within zero band; `classify_attractor` → `LIMIT_CYCLE` |
| White Gaussian noise | `surrogate_test(statistic="lambda1")` → `significant is False`; `classify_attractor` → not `STRANGE` (→ `UNDETERMINED`) |
| AR(1) correlated noise | `surrogate_test` → `significant is False` (no false chaos from linear autocorrelation) |

Unit tests:
- `iaaft_surrogates`: surrogate power spectrum and sorted-amplitude distribution match the original within tolerance; shape `(n, len)`; reproducible under fixed `seed`.
- `theiler_window`: returns expected window on a signal of known autocorrelation; neighbor sets in `largest_lyapunov` change when the window changes.
- `mutual_information_delay`: recovers the known delay on a sampled sine; returns a positive int within range.
- `cao_embedding_dimension`: returns ≈3 for Lorenz, ≈2 for a 2-D torus.
- `classify_attractor`: positive λ₁ + no surrogate → `UNDETERMINED`; positive λ₁ + significant surrogate → `STRANGE`; near-zero → `LIMIT_CYCLE`/`FIXED_POINT`.

New modules `lyapunov.py`, `surrogates.py`, `embedding.py`, `classify.py` should each land at high line coverage. A non-`slow` downsized Lorenz λ₁ test must run in default CI (the current rigorous Lorenz test is `@pytest.mark.slow` and never runs in CI).

## 8. Risks & Mitigations

- **Rosenstein linear-region detection is the classic fragile step.** Mitigation: return the divergence curve + fitted region in `LyapunovResult` so it is auditable; validate the auto-detected region against Lorenz/Rössler in tests.
- **Surrogate testing is compute-heavy** (n × λ₁ estimations). Mitigation: `n=200` default, vectorized IAAFT, seeded; keep the heavy surrogate test out of default CI (mark `slow`) but run a small-`n` smoke version in default CI.
- **Sano-Sawada spectrum may remain noisy on short biological series.** Mitigation: it is explicitly demoted to exploratory with a `RuntimeWarning`; λ₁ via Rosenstein is the headline; D_KY documented as derived from the exploratory spectrum.
- **Script migration blast radius.** Mitigation: scripts must import-and-run in tests/CI smoke (no numeric assertion here); numeric reconciliation deferred to Tier 1.

## 9. Definition of Done

- All §7 validation and unit tests written first, then passing.
- Old Lyapunov names removed; new modules in place; `__init__` updated.
- 3 scripts and `tests/` import and run against the new API.
- `CLAUDE.md`/`README.md` API snippets updated; headline numbers flagged pending re-validation; CHANGELOG entry added.
- Full existing test suite still green.
