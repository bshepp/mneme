# Scope and Support Status

Mneme's components are sorted into three tiers. The tier says how much you can rely on a component's output.

| Tier | Meaning |
|---|---|
| **Core** | Tested against known answers. Supported. |
| **Frozen** | Works within a documented operating range. Not under development. |
| **Experimental** | Not validated against known answers. Do not use its output as evidence for a scientific claim. |

Experimental components emit `mneme.ExperimentalWarning` when constructed. The default pipelines run core stages only.

## Core

| Component | Where | What it is tested against |
|---|---|---|
| BETSE loading | `mneme.data.betse_loader` | Frame order, cell counts and value ranges of synthetic exports |
| Cubical persistent homology | `mneme.core.topology.PersistentHomology` | Closed-form fields, and an independent union-find computation of H0 |
| Wasserstein and bottleneck distances | `mneme.core.topology` | Closed-form diagram pairs, and GUDHI |
| Subset GP reconstruction | `mneme.core.field_theory.SubsetGPReconstructor` | A known field; uncertainty coverage |
| Standard GP reconstruction | `mneme.core.field_theory.GaussianProcessReconstructor` | A known field |
| Wiener-filter reconstruction | `mneme.core.field_theory.WienerFilterReconstructor` | A known field |
| Steady-state and distinct-state analysis | `mneme.analysis.steady_state` | Exponential relaxations with known time constants; a double-well system with two known stable states |
| Preprocessing and I/O | `mneme.data.preprocessors`, `mneme.utils.io` | Round trips and unit tests |

### Known limits of core components

- **Persistence needs GUDHI for H1 and above.** Without it, only H0 is computed and the higher diagrams are returned empty, with a `RuntimeWarning`.
- **Subset GP discards data.** When there are more observations than `n_subset`, the rest are not used. `n_discarded_` reports how many.
- **Interpolated BETSE grids contain values that are not simulation output.** Grid points outside the convex hull of the cells are nearest-neighbour fill. The loader returns an `inside_hull` mask. Prefer `load_betse_cells()` when a regular grid is not required.

## Frozen

| Component | Where |
|---|---|
| Largest Lyapunov exponent | `mneme.core.lyapunov.largest_lyapunov` |
| Surrogate significance test | `mneme.core.surrogates.surrogate_test` |
| Gated classification | `mneme.core.classify.classify_attractor` |
| Lyapunov spectrum and Kaplan-Yorke dimension | `mneme.core.lyapunov.lyapunov_spectrum`, `mneme.core.classify.kaplan_yorke_dimension` |
| Embedding parameter selection | `mneme.core.embedding` |

These are kept as they are. Their measured accuracy and limits are in [Lyapunov Operating Range](LYAPUNOV_OPERATING_RANGE.md).

## Experimental

| Component | Where | Why it is experimental |
|---|---|---|
| Attractor detectors | `mneme.core.attractors` | They locate dense or recurrent regions. They cannot say what kind of attractor a region is, and report `UNDETERMINED`. |
| Symbolic regression | `mneme.models.symbolic` | Never tested on a system with known governing equations. |
| Variational autoencoder | `mneme.models.autoencoders` | No test that the latent space recovers known structure. |
| Neural field reconstruction | `mneme.core.field_theory.NeuralFieldReconstructor` | No accuracy test and no uncertainty estimate. |
| Data quality checker | `mneme.data.validation.QualityChecker` | Verdicts come from fixed thresholds that have not been calibrated. |

## Moving a component between tiers

A component moves to core when it has a test that checks its output against an answer known independently of the code, and that test runs in CI.
