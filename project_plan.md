# Mneme Project Plan v3

**Last updated:** 2026-09-27
**Previous versions:** v2 is in git history (`project_plan.md` before 2026-09-27). v1 is [docs/mneme_project_plan_v1_original.md](docs/mneme_project_plan_v1_original.md).

---

## Why v3

A review in September 2026 found that no result the project had published could be supported:

- The BETSE loader read frames out of time order, so every time-dependent result in the BETSE report was an artifact.
- The Lyapunov estimator behind the PhysioNet numbers was unreliable. Those numbers had already been withdrawn in May.
- The analysis pipeline reported success when stages failed, and labelled a sine wave and white noise as strange attractors.

Both reports are withdrawn. The defects are fixed. v3 re-orders the work so that claims follow validation.

## Purpose

Mneme studies whether biological tissue stores pattern memory in its bioelectric field, starting with simulated tissue.

## The hypothesis, and what would test it

**Hypothesis:** tissue stores pattern memory as attractors of its bioelectric field.

**What that predicts:** multistability. The same tissue settles into different stable voltage patterns depending on its history, and returns to them after a disturbance.

**What does not test it:** Lyapunov exponents. They measure how fast nearby trajectories separate on a chaotic attractor. The hypothesis does not predict chaos, and the BETSE runs analysed so far are relaxations toward rest.

## Rules for claims

1. A result is reported only with a null model or baseline beside it.
2. A component's output supports a claim only if the component is in the core tier. See [docs/SCOPE.md](docs/SCOPE.md).
3. A method is used only on data inside its measured operating range.
4. One run per condition demonstrates nothing. Conditions are replicated.

---

## Status

| Stage | Status |
|---|---|
| 0. Withdraw unsupported claims | Done. Pending merge of the withdrawal to the public site. |
| 1. Correctness fixes with regression tests | Done |
| 2. Measure and document the Lyapunov operating range; freeze | Done |
| 3. Multistability experiment | First study done: one stable state found in the published configuration. See [studies/convergence/RESULTS.md](studies/convergence/RESULTS.md). |
| 4. Rewrite the BETSE report from new results | Done for the first study, as its RESULTS.md. |
| 5. JOSS submission | Not ready. See below. |

## Stage 3: Multistability experiment

Protocol: [docs/MULTISTABILITY_PROTOCOL.md](docs/MULTISTABILITY_PROTOCOL.md).

| Step | What | Done when |
|---|---|---|
| 3.1 | Run one tissue to steady state | Frame-to-frame change stays below tolerance |
| 3.2 | Start the same tissue from many initial conditions | End states are collected and counted |
| 3.3 | Perturb a settled state | Return, or failure to return, is measured |
| 3.4 | Sweep gap-junction conductance | The number of end states is tracked across the sweep |

The original runs cannot stand in for this. They differ in cell count and in a fixed parameter, and were not run to steady state.

**First study (2026-09-27):** steps 3.1 to 3.3's counting were run on the published 2016 configuration, with and without coupling. Every run reached the same state. That configuration has no mechanism expected to give multistability, so the next study needs one: voltage-gated channels, or a gene regulatory network coupled to voltage. Steps 3.3 (perturbation) and 3.4 (sweep) are still to do.

## Stage 5: JOSS

JOSS revised its criteria in January 2026. Check the current wording before planning around this table.

| Criterion | Mneme today | Needed |
|---|---|---|
| Six months of public development history | Commits since July 2025 | Confirm the repo was public throughout |
| Releases | None | Tag a release once stage 1 is merged |
| Public issues and pull requests | 0 issues, 2 pull requests | Track work in issues |
| Evidence of research use | None | At least one reproducible analysis that holds up |
| AI usage disclosure | Not written | Write it; development has been AI-assisted |
| Validation | Core components tested against known answers | A validation section built from stage 3 |

Submission should wait for stage 4.

## Deferred

| Item | Why |
|---|---|
| Self-collected ECG (AD8232) | Whether healthy heart rate is chaotic is a contested question that better data has not settled. Recordings would also be too short for the surrogate test. |
| Outreach to labs | Waits for a result that holds up. |
| Theory development | Waits for stage 4. |
| Coverage target | Replaced by the rule that core components are tested against known answers. |

## Open questions

1. Does simulated tissue show more than one stable voltage pattern under the same parameters?
2. If so, how large a disturbance does each pattern survive?
3. Does the number of stable patterns change as gap-junction coupling changes?
4. Can persistent homology of the voltage field tell the stable patterns apart better than the mean voltage can?

Question 4 is where Mneme's topology code earns its place, and it has a built-in baseline.
