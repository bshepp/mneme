# Mneme Project Plan v3

**Last updated:** 2026-11-07
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
| 3. Multistability experiment | Steps 3.1 to 3.4 done. The published 2016 configuration has one state ([studies/convergence](studies/convergence/RESULTS.md)). Adding a voltage-coupled gene network gives two stable patterns; adding Kir2.1 plus a Na⁺ leak does not ([studies/multistability](studies/multistability/RESULTS.md)). Both gene-network patterns return from every displacement of up to 20 mV that is not aimed at the other pattern; displacement along the line between them flips the pattern, and the midpoint settles into a third stable state. Random Anion starts reach six distinct stable patterns in eight tries ([studies/basins](studies/basins/RESULTS.md)); a second perturbation round brings the count to eight. The pattern count depends on the Anion's gap-junction coupling: none at a tenth of the paper's value, two at 2e-14, one at three times the paper's ([studies/sweep](studies/sweep/RESULTS.md)). |
| 4. Rewrite the BETSE report from new results | Done for each study, as its RESULTS.md file. |
| 5. JOSS submission | Not ready. See below. |

## Stage 3: Multistability experiment

Protocol: [docs/MULTISTABILITY_PROTOCOL.md](docs/MULTISTABILITY_PROTOCOL.md).

| Step | What | Done when |
|---|---|---|
| 3.1 | Run one tissue to steady state | Frame-to-frame change stays below tolerance |
| 3.2 | Start the same tissue from many initial conditions | End states are collected and counted |
| 3.3 | Perturb a settled state | Return, or failure to return, is measured |
| 3.4 | Sweep gap-junction coupling | The number of end states is tracked across the sweep |

The original runs cannot stand in for this. They differ in cell count and in a fixed parameter, and were not run to steady state.

**First study (2026-09-27):** the published 2016 configuration, with and without coupling. Every run reached the same state.

**Second study (2026-09-29):** two mechanisms added. A gene network coupled to voltage (the 2018 patterns configuration) gave two distinct stable voltage patterns under one parameter set, each reached from more than one start, with the uniform state an unstable saddle. Kir2.1 plus a Na⁺ leak gave one state at all three leak strengths tried. Steps 3.3 (perturbing a patterned state) and 3.4 (sweep) are still to do, on the gene-network system.

**Third study (2026-11-05), step 3.3:** [studies/perturbation](studies/perturbation/RESULTS.md). A new seeded world of the gene-network tissue (254 cells) reached patterns A and B from the same two starts. Each pattern returned, to within 0.01 mV, from eleven displacements of up to 20 mV: partial and complete erasure of its depolarised region's Anion, and multiplicative Anion noise of up to 17-fold in single cells. Moving the Anion field along the line from one pattern to the other is what flips it: 25 % of the way returns to the starting pattern from either side (slowly from B), the midpoint settles into a third stable state, C, that held for 9,000 s, and 75 % of the way flips. A voltage-only displacement leaves no trace by the first sample. BETSE's scheduled interventions do nothing under the fast solver, so the saved state was displaced directly.

**Topology against mean voltage (2026-11-05), open question 4:** [studies/topology](studies/topology/RESULTS.md). The two patterns' means differ by 0.56 mV; their persistence diagrams differ by 18.6 mV (2-Wasserstein). Under 2 to 5 mV of per-cell noise, topology assigns a noisy sample to the right pattern 100 % of the time and the mean 95 % falling to 87.5 %. At 10 mV and above both fail, topology first. The per-cell comparison, which knows cell identities, beats both at every noise level.

**Basins (2026-11-05):** [studies/basins](studies/basins/RESULTS.md). Eight random Anion starts reached six distinct stable patterns. Fine-grained starts (25 µm) went to A or B; coarse starts (100 µm) went to B once and to three patterns not seen before, each 25 to 36 mV from every other.

**Second perturbation round (2026-11-06 to 07):** patterns C to F all survive random Anion disturbances of up to 16 mV. Erasing the depolarised region separates them: A, B and C regrow it; D and E move to neighbouring new patterns (D2, E2); F falls into A. Eight stable patterns are now known under the one parameter set.

**Sweep (2026-11-05 to 07), step 3.4:** [studies/sweep](studies/sweep/RESULTS.md). The Anion's gap-junction diffusion constant Dgj decides the pattern count: at a tenth of the paper's value both starts stay uniform; at three times it both reach the same single pattern; at the paper's value there are at least eight; at 2e-14 there are two, shallow ones, reached slowly. A tenfold weaker ionic coupling kept two patterns. The tenfold stronger one crashed BETSE.

## Stage 5: JOSS

JOSS revised its criteria in January 2026 (blog post "Preparing JOSS for a generative AI future", 2026-01). The fixed "three months of work" benchmark is gone. Editors now weigh evidence of scholarly contribution: a development history over months or years, contributions from more than one developer, evolution through community feedback, and research impact. Codebases generated largely by AI prompting do not by themselves meet the bar. The review-criteria page could not be reached from the development environment on 2026-11-05, so the exact checklist wording below is from the search summary and should be confirmed against https://joss.readthedocs.io/en/latest/review_criteria.html before submitting.

| Criterion | Mneme today | Needed |
|---|---|---|
| Development history (no fixed length since 2026-01; very new projects get extra scrutiny) | Commits since July 2025, one developer | Confirm the repo was public throughout; contributions from a second person would help |
| Releases | None | Tag a release once stage 1 is merged |
| Public issues and pull requests | 0 issues, 2 pull requests | Track work in issues |
| Evidence of research use | None | At least one reproducible analysis that holds up |
| AI usage disclosure | Not written | Write it; development has been AI-assisted, and JOSS now says AI-generated code alone is not a scholarly contribution |
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

1. Does simulated tissue show more than one stable voltage pattern under the same parameters? *Yes, for the gene-network tissue: at least eight, under one parameter set (studies/multistability, perturbation, basins).*
2. If so, how large a disturbance does each pattern survive? *All six patterns tested survive random Anion disturbances of up to 16 mV. Erasing a pattern's depolarised region (18 to 20 mV) is survived by A, B and C and not by D, E and F. Along the A–B line, a quarter of the way returns from either side and the midpoint is a third pattern (studies/perturbation).*
3. Does the number of stable patterns change as gap-junction coupling changes? *Yes, with the Anion's gap-junction diffusion: none at a tenth of the paper's value, two shallow ones at 2e-14, at least eight at the paper's 5e-14, one at three times it. Ionic coupling a tenth of the paper's kept two (studies/sweep).*
4. Can persistent homology of the voltage field tell the stable patterns apart better than the mean voltage can? *Yes, under 2 to 5 mV of noise; neither works at 10 mV and above. With six patterns, how far topology's advantage reaches depends on the interpolation grid: to 1 mV at the pre-stated 64 × 64, to 2 mV and beyond at 128 × 128 (studies/topology).*

Question 4 is where Mneme's topology code earns its place, and it has a built-in baseline.
