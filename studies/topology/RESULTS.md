# Topology against mean voltage: results

**Run:** 2026-11-05, on the settled patterns A and B of [studies/perturbation](../perturbation/RESULTS.md) (BETSE 1.5.0, 254 cells). **Numbers:** `results.json`, from `compare_statistics.py`. **Design:** [README.md](README.md).

## The two patterns

| | A | B |
|---|---|---|
| Mean voltage (mV) | −48.67 | −49.24 |
| Depolarised cells (above the midpoint) | 111 of 254 | 115 of 254 |
| Depolarised / polarised voltage (mV) | −37.9 / −57.1 | −38.5 / −58.1 |
| Long-lived H0 features of the superlevel filtration, persistence in mV | ∞, 31.3, 28.2 | ∞, 30.5, 28.0, 27.3 |
| Longest H1 feature (mV) | 5.1 | 3.6 |

The means differ by 0.56 mV. The patterns differ by 35.0 mV in the worst cell and 18.7 mV root-mean-square over cells. In the superlevel filtration (peaks first), A's depolarised region has three components that persist for about 30 mV before merging; B's has four. All other features persist for less than 8 mV.

## Telling a noisy sample's pattern

Each noisy sample (Gaussian noise of standard deviation σ added to every cell) is assigned to the nearer reference under each statistic. 40 draws per pattern per σ. Accuracy: fraction assigned to the pattern it came from (chance 0.5). Separation: the A-to-B distance divided by the median distance of a noisy sample from its own reference.

| σ (mV) | mean: accuracy | mean: separation | topology (raw): accuracy | separation | topology (≥ 5 mV): accuracy | separation | per-cell RMS: accuracy | separation |
|---|---|---|---|---|---|---|---|---|
| 0.5 | 1.000 | 24.6 | 1.000 | 6.6 | 1.000 | 11.3 | 1.000 | 37.7 |
| 1 | 1.000 | 14.3 | 1.000 | 3.7 | 1.000 | 7.3 | 1.000 | 18.8 |
| 2 | 0.950 | 7.4 | 1.000 | 1.9 | 1.000 | 3.0 | 1.000 | 9.4 |
| 5 | 0.875 | 3.6 | 1.000 | 0.7 | 1.000 | 0.7 | 1.000 | 3.8 |
| 10 | 0.700 | 1.5 | 0.613 | 0.3 | 0.600 | 0.3 | 1.000 | 1.9 |
| 20 | 0.575 | 0.7 | 0.425 | 0.1 | 0.425 | 0.1 | 1.000 | 0.9 |

The 2 mV threshold variant is within 0.025 of the raw one at every σ and is in `results.json`.

A-to-B distances: mean 0.56 mV; topology 18.6 mV (raw), 18.1 mV (≥ 2 mV), 17.1 mV (≥ 5 mV); per-cell RMS 18.7 mV.

## What this shows

- **At 2 to 5 mV of noise, topology separates the patterns and the mean voltage does not reliably.** Topology stays at 100% while the mean falls to 95% and then 87.5%. The two patterns have nearly the same mean, so the mean's 0.56 mV margin is lost once the noise on the mean (σ/√254, about 0.3 mV at σ = 5) approaches it. The topological distance between the patterns is 30 times larger than their mean difference.
- **At 10 mV and above, both fail, and topology fails first.** Noise of that size creates many new features with persistence of its own order, and the Wasserstein distance counts them: the median distance of a noisy sample from its own pattern grows from 2.8 mV at σ = 0.5 to 59 mV at σ = 10, past the 18.6 mV that separates the patterns. Dropping features below 5 mV does not help there because the noise features are larger than that.
- **The per-cell comparison beats both at every σ**, as it must: it knows which cell is which. Topology is the better summary statistic only where the cell identities are unavailable or not comparable, which is the situation the project cares about (two tissues, or one tissue imaged twice without registration).

## What this does not show

- One pair of patterns, one tissue, one noise model (independent per-cell Gaussian). Spatially correlated noise, or a different pair of patterns, could order the statistics differently.
- No smoothing was applied before the persistence computation. A denoising step would likely extend topology's range toward higher σ, but it was not part of the pre-stated design and is not reported.
- The grid resolution is a free parameter of the design, fixed at 64 × 64 before the runs. The same draws on a 128 × 128 grid (`results_two_res128.json`, added 2026-11-08) give an A-to-B topology distance of 24.2 mV instead of 18.6, the same accuracies at every σ up to 5 mV, and 0.688 (raw) at σ = 10 against the mean's 0.700, and 0.475 at σ = 20 against 0.575. "Both fail at 10 mV and topology first" holds at both resolutions, by a smaller margin at the finer one. See [Grid resolution](#grid-resolution-added-2026-11-08) below.
- The field is interpolated linearly inside the tissue's convex hull and takes the nearest cell's value outside it; the tissue is a diagonal strip, so 77 % of the grid points are filled that way (see the design). References and samples share the path, so the comparison is fair, but the diagrams are those of that interpolant, not of the cell values alone.
- The separation ratios at σ = 5 (0.7) say the patterns are closer in topology than the noise moves a sample, yet the accuracy is 100%: the two distances move together under noise, so the nearer reference is still the right one. The accuracy is the result; the separation ratio is a description of the margin.

## Six patterns (added 2026-11-06)

The same test with all six patterns found by then as references: A and B, C from the perturbation study, and D, E and F from the basin study (`../basins/end_states.csv`). Chance is one in six. **Numbers:** `results_six.json`, from `compare_statistics.py --end-states ../basins/end_states.csv`.

### Distances between patterns

| Statistic | Smallest pair distance | Which pair | Largest pair distance |
|---|---|---|---|
| mean voltage | 0.013 mV | D vs E | 1.03 mV (B vs D) |
| topology, raw | 5.6 mV | E vs F | 18.6 mV (A vs B) |
| topology, ≥ 5 mV | 2.9 mV | E vs F | 17.1 mV (A vs B) |
| per-cell RMS | 5.0 mV | A vs F | 18.7 mV (A vs B) |
| worst cell | 24.6 mV | A vs F | 35.6 mV (B vs C) |

D, E and F have means within 0.06 mV of each other. Topologically, A, B and C are far from everything (15 to 19 mV, apart from A–D, A–E, A–F and B–C at 6 to 8 mV), while D, E and F sit 5.6 to 8.7 mV from each other.

### Accuracy

| σ (mV) | mean | topology (raw) | topology (≥ 2 mV) | topology (≥ 5 mV) | per-cell RMS |
|---|---|---|---|---|---|
| 0.5 | 0.771 | 1.000 | 1.000 | 0.838 | 1.000 |
| 1 | 0.683 | 0.996 | 0.988 | 0.817 | 1.000 |
| 2 | 0.579 | 0.767 | 0.750 | 0.621 | 1.000 |
| 5 | 0.408 | 0.388 | 0.408 | 0.412 | 1.000 |
| 10 | 0.275 | 0.267 | 0.275 | 0.242 | 1.000 |
| 20 | 0.175 | 0.188 | 0.212 | 0.200 | 1.000 |

### What this adds

- **Topology separates all six patterns at up to 1 mV of noise, where the mean voltage already fails.** The mean is at 77 % with almost no noise, because three of the six patterns have the same mean to 0.06 mV. Topology is at 100 %.
- **Topology's range shrinks with more patterns, at this grid resolution.** With two patterns it held to 5 mV of noise; with six, on the pre-stated 64 × 64 grid, it is at 77 % by 2 mV and near chance by 5 mV. The hard pairs are E–F and D–E, 5.6 and 8.7 mV apart topologically, against 18.6 mV for A–B. A statistic's useful range is set by its closest pair. On a finer grid the closest pairs move apart and the range extends; see the next section.
- **Dropping short bars hurts here.** The 5 mV threshold removes real structure that distinguishes D, E and F (E's third-longest finite H0 feature, 4.2 mV, is dropped; F's, at 5.2 mV, is just kept, and D's is 10.3 mV, so the threshold's cost is not fully explained by that one feature), so it is worse than the raw distance at every σ below 5 mV. The 2 mV threshold is indistinguishable from raw. The threshold was fixed before these patterns existed, and this is what came of it.
- **The per-cell comparison stays at 100 % to 20 mV**, as before.

## Grid resolution (added 2026-11-08)

A review of the script found that the persistence diagrams are not converged at the pre-stated 64 × 64 grid (grid step 8.4 µm against a median cell spacing of 16.9 µm): the number of bars a noisy field produces keeps growing with the grid. The six-pattern test was therefore rerun with the same seed and draws at 32 × 32 and 128 × 128 (`compare_statistics.py --resolution`; `results_six_res32.json`, `results_six_res128.json`). The mean-voltage and per-cell statistics do not depend on the grid.

| Grid | A–B topology distance (mV) | Smallest topology pair (mV) | Raw topology accuracy at σ = 0.5 / 1 / 2 / 5 / 10 mV |
|---|---|---|---|
| 32 × 32 | 16.7 | E–F, 3.7 | 1.000 / 0.971 / 0.713 / 0.417 / 0.263 |
| **64 × 64 (pre-stated)** | 18.6 | E–F, 5.6 | 1.000 / 0.996 / 0.767 / 0.388 / 0.267 |
| 128 × 128 | 24.2 | A–F, 10.6 | 1.000 / 1.000 / 0.992 / 0.546 / 0.204 |

Chance is 0.167; the mean voltage is at 0.771 / 0.683 / 0.579 / 0.408 / 0.275 at every grid.

- **The distances between patterns grow with the grid**, and the closest pair changes: E–F at 32 and 64, A–F at 128. The reference diagrams gain long bars as the interpolant resolves more of each pattern's structure.
- **Topology's range with six patterns is set by the grid, not by the method.** At 128 × 128 it separates all six patterns at up to 2 mV of noise (99 %) and is well above chance at 5 mV (55 %), where the 64 × 64 result was 77 % and 39 %. The statement above that topology is "near chance by 5 mV" with six patterns describes the 64 × 64 discretisation.
- **At 10 mV and above, every grid fails**, and the finer grid fails harder (0.20 at 128 against 0.27 at 64), because noise produces more bars on a finer grid and the Wasserstein distance counts them.
- **The two-pattern headline is stable**: at 128 × 128 topology is still 100 % at 2 and 5 mV, and still fails at 10 mV (0.688 against the mean's 0.700).
- **What this does not settle:** the resolution at which the diagrams converge (128 is not shown to be it), whether smoothing would do what a finer grid does, and whether the design should fix the grid by cell spacing rather than by a count. The 64 × 64 numbers stand as the pre-registered result; the resolution dependence is reported beside them so the result is not read as a property of persistent homology in general.
