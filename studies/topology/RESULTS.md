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

The means differ by 0.57 mV. The patterns differ by 35.0 mV in the worst cell and 18.7 mV root-mean-square over cells. In the superlevel filtration (peaks first), A's depolarised region has three components that persist for about 30 mV before merging; B's has four. All other features persist for less than 8 mV.

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

The 2 mV threshold variant is within 0.02 of the raw one at every σ and is in `results.json`.

A-to-B distances: mean 0.57 mV; topology 18.6 mV (raw), 18.1 mV (≥ 2 mV), 17.1 mV (≥ 5 mV); per-cell RMS 18.7 mV.

## What this shows

- **At 2 to 5 mV of noise, topology separates the patterns and the mean voltage does not reliably.** Topology stays at 100% while the mean falls to 95% and then 87.5%. The two patterns have nearly the same mean, so the mean's 0.57 mV margin is lost once the noise on the mean (σ/√254, about 0.3 mV at σ = 5) approaches it. The topological distance between the patterns is 30 times larger than their mean difference.
- **At 10 mV and above, both fail, and topology fails first.** Noise of that size creates many new features with persistence of its own order, and the Wasserstein distance counts them: the median distance of a noisy sample from its own pattern grows from 2.8 mV at σ = 0.5 to 59 mV at σ = 10, past the 18.6 mV that separates the patterns. Dropping features below 5 mV does not help there because the noise features are larger than that.
- **The per-cell comparison beats both at every σ**, as it must: it knows which cell is which. Topology is the better summary statistic only where the cell identities are unavailable or not comparable, which is the situation the project cares about (two tissues, or one tissue imaged twice without registration).

## What this does not show

- One pair of patterns, one tissue, one noise model (independent per-cell Gaussian). Spatially correlated noise, or a different pair of patterns, could order the statistics differently.
- No smoothing was applied before the persistence computation. A denoising step would likely extend topology's range toward higher σ, but it was not part of the pre-stated design and is not reported.
- The separation ratios at σ = 5 (0.7) say the patterns are closer in topology than the noise moves a sample, yet the accuracy is 100%: the two distances move together under noise, so the nearer reference is still the right one. The accuracy is the result; the separation ratio is a description of the margin.
