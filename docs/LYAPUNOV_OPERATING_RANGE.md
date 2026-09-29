# Lyapunov Operating Range

**Status:** frozen. The estimators are kept as they are and are not under development.
**Measured:** 2026-09-26, against `main` at `e3225bd`. The vectorised estimator introduced afterwards reproduces these numbers to 1e-13.

This page records what the Lyapunov tools were measured to do, so that a number from them can be judged.

## Summary

| Question | Answer |
|---|---|
| Does the surrogate test call noise chaotic? | No. 0 false positives in 7 non-chaotic signals. |
| How long must a series be for the test to detect chaos? | About 4,000 points. It missed Lorenz and Rössler at 3,000 and below. |
| How accurate is λ₁? | Within 2% at the conditions it was tuned on. Between 14% and 81% off elsewhere. |
| Does `fit_r2` show whether an estimate is reliable? | No. It was at least 0.997 in every case below, right or wrong. |
| Does a passed test prove chaos? | No. It shows the series is not linear noise. |

## How the constants were chosen

The scaling-region detector has constants (`_SAT_LEVEL`, `_FLAT_LEN`, `_MS_MIN`, `_MS_MAX`) that were tuned until the test fixtures passed. The fixtures are Lorenz and Rössler series of about 6,000 points. Three of the constants are absolute sample counts, so they carry the fixtures' sampling rate with them. No system was held out.

## Accuracy of `largest_lyapunov`

6,000 points, embedding parameters estimated automatically.

| Signal | True λ₁ | Estimate | Error |
|---|---|---|---|
| Lorenz x, dt = 0.01 (tuning condition) | 0.906 | 0.921 | +2% |
| Lorenz x, dt = 0.02 | 0.906 | 0.927 | +2% |
| Lorenz x, dt = 0.05 | 0.906 | 0.902 | 0% |
| Lorenz x, dt = 0.005 | 0.906 | 1.231 | +36% |
| Lorenz z, dt = 0.01 | 0.906 | 1.638 | +81% |
| Lorenz full 3-D state | 0.906 | 1.061 | +17% |
| Lorenz x + 5% measurement noise | 0.906 | 0.834 | −8% |
| Rössler x, dt = 0.05 | 0.071 | 0.057 | −19% |
| Rössler x, dt = 0.1 | 0.071 | 0.052 | −27% |
| Rössler x, dt = 0.2 | 0.071 | 0.034 | −52% |
| Chen x, dt = 0.002 (held out) | ≈ 2.0 | 3.080 | ≈ +54% |
| Chen x, dt = 0.005 (held out) | ≈ 2.0 | 1.725 | ≈ −14% |

Signals that are not chaotic:

| Signal | Estimate |
|---|---|
| Sine, with 0%, 1% and 20% noise | 0.000 to 0.001 |
| Two-frequency torus | 0.001 |
| Van der Pol limit cycle | 0.004 |
| White noise | 0.002 |
| AR(1), φ = 0.95 | 0.007 |
| AR(2) narrowband noise | 0.011 |
| Random walk | 0.002 |
| Van der Pol with dynamical noise | 0.091 |

The last row matters: a noisy limit cycle produces a clearly positive estimate with R² = 0.997. Only the surrogate test stops it being read as chaos.

## Surrogate test: false positives

39 surrogates, 3,000 points.

| Signal | λ₁ | Null mean ± sd | Effect | Significant |
|---|---|---|---|---|
| White noise | 0.002 | 0.002 ± 0.000 | +1.01 | no |
| AR(1) | 0.008 | 0.009 ± 0.003 | −0.38 | no |
| AR(2) narrowband | 0.009 | 0.011 ± 0.004 | −0.43 | no |
| Sine + 5% noise | 0.001 | 0.001 ± 0.000 | −0.18 | no |
| Van der Pol + dynamical noise | 0.094 | 0.109 ± 0.049 | −0.30 | no |
| Threshold AR (nonlinear, stochastic) | 0.003 | 0.003 ± 0.000 | +1.66 | no |
| Non-stationary AR(1) | 0.009 | 0.012 ± 0.003 | −0.99 | no |

## Surrogate test: detection of chaos against length

39 surrogates.

| Signal | Points | λ₁ (true) | Null mean ± sd | Effect | Significant |
|---|---|---|---|---|---|
| Lorenz x, start A | 1,000 | 1.394 (0.906) | 1.221 ± 0.422 | +0.41 | no |
| Lorenz x, start A | 3,000 | 1.146 (0.906) | 1.383 ± 0.680 | −0.35 | no |
| Lorenz x, start A | 4,000 | 0.667 (0.906) | 2.010 ± 0.304 | −4.42 | yes |
| Lorenz x, start B | 4,000 | 0.734 (0.906) | 2.401 ± 0.071 | −23.53 | yes |
| Lorenz x, start B | 6,000 | 0.712 (0.906) | 2.455 ± 0.053 | −32.63 | yes |
| Lorenz x, start C | 8,000 | 0.873 (0.906) | 2.535 ± 0.050 | −33.26 | yes |
| Rössler x | 3,000 | 0.095 (0.071) | 0.125 ± 0.038 | −0.79 | no |
| Rössler x | 8,000 | 0.066 (0.071) | 0.164 ± 0.004 | −25.69 | yes |

Detection works through the lower tail. The chaotic series has a λ₁ well below that of its surrogates, so the test is detecting "more predictable than linear noise".

## How to use the tools

1. **Use at least 4,000 points.** Below that, "not significant" tells you nothing. `surrogate_test` warns when the series is shorter.
2. **Use at least 39 surrogates** at α = 0.05. `surrogate_test` raises an error below that, because significance would be unreachable.
3. **Report λ₁ with its limits.** Expect errors of tens of percent unless your sampling resembles the tuning conditions.
4. **Read `STRANGE` as "consistent with chaos".** Non-stationarity and nonlinear stochastic dynamics can also pass the test.
5. **Do not use these tools on discrete maps.** The estimator rejects single-step divergence, which is where a map puts all of its divergence.
6. **Do not use them on transients.** A system relaxing to rest has no sustained divergence to measure.

## Reproducing the measurements

The scripts are in `review_artifacts/2026-09-26/`. Run them from the repo root with `PYTHONPATH=src`.
