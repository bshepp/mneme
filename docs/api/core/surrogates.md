# Surrogates

IAAFT surrogate data and the rank + effect-size significance gate.

Frozen tier. `surrogate_test` raises when the surrogate count cannot reach `alpha` (at least 39 at alpha 0.05) and warns below about 4,000 points, where the test was measured to lack power. See the [Lyapunov Operating Range](../../LYAPUNOV_OPERATING_RANGE.md).

## IAAFT Surrogates

::: mneme.core.surrogates.iaaft_surrogates

## Result Type

::: mneme.core.surrogates.SurrogateResult

## Significance Test

::: mneme.core.surrogates.surrogate_test
