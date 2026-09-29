# BETSE Simulation Analysis Report (withdrawn)

**Withdrawn:** 2026-09-27

This report, first published 2026-02-13, has been withdrawn. Its findings should not be cited or relied on.

## Why

A review in September 2026 found defects that invalidate the report's conclusions:

- **Frames were analysed out of time order.** The BETSE loader sorted exported frames incorrectly, so every time-dependent result (recurrence, Wasserstein time series, topology timelines, Lyapunov values, symbolic regression, and first/middle/last comparisons) was computed on a scrambled sequence.
- **The Lyapunov estimator used was unreliable.** It has since been replaced, and all results it produced are withdrawn.
- **No claim was tested against a null model or baseline.**

## What happens next

The loader and analysis code are being corrected, and the simulations will be re-analysed. A new report will be published only for results that hold up under null-model testing.

The original text remains available in the repository history.
