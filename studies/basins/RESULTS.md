# Basin mapping: results

**Run:** 2026-11-05, BETSE 1.5.0, same seeded world and parameters as [studies/perturbation](../perturbation/RESULTS.md) (254 cells). **Numbers:** `results.json`, from `analyze.py`. **Design:** [README.md](README.md).

## Findings in one table

| Start: random Anion field, correlation length | Seed | Voltage spread at start (mV) | End state | Distance to A / B / C (mV) | Settled |
|---|---|---|---|---|---|
| 25 µm (about one cell) | 1 | 16.6 | **A** | 0.00 / 35.04 / 28.58 | yes |
| 25 µm | 2 | 19.5 | **A** | 0.00 / 35.04 / 28.58 | yes |
| 25 µm | 3 | 20.0 | **B** | 35.06 / 0.63 / 35.67 | no (rate still falling) |
| 25 µm | 4 | 21.0 | **B** | 35.05 / 0.22 / 35.65 | no (1.2 mV to go by the fit) |
| 100 µm (about five cells) | 1 | 20.3 | **new, D** | 32.55 / 32.85 / 33.74 | yes |
| 100 µm | 2 | 15.3 | **new, E** | 32.29 / 35.58 / 32.66 | yes |
| 100 µm | 3 | 18.6 | **B** | 35.04 / 0.00 / 35.65 | yes |
| 100 µm | 4 | 22.0 | **new, F** | 24.64 / 35.54 / 28.38 | yes |

Same-state threshold 1.0 mV, largest per-cell difference; settling criteria as in the earlier studies. A and B are the perturbation study's reference states; C is its third state, reached from the midpoint of the A–B line.

**Eight random starts reached six different stable patterns.** Fine-grained starts (25 µm) went to A or B, two each. Coarse starts (100 µm) went once to B and three times to patterns not seen before. D, E and F are 30.5 to 33.9 mV apart from each other and 24.6 to 35.6 mV from A, B and C, so they are as distinct from one another as A is from B.

## The new patterns

| | A | B | C | D | E | F |
|---|---|---|---|---|---|---|
| Depolarised cells | 111 | 115 | 106 | 119 | 112 | 112 |
| Depolarised cells shared with A | — | 22 | 77 | 43 | 55 | 97 |
| … with B | 22 | — | 26 | 84 | 25 | 28 |
| … with C | 77 | 26 | — | 50 | 72 | 88 |
| Depolarised / polarised voltage (mV) | −37.9 / −57.1 | −38.5 / −58.1 | −37.2 / −56.6 | −38.2 / −57.0 | −37.6 / −56.6 | −37.5 / −56.7 |
| Mean voltage (mV) | −48.67 | −49.24 | −48.50 | −48.20 | −48.22 | −48.26 |

Every pattern depolarises 106 to 119 of the 254 cells, to the same two voltage levels, in a different arrangement. The means lie within 1.0 mV of each other. F shares 97 of its 112 depolarised cells with A yet is 24.6 mV from it in the worst cell: the patterns are not nested variants with a few cells swapped but distinct arrangements with partial overlap.

## Which start leads where

| Correlation length of the start | Outcomes |
|---|---|
| 25 µm, four seeds | A, A, B, B |
| 100 µm, four seeds | D, E, B, F |

A start with structure finer than the pattern's own regions (about five cells across) was smoothed away and the tissue chose A or B, the two patterns the earlier gradient and near-uniform starts also reached. A start with structure on the pattern's own scale left its imprint: three of four coarse starts settled into patterns that match none seen before. The relaxation times of the coarse starts (220 to 620 s) were no longer than the fine ones (about 400 s), so the new patterns were reached as directly as A and B.

## What this does and does not show

- **The tissue has many stable voltage patterns under one parameter set**, at least six, and the random starts found three in four tries, so the count is not close to exhausted. Every pattern found is reached by relaxation and, where tested (A, B, C), survives displacement.
- **Which pattern a start reaches depends on its spatial scale as well as its direction.** This is a finer statement of the earlier one that history decides the pattern.
- **Baselines.** Replicate noise is 0 mV in this deterministic model; the smallest distance between any two of the six patterns is 24.6 mV; the largest remaining drift of any settled run is 0.011 mV. The two unsettled runs (25 µm, seeds 3 and 4) were 0.63 and 0.22 mV from B and approaching it; B's own slow relaxation (τ about 1,400 s) is the reason, as in the perturbation study.
- **Not shown:** the basin sizes of D, E and F, their stability under perturbation, whether any natural (gradient-like) start reaches them, or anything about a different tissue, geometry or parameter set. Four seeds per correlation length is enough to show the effect of scale, not to estimate how often each pattern is reached.
