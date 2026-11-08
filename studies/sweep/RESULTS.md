# Coupling sweep: results

**Run:** 2026-11-05 to 06, BETSE 1.5.0, same seeded world as [studies/perturbation](../perturbation/RESULTS.md) (254 cells). **Numbers:** `results.json`, from `analyze.py`; the final per-cell voltages of every run are in `end_states.csv`. **Design:** [README.md](README.md).

Each value was run from the two starting conditions that reach patterns A and B at the paper's values: the bitmap Anion gradient ("A start") and a uniform Anion plus a 1 % x gradient ("B start"). 6,000 s init phase; settling criteria and the 1.0 mV same-state threshold as in the earlier studies.

## Sweep 1: the Anion's gap-junction diffusion constant, Dgj

| Dgj (m²/s) | A start → | B start → | Distinct end states | Settled | Depolarised / polarised voltage (mV) |
|---|---|---|---|---|---|
| 5e-15 | uniform, −37.18 mV | uniform, −37.18 mV | **1**, and it is uniform: no pattern | yes, both | — (spread 0.00 mV) |
| 2e-14 | pattern J (179 depolarised cells) | pattern K (179 depolarised cells) | **2**, 23.6 mV apart | yes, after a further 12,000 s sim phase: both frozen, 0.000 mV of movement over the final 3,000 s | −37.0 / −48.6 and −36.5 / −47.6 |
| **5e-14 (paper)** | **A** | **B** | **≥ 8** (A to F, D2, E2; perturbation and basins studies) | yes | −38 / −57 |
| 1.5e-13 | pattern G | pattern G, identical to 0.00 mV | **1** | yes, both (τ ≈ 150 s) | −39.1 / −63.2 |

The number of stable patterns changes with the Anion's coupling, and in both directions. With the Anion ten times less mobile between cells the tissue stays uniform: the uniform state, a saddle at the paper's value, is stable here, and neither start left it (the A start's 17.6 mV initial spread decayed to 0.00). With the Anion three times more mobile, the two starts that reach different patterns at the paper's value converge on the same one, a pattern of 90 depolarised cells with a deeper polarised level, and nothing in this study says whether it is the only one. In between, at 2e-14, the tissue had not settled after 6,000 s: both runs were still rearranging at rates (5e-3 to 2e-2 mV/s) that were not falling. A further 12,000 s sim phase from each saved state settled both (added 2026-11-07): the A start moved another 13 mV before freezing, the B start 0.8 mV, and the two ended 23.6 mV apart, each with 179 of 254 cells depolarised and a voltage step of only 11 mV, half that at the paper's value. So 2e-14 has at least two patterns too, shallower in voltage and slower to form. The A-start continuation began 1.3 mV from its saved state, a larger start-of-phase transient than the 0.1 to 0.2 mV seen elsewhere; it did not affect the end state's settling. End states are in `../perturbation/end_states.csv`.

## Sweep 2: gap junction surface area (ionic coupling)

| Surface area | A start → | B start → | Distinct end states | Settled |
|---|---|---|---|---|
| 1e-9 (nearly closed) | pattern H: 77 depolarised cells at −38.1 mV, the rest at −67.2 mV | pattern I: 79 depolarised cells at −36.8 mV, the rest at −67.2 mV | **2**, 36.6 mV apart | yes, both; no movement at all over the final 4,500 s |
| **1e-8 (paper)** | A | B | ≥ 8 | yes | |
| 1e-7 (very open) | **BETSE failed**: `IndexError: list index out of range` during the init | the same failure | — | — |

With the ionic coupling ten times weaker the two starts still reach two distinct patterns, each with a wider voltage range (29 mV between the two levels, against 19 mV at the paper's value) and frozen: the convergence rate reached exactly zero. The open-junction value could not be run; BETSE 1.5.0 exits with an index error in both runs. The failure is BETSE's, not a simulation outcome, and is left as is.

## What this does and does not show

- **Multistability of the gene-network tissue depends on the Anion's gap-junction coupling.** Too little, and no pattern forms; too much, and the two starts agree on one pattern. The paper's value sits in a window where many patterns coexist. This answers open question 3 of the plan for one parameter, one tissue, two starts per value, three values besides the paper's.
- **Weaker ionic coupling kept two patterns.** The protocol's own sweep variable did not remove multistability over the one decade that ran.
- **Baselines.** Two starts per value is the minimum that can show two states, so "1 state" here means "these two starts agreed", not "only one state exists". The two-state result at the paper's value, from the same two starts, is the comparison. Replicate noise is 0 mV; the smallest distance between any two distinct end states in this study is 23.6 mV; the 1.5e-13 runs agree to 0.00 mV.
- **Not shown:** anything at 1e-7 (failed), finer resolution in Dgj, the perturbation stability of G, H or I, or any result for another tissue or geometry. The patterns found at other coupling values are not the same objects as A to F and were not compared with them beyond the table.
