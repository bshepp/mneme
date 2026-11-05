# Perturbation study: results

**Run:** 2026-11-05, BETSE 1.5.0, Python 3.13, Linux (cloud container, 4 cores). **Numbers:** `results.json`, from `analyze.py`; the displacements applied are in the run directory's `perturbations.json`, summarised in `results.json` under `targets` and per run. **Design:** [README.md](README.md).

## Findings in one table

| Perturbation of a settled pattern | Voltage displaced by (mV, worst cell, first frame) | Pattern A | Pattern B |
|---|---|---|---|
| None (state written back unchanged) | 0.1 / 0.2 | returned | returned |
| Voltage only, 20 mV shift of the depolarised cells, Anion untouched | 0.1 / 0.2 (gone before the first frame) | returned | returned |
| Anion of the depolarised cells moved 25 % toward the polarised value | 1.8 / 1.6 | returned in 240 s | returned in 120 s |
| … 50 % | 3.9 / 3.5 | returned in 480 s | returned in 330 s |
| … 100 %: depolarised region fully erased | 20.2 / 18.7 | returned in 1,080 s | returned in 870 s |
| Multiplicative Anion noise, exp(0.25 z), two seeds | 1.0, 1.0 / 1.0, 0.9 | returned, both | returned, both |
| … exp(0.5 z) | 2.3, 2.8 / 2.5, 2.5 | returned, both | returned, both |
| … exp(1.0 z) (up to 17-fold changes in single cells) | 13.0, 8.8 / 7.3, 8.2 | returned in 450 and 810 s | returned in 360 and 330 s |
| Anion field moved 25 % of the way to the other pattern | 16.3 / 22.4 | returned in 1,230 s | **left B**, reached A after about 6,000 s |
| … 50 % (the midpoint; the same state from both sides) | 24.9 / 31.6 | **third state C** | **third state C** |
| … 75 % | 29.1 / 35.2 | **flipped to B** after about 6,000 s | **flipped to A** in about 3,000 s |

Both patterns returned from every perturbation that did not move the state toward the other pattern, including complete erasure of the depolarised region's Anion. Along the line from A to B, A's basin extends past the 25 % point, B's does not reach its own 25 % point, and the midpoint settles into a third stable state.

## Reference states

Settling criteria as in the earlier studies: rate below 1e-4 mV/s over the final 20 samples, estimated remaining drift below 0.1 mV. Same-state threshold 1.0 mV, largest per-cell difference.

| | A (from the bitmap gradient) | B (from uniform + 1 % x gradient) |
|---|---|---|
| Settled by the end of the 6,000 s init | Yes (0.0002 mV to go, τ = 369 s) | **No** (0.59 mV to go, τ = 1,431 s) |
| Moved during the 3,000 s control continuation | 0.0002 mV | 0.177 mV |
| Settled by the end of the continuation | Yes | Yes |
| Reference state | Last frame of the continuation (9,000 s) | Last frame of the continuation (9,000 s) |
| Mean / spread (mV) | −48.67 / 34.14 | −49.24 / 33.66 |
| Depolarised cells (above the midpoint voltage) | 111 of 254 | 115 of 254 |
| Voltage of depolarised / polarised cells (mV) | −37.9 / −57.1 | −38.5 / −58.1 |
| Anion in depolarised / polarised cells (mmol/L) | 2.16 / 0.22 | 1.99 / 0.20 |

A and B differ by 35.0 mV in the worst cell and share 22 depolarised cells. This is a new seeded world (254 cells; the multistability study's had 246), so these are new instances of the two patterns, reached from the same two starting conditions.

Pattern B had not settled when its state was saved for perturbation. Its perturbed runs therefore start 0.18 mV from the reference, and the `ctrl` run's displacement of 0.22 mV at the first frame is that offset plus the start-of-phase transient. Every B result is read against that baseline.

## Return

| Run | Anion change, worst cell / mean (fraction) | Displacement (mV) | Distance from own pattern at the end (mV) | Return time (s) |
|---|---|---|---|---|
| A ctrl | 0 / 0 | 0.09 | 0.000 | 30 |
| A erase 25 % | 0.24 / 0.09 | 1.84 | 0.000 | 240 |
| A erase 50 % | 0.48 / 0.19 | 3.93 | 0.001 | 480 |
| A erase 100 % | 0.97 / 0.38 | 20.21 | 0.008 | 1,080 |
| A noise 0.25, seeds 1, 2 | 1.05 / 0.18; 1.04 / 0.21 | 0.95; 1.00 | 0.000; 0.000 | 30; 30 |
| A noise 0.5 | 3.20 / 0.37; 3.15 / 0.44 | 2.29; 2.78 | 0.000; 0.001 | 120; 300 |
| A noise 1.0 | 16.7 / 0.89; 16.2 / 1.11 | 13.02; 8.76 | 0.001; 0.004 | 450; 810 |
| A toward B 25 % | 5.7 / 1.09 | 16.30 | 0.012 (0.013 mV to go) | 1,230 |
| B ctrl | 0 / 0 | 0.22 | 0.000 | 30 |
| B erase 25 % | 0.24 / 0.10 | 1.59 | 0.001 | 120 |
| B erase 50 % | 0.48 / 0.20 | 3.53 | 0.002 | 330 |
| B erase 100 % | 0.96 / 0.39 | 18.72 | 0.007 | 870 |
| B noise 0.25 | as A | 1.03; 0.89 | 0.000; 0.000 | 60; 30 |
| B noise 0.5 | as A | 2.48; 2.53 | 0.001; 0.001 | 90; 120 |
| B noise 1.0 | as A | 7.34; 8.24 | 0.002; 0.003 | 360; 330 |

The return time is the first time after which the run stays within 1.0 mV of its reference. It grows with the displacement: about 240 s per 2 mV for the erase series, and 1,080 s for a 20 mV displacement, consistent with a relaxation time of about 400 s (A) toward a state 20 mV away. The noise runs used the same random fields for A and B (same seeds), so the two columns are the same disturbance applied to the two patterns.

Every one of these 22 runs met the settling criterion except A toward B 25 %, which was 0.013 mV from settled at 3,000 s with 0.012 mV to go, inside the threshold.

### The voltage-only runs

Shifting the depolarised cells' voltage by 20 mV while leaving the Anion alone left no trace at the first sampled frame: the displacement there (0.09 and 0.22 mV) equals the control's. Voltage is the fast variable of this system and relaxes to the Anion's dictate in far less than the 30 s sampling interval, so the saved state's voltage carries no memory of its own. BETSE also recomputes the voltage from the charge state when a phase starts, which may have removed the edit before the first time step. Either way the result is the same: there is nothing to perturb in the voltage alone.

## Flip, and a third state

The "toward" runs move every cell's Anion a fraction f of the way from one pattern's field to the other's. The point at f from A is the point at 1 − f from B, so the three distinct starting states were each run twice from the two sides (exact repeats, to within 0.02 mV: the two sides compute the same field in a different floating-point order).

| Start on the A–B line | Displacement from A / from B (mV) | End of the 3,000 s run | After a further 6,000 s |
|---|---|---|---|
| 75 % A + 25 % B | 16.3 / 35.2 | A, within 0.012 mV | — |
| 50 % A + 50 % B | 24.9 / 31.6 | **C**: 28.6 mV from A, 35.6 mV from B, settled (0.001 mV to go, τ = 185 s) | still C: moved 0.001 mV, settled |
| 25 % A + 75 % B | 29.1 / 22.4 | 7.3 mV from B and 35.2 mV from A, not settled | **B**, within 0.000 mV, settled |

**State C** depolarises 106 cells at −37.2 mV against −56.6 mV, the same two voltage levels as A and B. Of its 106 depolarised cells, 77 are also depolarised in A and 26 in B (A and B share 22). Its mean voltage, −48.50 mV, lies between A's and B's. It was reached only from the midpoint, which was run twice as an exact repeat; nothing else in this study or the earlier ones started near it. It stayed put for 9,000 s under the settling criterion.

**The basin boundary along this line is not at the midpoint.** From A's side, 25 % of the way to B still returns to A. From B's side, 25 % of the way to A does not return to B: it ends at A, after about 6,000 s. B's basin is the narrower one in this direction, and B relaxes more slowly (τ ≈ 1,400 s against 370 s), which is also why the B-ward run took 6,000 s to arrive.

## What this does and does not show

- **Both patterns are attractors in the operational sense of this study**: each returned, to within 0.01 mV, from eleven distinct displacements of up to 20 mV, including full erasure of its depolarised region and random Anion changes of up to 17-fold in single cells. The control runs moved 0.000 mV. Replicate noise in the earlier studies was 0 mV.
- **The pattern is not stored in the depolarised cells' Anion alone.** Setting those cells' Anion to the polarised level, so that the Anion field was nearly uniform at 0.2 mmol/L, still regrew the same pattern in about 1,000 s. The residual structure in the polarised cells' Anion, or the tissue geometry acting on the near-uniform field, is enough to select the pattern. The earlier finding that a uniform Anion plus a 1 % gradient goes to B, not A, says the selection is sensitive to small residuals.
- **Displacement toward the other pattern is what flips it.** Random and erasing displacements of the same voltage size (13 to 20 mV) all returned; a 16 mV displacement along the A–B line was the only one that strained the return (1,230 s), and 22 mV along it left B. Size alone does not decide the outcome; direction does.
- **There are at least three stable patterns**, not two, under this parameter set. C was found by a single construction, so nothing here says how large its basin is or whether natural starts reach it.
- **What is not shown:** how the basin boundary runs in any direction other than the A–B line; any result for a different tissue, geometry or parameter set; anything about real tissue. The relaxation times quoted are from fits to the final half of each run and are descriptive.
- **The same-state threshold** of 1.0 mV is far below the 28 to 36 mV between the three states and far above the 0.000 to 0.012 mV reached by the returning runs, so the three outcomes are unambiguous for every run.
