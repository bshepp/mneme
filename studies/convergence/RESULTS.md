# Convergence study: results

**Run:** 2026-09-27, BETSE 1.5.0, Python 3.13, Windows 11.
**Data:** 13 runs, 42 cells, 418 frames each, 25,020 s of simulated time.
**Numbers:** [results.json](results.json). Produced by `analyze.py`.

## Finding

**Every run ended in the same voltage pattern. No multistability was found.**

Runs that started up to 48.8 mV apart ended within 0.32 mV of each other. This held with cells uncoupled and with cells coupled.

## Caveat: the runs did not meet the settling criterion

The criterion was fixed before the analysis: estimated remaining drift below 0.1 mV. The runs reached 0.20 to 0.32 mV. They are close to steady state, not at it.

This does not change the finding, for two reasons:

- The gap between runs was still shrinking at the end, and shrank at every frame after 3,600 s.
- The remaining gap between runs (0.17 to 0.32 mV) is the same size as the remaining drift of each run.

A longer run would be needed to state the end state to better than about 0.3 mV.

## Checks

| Check | Result |
|---|---|
| Same cells in every run | Largest coordinate difference 0 µm |
| Replicate noise | 0 mV. An exact repeat reproduced the run exactly, so BETSE is deterministic here. |
| Numerical stability | No run reported instability |
| Continuity between BETSE's two phases | Largest jump 0.20 mV, equal to the ordinary step size at that point |

## Runs

| Coupling | Start Na⁺/K⁺ | Mean at start (mV) | Mean at end (mV) | Final rate (mV/s) | Relaxation time (s) | Estimated drift left (mV) |
|---|---|---|---|---|---|---|
| Off | 145 / 5 | −52.81 | −36.94 | 5.5e-05 | 5,873 | 0.32 |
| Off | 120 / 30 | −59.75 | −36.94 | 5.0e-05 | 5,749 | 0.29 |
| Off | 100 / 50 | −63.59 | −36.93 | 4.7e-05 | 5,622 | 0.26 |
| Off | 75 / 75 | −67.34 | −36.92 | 4.2e-05 | 5,475 | 0.23 |
| Off | 50 / 100 | −70.18 | −36.92 | 3.8e-05 | 5,391 | 0.21 |
| Off | 10 / 140 | −60.81 | −36.91 | 3.7e-05 | 5,388 | 0.20 |
| On | 145 / 5 | −52.65 | −37.15 | 4.4e-05 | 5,604 | 0.25 |
| On | 120 / 30 | −59.71 | −37.15 | 4.3e-05 | 5,667 | 0.24 |
| On | 100 / 50 | −63.62 | −37.14 | 4.2e-05 | 5,701 | 0.24 |
| On | 75 / 75 | −67.43 | −37.14 | 4.2e-05 | 5,729 | 0.24 |
| On | 50 / 100 | −70.33 | −37.14 | 4.2e-05 | 5,744 | 0.24 |
| On | 10 / 140 | −61.22 | −37.13 | 4.2e-05 | 5,748 | 0.24 |

"Mean at start" is the first exported frame.

## Distinct end states

| Coupling | Largest gap at start (mV) | Largest gap at end (mV) | Distinct end states |
|---|---|---|---|
| Off | 48.83 | 0.32 | 1 |
| On | 48.05 | 0.17 | 1 |

Threshold for "same state": every cell within 1.0 mV.

## How the gap between runs closed

Largest per-cell difference between any two runs, in mV.

| Time (s) | Coupling off | Coupling on |
|---|---|---|
| 0 | 48.83 | 48.05 |
| 600 | 30.86 | 28.61 |
| 1,800 | 27.34 | 24.01 |
| 3,600 | 1.64 | 1.58 |
| 7,200 | 1.20 | 0.86 |
| 10,800 | 0.90 | 0.61 |
| 14,400 | 0.68 | 0.44 |
| 18,000 | 0.52 | 0.32 |
| 21,600 | 0.40 | 0.23 |
| 25,020 | 0.32 | 0.17 |

Most of the gap closed between 1,800 s and 2,700 s. After that it closed slowly, with a decay time of about 13,700 s uncoupled and 10,900 s coupled. That is slower than each run's own relaxation time of about 5,500 s, so the runs have a slow mode that the single-exponential fit does not capture. The drift estimates above may therefore be low.

## The end state

| | Coupling off | Coupling on |
|---|---|---|
| Mean voltage (mV) | −36.93 | −37.14 |
| Range across cells (mV) | −51.43 to 8.88 | −51.24 to 5.70 |

The pattern across cells comes from the configuration, which assigns different membrane permeabilities to different regions of the tissue.

Coupling changed the end state by up to 9.9 mV in individual cells and narrowed the range across cells. It did not create a second state.

## What this does and does not show

**It shows** that this published configuration has one stable voltage pattern, reached from six widely separated starting ion concentrations.

**It does not show** that simulated tissue cannot be multistable. This configuration has no element that would be expected to produce more than one stable state:

- It has no voltage-gated channels.
- Its gene regulatory network is switched off.
- Its coupled runs used a coupling 50 times weaker than BETSE's default.

## The original runs, re-examined

The two runs behind the withdrawn report were re-analysed in correct frame order, with the same method.

| Run | Internal K⁺ / protein | Cells | Mean at end (mV) | Relaxation time (s) | Estimated drift left (mV) | Settled |
|---|---|---|---|---|---|---|
| sim_1 | 5 / 10 | 153 | −38.68 | 3,755 | 1.21 | No |
| sim_2 | 65 / 80 | 156 | −57.12 | 3,566 | 0.38 | No |

These two end about 18 mV apart. The withdrawn report read that as two basins of attraction. It is not evidence of that:

- **They are different systems.** Internal protein is 10 in one and 80 in the other. Protein cannot cross the membrane, so it is a fixed parameter. This study held it fixed and found one end state.
- **They are different tissues.** The cell counts differ.
- **Neither had settled.**

## Next

To test for multistability, the configuration needs a mechanism that could produce it. Candidates, in BETSE's own terms:

| Mechanism | Why it could matter |
|---|---|
| Voltage-gated channels | Give positive feedback between voltage and permeability |
| Gene regulatory network coupled to voltage | The 2018 "patterns" configuration uses one |
| Stronger gap-junction coupling | Needs a smaller time step, so roughly 100 times the compute |

Each should be tested with this same design: one tissue, fixed parameters, many starting conditions, and an exact repeat.
