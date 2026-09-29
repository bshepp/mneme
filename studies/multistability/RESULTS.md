# Multistability study: results

**Run:** 2026-09-28 to 2026-09-29, BETSE 1.5.0, Python 3.13, Windows 11.
**Numbers:** `results_grn.json`, `results_vgc.json`. Produced by `../convergence/analyze.py`.
**Design:** [README.md](README.md).

## Findings in one table

| Method | Mechanism added | Starts compared | Distinct end states | Settled | Verdict |
|---|---|---|---|---|---|
| GRN | Gene network coupled to voltage | 6 (+ repeat) | 3, of which 2 stable | Yes, all | **Two stable patterns: multistability** |
| VGC, Na⁺ leak 3e-18 | Kir2.1 + Na⁺ leak | 2 | 1 | No (0.2 mV to go) | One state |
| VGC, Na⁺ leak 1e-17 | Kir2.1 + Na⁺ leak | 2 (+ repeat) | 1 | No (0.5 mV to go) | One state |
| VGC, Na⁺ leak 3e-17 | Kir2.1 + Na⁺ leak | 1 (other start unstable) | — | No (1.4 mV to go) | No comparison possible |

Replicate noise was 0 mV in both methods: BETSE reproduced each repeated run exactly.

## Method GRN: gene network coupled to voltage

246 cells, 215 frames, 6,420 s of simulated time. Every run met the settling criterion (rate below 1e-4 mV/s; estimated remaining drift below 0.1 mV, with the largest at 0.0003 mV).

### End states

| Start (Anion distribution) | Mean at start (mV) | Mean at end (mV) | Spread across cells at end (mV) | Relaxation time (s) |
|---|---|---|---|---|
| Bitmap gradient (the paper's) | −48.35 | −48.37 | 33.51 | 410 |
| Bitmap gradient, exact repeat | −48.35 | −48.37 | 33.51 | 410 |
| x gradient | −45.33 | −48.37 | 33.51 | 409 |
| y gradient | −45.36 | −48.37 | 33.51 | 411 |
| Uniform | −37.18 | −37.18 | 0.00 | 0 |

Distance matrix between end states, largest per-cell difference in mV:

| | bitmap | gradx | grady | uniform |
|---|---|---|---|---|
| bitmap | 0 | 0.0003 | 0.00001 | 26.3 |
| gradx | 0.0003 | 0 | 0.0003 | 26.3 |
| grady | 0.00001 | 0.0003 | 0 | 26.3 |
| uniform | 26.3 | 26.3 | 26.3 | 0 |

### What the two states are

**The patterned state.** Three starts that differ in the direction and shape of the initial Anion gradient (up to 21.8 mV apart at the start) ended in the *same* pattern, to within 0.0003 mV. The pattern has 93 of 246 cells depolarised to about −34 mV and the rest at about −59 mV. The starting gradient decided how long convergence took (3,400 s from the y gradient, 5,000 s from the x gradient), not where it ended. The pattern is therefore set by the tissue, not by the initial condition.

**The uniform state.** With a perfectly uniform Anion, every cell stayed at exactly −37.18 mV: the spread across cells was 0.000 mV at every frame. This is a symmetric fixed point. A perfectly symmetric start cannot break symmetry, so this run alone cannot say whether the state is stable or a saddle that any asymmetry would leave.

### Perturbation test

Two more runs started from the uniform Anion concentration plus a small x gradient: 1% and 0.1% of the concentration. The 0.1% start differed from perfect uniformity by 0.007 mV.

| Start | Spread at start (mV) | Spread at end (mV) | Distance from the uniform state at end (mV) | Distance from pattern A at end (mV) | Settled |
|---|---|---|---|---|---|
| Uniform + 1% gradient | 0.074 | 33.29 | 26.9 | 31.9 | Yes |
| Uniform + 0.1% gradient | 0.007 | 33.29 | 26.9 | 31.9 | Yes |

Both left the uniform state within 600 s. **The uniform state is unstable**: it is a saddle, not an attractor.

Both ended in the same new pattern, within 0.01 mV of each other, and that pattern is **not** the one the gradient starts reached. Call the pattern from the bitmap, x and y gradient starts *pattern A* and the pattern from the near-uniform starts *pattern B*.

| | A: cells depolarised | A: centroid of depolarised cells (x, y in µm) | B: cells depolarised | B: centroid (x, y) | Cells depolarised in both |
|---|---|---|---|---|---|
| | 93 of 246 | (285, 259) | 96 of 246 | (262, 273) | 28 |

The two patterns depolarise different regions of the same tissue to the same voltages (about −37 mV against −55 mV). A and B differ by 31.9 mV in the worst cell.

Grouping all seven end states at the 1.0 mV threshold gives three states: A (four runs, including the repeat), B (two runs), and the uniform saddle (one run).

## Method VGC: voltage-gated channels

42 cells, 418 frames, 25,020 s of simulated time. No run met the settling criterion; remaining drift was 0.19 to 1.40 mV against a limit of 0.1 mV.

### End states

| Na⁺ leak (m²/s) | Start | Mean at start (mV) | Mean at end (mV) | Spread at end (mV) | Drift left (mV) |
|---|---|---|---|---|---|
| 3e-18 | Depolarised (145/5) | −27.99 | −22.69 | 57.45 | 0.21 |
| 3e-18 | Polarised (10/140) | −43.42 | −22.68 | 57.41 | 0.19 |
| 1e-17 | Depolarised | −13.64 | −6.44 | 45.93 | 0.49 |
| 1e-17 | Depolarised, exact repeat | −13.64 | −6.44 | 45.93 | 0.49 |
| 1e-17 | Polarised | −19.63 | −6.44 | 45.81 | 0.43 |
| 3e-17 | Depolarised | −5.40 | +4.75 | 42.97 | 1.40 |
| 3e-17 | Polarised | numerically unstable, no data | | | |

For comparison, the same tissue with no channels added ended at a mean of −36.94 mV (convergence study).

### How the gap between the two starts closed

Largest per-cell difference between the depolarised and polarised runs, in mV:

| Time (s) | Leak 3e-18 | Leak 1e-17 |
|---|---|---|
| 0 | 49.5 | 44.5 |
| 600 | 29.4 | 31.7 |
| 1,800 | 20.3 | 20.4 |
| 3,600 | 0.9 | 0.6 |
| 7,200 | 1.8 | 0.4 |
| 14,400 | 0.3 | 0.2 |
| 25,020 | 0.04 | 0.09 |

At both leak strengths the two starts ended within 0.1 mV of each other, closer than each run's own remaining drift. One state.

### What the channels did

The Na⁺ leak depolarised the tissue in proportion to its strength (−36.9 → −22.7 → −6.4 → +4.8 mV mean) and narrowed the spread across cells. The inward-rectifier channel did not create a second stable voltage. At the strongest leak the polarised start was numerically unstable at this time step.

## Comparison of the two methods

| | GRN | VGC |
|---|---|---|
| Produced a spatial pattern | Yes, and the same one from three different starts | No new pattern; the existing profile pattern shifted with the leak |
| Second stable state | Yes: two distinct patterns, each reached from more than one start | No |
| Settled by the criterion | Yes, in about 400 s | No, after 25,000 s |
| Compute per run | About 1 hour | About 12 hours with 13 processes sharing the machine |
| Numerical trouble | Two starting conditions unusable | One run unstable |

## What this does and does not show

- **The GRN system is multistable.** Under one parameter set it has at least two stable voltage patterns. Each is an attractor in the operational sense used here: reached from more than one distinct start, settled by the pre-set criterion, and separated from the other by 31.9 mV against a replicate noise of 0 mV. This is the first result in this project that shows the behaviour the hypothesis predicts.
- **Which pattern the tissue reaches depends on its history.** Large asymmetric starts (three different gradients) went to pattern A. Nearly uniform starts went to pattern B. That is pattern memory in the sense the project set out to look for, in a simulation.
- **What is not shown:** how many patterns there are (only two starts reached B, and both were small x gradients), how large a disturbance each pattern survives, and whether either resembles anything in real tissue.
- **Kir2.1 plus a Na⁺ leak did not give bistability** at the three strengths tried. Other channel combinations, or a stronger Kir, were not tried.
- **The VGC runs did not settle** by the pre-set criterion. The one-state conclusion rests on the two starts being closer to each other than either is to its own end point.
- **Only one tissue geometry and one parameter set per method were used.** Nothing here says how general either result is.

## Next

1. Map the GRN basins: more starts (random Anion fields with several seeds, gradients at other angles and amplitudes) to count the patterns and see which starts lead where.
2. Perturb the patterned GRN states (protocol step 3.3): displace part of pattern A or B and measure whether it returns, and how large a displacement flips it to the other.
3. A parameter sweep (protocol step 3.4) over the Anion's gap-junction diffusion or the K⁺ channel inhibition constant, to find where the number of patterns changes.
3. For VGC: a stronger Kir2.1 and a smaller time step, if bistability from channels is still wanted.
