# Multistability study: two candidate mechanisms

The [convergence study](../convergence/README.md) found one stable state in the published 2016 configuration, which has no mechanism expected to produce more. This study adds one, two different ways, and applies the same design to each: one tissue, fixed parameters, several starting conditions, one exact repeat.

## Method GRN: gene network coupled to voltage

| Item | Value |
|---|---|
| Base configuration | `patterns_2018.yaml` from the BETSE source, `doc/yaml/paper/2018_PBMB/Patterns/` |
| Mechanism | A cytosolic "Anion" inhibits a K⁺ leak channel; gap junctions are voltage sensitive; the Anion moves between cells through gap junctions |
| Tissue | The paper's geometry at half size (500 µm world, about 246 cells), one seeded world shared by all runs |
| Varied: starting condition | The Anion's initial spatial distribution: the paper's bitmap gradient, an x gradient, a y gradient, uniform |
| Perturbation test | Uniform plus an x gradient of 1% and of 0.1% of the concentration, to test whether the uniform state is stable |
| Replicate | One exact repeat of the bitmap start |
| Changed from the paper | The tissue cut in the simulation phase is disabled, so the runs are undisturbed |
| Duration | 500 s initialisation plus 6,000 s simulation, sampled every 30 s, fast solver, time step 0.01 s |

Two starting conditions were tried and dropped: a radial gradient fails inside BETSE 1.5.0 with an array-shape error, and a reversed x gradient (negative slope) is numerically unstable.

## Method VGC: voltage-gated channels

| Item | Value |
|---|---|
| Base configuration | `attractors_2016_1.yaml`, as in the convergence study |
| Mechanism | An inward-rectifier K⁺ channel (Kir2.1, fully-open permeability 5e-17 m²/s) plus a Na⁺ leak, both active from the start |
| Tissue | 80 µm world, 42 cells, lattice disorder 0 so every seed gives the same cells |
| Varied: Na⁺ leak strength | 3e-18, 1e-17, 3e-17 m²/s (the base membrane Na⁺ permeability is 7.5e-19) |
| Varied: starting condition | Internal Na⁺/K⁺ of 145/5 (depolarised) and 10/140 (polarised) mmol/L |
| Replicate | One exact repeat of the 145/5 start at leak 1e-17 |
| Duration | 3,600 s initialisation plus 21,600 s simulation, sampled every 60 s, full solver, time step 0.01 s |

The run at leak 3e-17 from the polarised start was numerically unstable and produced no data.

## What would count as multistability

Within one method and one parameter set: runs that all settle (by the criteria in `../convergence/analyze.py`) and end in more than one distinct state, with the gap between states far larger than the replicate noise and than each run's remaining drift.

## Reproducing

```bash
# From a directory holding paper_grn.yaml (patterns_2018.yaml) with its geo/
# and extra_configs/, or paper_vgc.yaml (attractors_2016_1.yaml) with its own.
python make_configs.py grn full     # or: vgc full
# GRN runs share one world: seed once, then run with SKIP_SEED=1.
betse --headless seed full_grn_ic_bitmap.yaml
for n in $(cat full_grn_runs.txt); do SKIP_SEED=1 bash ../convergence/run_one.sh "$n" & done; wait
# VGC runs seed themselves.
for n in $(cat full_vgc_runs.txt); do bash ../convergence/run_one.sh "$n" & done; wait

python ../convergence/analyze.py . --init-sampling 30 --sim-sampling 30 --out results_grn.json
python ../convergence/analyze.py . --out results_vgc.json
```

## Results

See [RESULTS.md](RESULTS.md).
