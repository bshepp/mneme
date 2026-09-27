# Convergence study

Do runs of one simulated tissue, started from different ion concentrations, settle to the same voltage pattern?

This is steps 1 to 3 of the [multistability protocol](../../docs/MULTISTABILITY_PROTOCOL.md), applied to a published BETSE configuration.

## Design

| Item | Value |
|---|---|
| Base configuration | `attractors_2016_1.yaml` from the BETSE source, `doc/yaml/paper/2016_Frontiers/Attractors/` |
| Tissue | One circular cluster, 80 µm world, lattice disorder 0 so that every seed gives the same cells |
| Held fixed | All parameters, including internal protein and chloride |
| Varied: starting condition | Internal Na⁺ / K⁺ of 145/5, 120/30, 100/50, 75/75, 50/100, 10/140 mmol/L |
| Varied: coupling | Gap-junction surface area 1e-15 m² (uncoupled, as published) and 1e-9 m² (coupled) |
| Replicate | One exact repeat of the published starting condition |
| Duration | 3,600 s initialisation plus 21,600 s simulation, sampled every 60 s |
| Time step | 0.01 s |

### Why these choices

- **Protein is held fixed.** The two published configurations differ in internal protein (10 against 80 mmol/L). Protein cannot cross the membrane, so that difference changes the system, not only its starting point. Runs that differ in it are expected to end in different states.
- **Lattice disorder is zero.** BETSE ties starting ion concentrations to the seeded world, so each run is seeded separately. With disorder at zero the seeds are identical, which `analyze.py` checks.
- **Coupling is 1e-9, not BETSE's default of 5e-8.** The default was numerically unstable at this time step. 1e-9 was the strongest value tested that stayed stable.

## Criteria

Fixed in `analyze.py` before the runs were analysed.

| Criterion | Value |
|---|---|
| Settled: rate | Below 1e-4 mV/s over the final 20 samples |
| Settled: estimated remaining drift | Below 0.1 mV |
| Same state | Every cell agrees to within 1.0 mV |

## Reproducing

```bash
# 1. Get BETSE and the published configuration
pip install betse
git clone --depth 1 https://github.com/betsee/betse.git
cp -r betse/doc/yaml/paper/2016_Frontiers/Attractors/geo .
cp -r betse/doc/yaml/paper/2016_Frontiers/Attractors/extra_configs .
cp betse/doc/yaml/paper/2016_Frontiers/Attractors/attractors_2016_1.yaml paper.yaml

# 2. Generate the 13 configurations and run them
python make_configs.py full
for n in $(cat full_runs.txt); do bash run_one.sh "$n" & done; wait

# 3. Analyse
python analyze.py . --out results.json
```

Each run takes about 4.5 hours on one core. On Windows, BETSE needs backslashes in the paths it is given; `run_one.sh` handles the log path.

## Results

See [RESULTS.md](RESULTS.md).
