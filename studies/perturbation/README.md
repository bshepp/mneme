# Perturbation study: do the gene-network patterns survive a disturbance?

Protocol step 4 of [docs/MULTISTABILITY_PROTOCOL.md](../../docs/MULTISTABILITY_PROTOCOL.md), on the system that [studies/multistability](../multistability/RESULTS.md) found to have two stable voltage patterns: the 2018 gene-network (GRN) configuration, in which a cytosolic "Anion" inhibits a K⁺ leak channel and moves between cells through voltage-sensitive gap junctions.

## Question

A settled pattern is displaced by a stated amount and the run continues. Does it return to the same pattern, flip to the other, or end somewhere else? How large a displacement flips it?

## Design

| Item | Value |
|---|---|
| Base configuration | `patterns_2018.yaml` (in [studies/betse_configs](../betse_configs/README.md)), world size 500 µm, the paper's cut disabled, as in the multistability study |
| Tissue | One seeded world shared by every run |
| Solver | BETSE's fast (equivalent-circuit) solver, as in the multistability study |
| Settling | Pattern A: init phase of 6,000 s from the paper's bitmap Anion gradient. Pattern B: the same from a uniform Anion plus a 1% x gradient. Then an unperturbed sim phase of 3,000 s as the control continuation. |
| Perturbation | The state saved at the end of the settling init is loaded, changed, saved as a new init file, and a 3,000 s sim phase is run from it |
| Sampling | Every 30 s |

### Why the state is edited directly

BETSE's scheduled interventions ("change K mem" and the like) have no effect under the fast solver: that solver fixes the leak conductance when the phase starts and only the network channels and gap junctions evolve. The gene network's channels have no scheduled on/off switch. So the displacement is applied to the saved state itself, which is also the most direct reading of the protocol ("displace the voltages by a stated amount, and continue the run").

The state of this system is the per-cell Anion concentration, which is slow and carries the pattern, and the per-cell voltage, which is fast and reads the Anion out through the K⁺ channel. Depolarised cells are the ones with a high Anion concentration (the K⁺ leak is inhibited there).

### Perturbations, applied to each pattern

| Label | Change | Levels |
|---|---|---|
| `ctrl` | None: the saved state written back unchanged. Baseline for the round trip through the pickle file. | 1 |
| `erase_fF` | In the depolarised cells, move the Anion F of the way to the mean Anion of the polarised cells. Partly or fully erases the pattern's depolarised region. | F = 25, 50, 100 % |
| `toward_fF` | Move every cell's Anion F of the way to the other pattern's Anion field. Walks along the line from one pattern to the other. | F = 25, 50, 75 % |
| `noise_sS_rR` | Multiply every cell's Anion by exp(S·z), z standard normal, seed R. A random direction. | S = 0.25, 0.5, 1.0; R = 1, 2 |
| `vm_only` | Shift the depolarised cells' voltage 20 mV toward the polarised value, Anion untouched. Shows the fast variable's own return. 600 s run. | 1 |

Depolarised cells are those above the midpoint between the pattern's lowest and highest cell voltage. The displacement is reported two ways: the Anion change applied, and the largest per-cell voltage difference from the reference pattern at the first sampled frame (30 s).

### Outcome of a run

With the reference state of each pattern taken as the last frame of its control continuation, and the same-state threshold of 1.0 mV (largest per-cell difference) from the earlier studies:

| Outcome | Condition at the end of the run |
|---|---|
| returned | within 1.0 mV of its own pattern |
| flipped | within 1.0 mV of the other pattern |
| other | neither |

Settling is judged by the criteria in `../convergence/analyze.py` (rate below 1e-4 mV/s over the final 20 samples, estimated remaining drift below 0.1 mV). The return time is the first time after which the distance to the pattern's own reference stays below the threshold.

### Baselines

| Claim | Baseline |
|---|---|
| "The pattern returned" | The `ctrl` run's distance from the reference, and the control continuation's own movement |
| "The pattern flipped" | The A-to-B distance, which must be far larger than the threshold |

## Reproducing

```bash
mkdir runs && cd runs
cp -r ../../betse_configs/2018_PBMB_Patterns/{extra_configs,geo} .
cp ../../betse_configs/2018_PBMB_Patterns/patterns_2018.yaml paper_grn.yaml
python ../make_configs.py settle
betse --headless seed settle_A.yaml                       # the shared world
for n in settle_A settle_B; do SKIP_SEED=1 bash ../../convergence/run_one.sh "$n" . & done; wait
python ../make_configs.py perturb                         # writes INITS/init_pert_*.betse.gz
for n in $(cat perturb_runs.txt); do PHASES="sim,plot sim" bash ../../convergence/run_one.sh "$n" . & done; wait
python ../analyze.py . --out ../results.json
```

Each settling run takes about two hours and each perturbed continuation about fifty minutes on one core (BETSE 1.5.0, about one simulated second per second).

## Results

See [RESULTS.md](RESULTS.md).
