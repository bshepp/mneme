# Coupling sweep on the gene-network tissue

Protocol step 5 of [docs/MULTISTABILITY_PROTOCOL.md](../../docs/MULTISTABILITY_PROTOCOL.md): does the number of stable patterns change with gap-junction coupling?

## Design

Two couplings are swept separately, each from the two starting conditions that reach patterns A and B at the paper's values (the bitmap Anion gradient, and a uniform Anion plus a 1% x gradient).

| Sweep | Parameter | Paper value | Values run | What it couples |
|---|---|---|---|---|
| `dgj` | The Anion's gap-junction diffusion constant `Dgj` (network file) | 5e-14 m²/s | 5e-15, 2e-14, 1.5e-13 | Movement of the pattern's memory variable between cells. The paper marks it as a master variable, range 1e-15 to 1e-13 |
| `gj` | `gap junction surface area` (main config) | 1e-8 | 1e-9, 1e-7 | Ionic current between cells, so how far a voltage difference spreads. Paper range 1e-9 to 1e-7 |

Everything else is as in [studies/perturbation](../perturbation/README.md): the same seeded world, fast solver, 6,000 s init phase sampled every 30 s, no sim phase. The paper's own values are the perturbation study's settle runs.

## Analysis

`analyze.py` groups runs by coupling value, judges each settled or not, and counts the distinct end states at each value, with reference patterns A and B included so a run can be labelled A, B or other. Two starts per value is the minimum that can show two states; a count of one at some value means both starts converged, and a count of two that both patterns still exist there (or that one was replaced).

| Claim | Baseline it must beat |
|---|---|
| "The number of states changes with coupling" | The two states at the paper's value, from the same two starts |

## Reproducing

In the perturbation study's run directory, after its world is seeded:

```bash
python ../../sweep/make_configs.py dgj
python ../../sweep/make_configs.py gj
for n in $(cat sweep_dgj_runs.txt sweep_gj_runs.txt); do PHASES="init,plot init" SKIP_SEED=1 bash ../../convergence/run_one.sh "$n" . & done; wait
python ../../sweep/analyze.py . --out ../../sweep/results.json
```

## Results

See [RESULTS.md](RESULTS.md).
