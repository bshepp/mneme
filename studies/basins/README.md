# Basin mapping: random starts on the gene-network tissue

Follows [studies/multistability](../multistability/RESULTS.md), where the 2018 gene-network (GRN) tissue reached pattern A from three gradient starts and pattern B from two near-uniform starts. Five starts cannot say how many patterns there are or how the starting condition decides the outcome. This study starts the same tissue from random Anion fields.

## Design

| Item | Value |
|---|---|
| Tissue, parameters, solver | As in [studies/perturbation](../perturbation/README.md): the same seeded world, the paper's parameters, fast solver |
| Varied: starting condition | The Anion's initial distribution, from a random bitmap. Each cell's Anion starts at the paper's 1.0 mmol/L times the bitmap's red value at the cell (0 to 1), the mechanism of the paper's own bitmap start |
| Random fields | Gaussian-filtered white noise scaled to [0, 1], at two correlation lengths, 25 µm and 100 µm (cells are about 20 µm apart; the world is 500 µm), four seeds each |
| Duration | 6,000 s init phase, sampled every 30 s; no sim phase |
| Replicate | BETSE reproduced identical runs exactly in the earlier studies (0 mV replicate noise), so none is added here |

## Analysis

`analyze.py` judges each run settled or not (criteria of `../convergence/analyze.py`), measures the largest per-cell difference between its end state and the reference patterns A and B (from the perturbation study's control continuations), and counts distinct states among the runs and the references at the 1.0 mV threshold.

| Claim | Baseline it must beat |
|---|---|
| "A random start reached A (or B)" | Within 1.0 mV of the reference, against an A-to-B distance of about 30 mV |
| "There is a third pattern" | More than 1.0 mV from both references, settled, and farther from them than its own remaining drift |

## Reproducing

In the perturbation study's run directory, after its world is seeded:

```bash
python ../../basins/make_configs.py 4
for n in $(cat basin_runs.txt); do PHASES="init,plot init" SKIP_SEED=1 bash ../../convergence/run_one.sh "$n" . & done; wait
python ../../basins/analyze.py . --out ../../basins/results.json
```

## Results

See [RESULTS.md](RESULTS.md).
