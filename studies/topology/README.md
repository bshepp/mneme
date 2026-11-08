# Topology against mean voltage: telling the two patterns apart

Open question 4 of [project_plan.md](../../project_plan.md): can persistent homology of the voltage field tell the stable patterns apart better than the mean voltage can? This is the first use of Mneme's topology code on a question with a built-in baseline.

## Design

The two reference patterns are the settled states A and B of [studies/perturbation](../perturbation/README.md). They depolarise different regions of the same tissue to about the same voltages, so their means are close while their shapes differ.

Gaussian measurement noise of standard deviation σ is added independently to every cell's voltage, and the noisy sample is assigned to whichever reference is nearer under one statistic:

| Statistic | Distance | Tier |
|---|---|---|
| mean | difference of the mean voltage over cells | baseline |
| topology | 2-Wasserstein distance between persistence diagrams (H0 and H1 of the superlevel filtration, summed) of the field interpolated to a 64 × 64 grid; raw, and with features of persistence below 2 mV and below 5 mV dropped | core |
| cells | root-mean-square per-cell difference | ceiling: uses cell identities, which no summary statistic has |

σ runs from 0.5 to 20 mV; 40 draws per pattern per σ. Accuracy is the fraction assigned to the pattern they came from (chance 0.5). The separation ratio is the A-to-B distance over the median distance of a noisy sample from its own reference.

The persistence thresholds were fixed before the reference patterns were analysed: 5 mV is well below the patterns' depolarised-to-polarised step of about 25 mV, and the raw (0 mV) variant is reported beside them so the effect of dropping short bars is visible.

## Reproducing

```bash
python ../topology/compare_statistics.py <perturbation run directory> --out ../topology/results.json
python ../topology/compare_statistics.py --end-states ../basins/end_states.csv --out ../topology/results_six.json
```

The second form takes every voltage column of the CSV as a reference pattern (here the six patterns A to F) and needs no run directory.

## Results

See [RESULTS.md](RESULTS.md).
