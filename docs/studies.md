# Simulation Studies

The results currently asserted by this project are the simulation studies below. Each is one tissue geometry and one parameter set, run in BETSE 1.5.0 under the [multistability protocol](MULTISTABILITY_PROTOCOL.md), with the criterion for "settled" and the same-state threshold fixed before the runs were analysed. Each study has a design page (what was run and why, with the baseline every claim must beat) and a results page (the numbers, and what they do and do not show). The configs, analysis scripts and result files are in the repository under [`studies/`](https://github.com/bshepp/mneme/tree/main/studies).

| Study | Question | Finding | Pages |
|---|---|---|---|
| Convergence (2026-09) | Does the published 2016 configuration have more than one stable state? | No. Thirteen runs from starts up to 49 mV apart ended within 0.3 mV of each other, coupled or not. | [Design](studies/convergence/README.md) · [Results](studies/convergence/RESULTS.md) |
| Multistability (2026-09) | Does adding a mechanism give more than one state? | A voltage-coupled gene network (the 2018 patterns configuration) gives two stable voltage patterns under one parameter set; Kir2.1 plus a Na⁺ leak gives one. | [Design](studies/multistability/README.md) · [Results](studies/multistability/RESULTS.md) |
| Perturbation (2026-11) | Do the gene-network patterns survive a disturbance? | Both patterns return from every displacement of up to 20 mV that is not aimed along the line to the other pattern, including full erasure of the depolarised region's Anion. Half way along that line is a third stable pattern. A second round found that patterns from random starts sit in shallower basins. | [Design](studies/perturbation/README.md) · [Results](studies/perturbation/RESULTS.md) |
| Basins (2026-11) | Which pattern does a random start reach? | Eight random Anion starts reached six distinct patterns. Fine-grained starts went to the two known patterns; coarse starts mostly to new ones. | [Design](studies/basins/README.md) · [Results](studies/basins/RESULTS.md) |
| Sweep (2026-11) | Does the pattern count change with gap-junction coupling? | Yes, with the Anion's diffusion constant: none at a tenth of the paper's value, two at 2e-14, at least eight at the paper's value, one at three times it. Tenfold weaker ionic coupling kept two. | [Design](studies/sweep/README.md) · [Results](studies/sweep/RESULTS.md) |
| Topology (2026-11) | Does persistent homology tell the patterns apart better than the mean voltage? | For two patterns, yes at 2 to 5 mV of per-cell noise; both fail at 10 mV and above. For six patterns, topology holds to 1 mV of noise where the mean already fails. | [Design](studies/topology/README.md) · [Results](studies/topology/RESULTS.md) |

## How to read them

- **Simulation results only.** Nothing here is a statement about real tissue. The gene-network tissue is a published BETSE model; the patterns found are properties of that model at one parameter set.
- **One geometry, one parameter set.** Every gene-network result is from one seeded world of 254 cells (246 in the first multistability study). Nothing says how general the behaviour is.
- **Baselines.** Replicate noise is 0 mV: BETSE reproduces a repeated run exactly. Two states count as different when the largest per-cell voltage difference between them exceeds 1.0 mV; the patterns found are 24 to 36 mV apart. Topology is judged against the mean voltage, and against a per-cell comparison that knows cell identities.
- **Core-tier code only.** The analyses use BETSE loading, the settling criterion, and (for the topology study) cubical persistent homology with the Wasserstein distance. None uses the frozen or experimental components.
- **Rules for claims** are in the [scope document](SCOPE.md): a result is reported only beside a null model or baseline, only from core-tier output, only inside a method's measured operating range, and only with replication.

The per-cell voltage trace of every gene-network run is in the repository under `data/raw/betse_grn_2026-11/` as gzipped CSV.
