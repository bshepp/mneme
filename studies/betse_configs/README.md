# BETSE paper configurations used by the studies

Copies of the simulation configurations that `studies/convergence`,
`studies/multistability` and `studies/perturbation` start from. They are taken
unchanged from the BETSE source repository, `doc/yaml/paper/`, at commit
`4bba0624092f02ef52711d715d16ae7007d49aea` (BETSE 1.5.0), so the studies can be
reproduced without a checkout of BETSE.

| Directory | Source | Used by |
|---|---|---|
| `2016_Frontiers_Attractors/` | `doc/yaml/paper/2016_Frontiers/Attractors/` | convergence study (`paper.yaml`), VGC method (`paper_vgc.yaml`) |
| `2018_PBMB_Patterns/` | `doc/yaml/paper/2018_PBMB/Patterns/` | GRN method (`paper_grn.yaml`), perturbation study |

The files are copyright Alexis Pietak and Cecil Curry and redistributed under
the BETSE BSD 2-clause license in `LICENSE.betse`. The models are described in
Pietak and Levin, *Front. Bioeng. Biotechnol.* 2016 (attractors) and Pietak and
Levin, *Prog. Biophys. Mol. Biol.* 2018 (patterns); cite those papers when
using or adapting them.

The study generators expect `paper_grn.yaml` or `paper_vgc.yaml` next to the
`geo/` and `extra_configs/` directories. From a study directory:

```bash
mkdir runs && cd runs
cp -r ../../betse_configs/2018_PBMB_Patterns/{extra_configs,geo} .
cp ../../betse_configs/2018_PBMB_Patterns/patterns_2018.yaml paper_grn.yaml
```
