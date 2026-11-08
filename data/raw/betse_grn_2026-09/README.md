# BETSE gene-network runs, September 2026

The per-cell membrane voltage of every gene-network run behind the
multistability study (`studies/multistability/RESULTS.md`, method GRN),
packed from BETSE 1.5.0's `Vmem2D_TextExport` output by
`studies/pack_runs.py`. One seeded world of 246 cells (the 2018 patterns
configuration at a 500 µm world, the paper's cut disabled) is shared by
every run; its cell positions are in `cells.csv`. This is an earlier
seeding than the 254-cell world of the November studies in
`../betse_grn_2026-11/`, so patterns A and B here are different instances
of the two patterns.

| File | Contents |
|---|---|
| `<run>__<phase>.csv.gz` | One row per sampled frame: `frame`, `time_s`, then `cell_0` … `cell_245` in mV to 4 decimals. The sampling interval is 30 s and the first frame is at 30 s. |
| `cells.csv` | `cell`, `x_um`, `y_um` |
| `manifest.csv` | Every file with its run, phase, frame count, end time and the final mean and spread |

Each run has a 480 s `init` phase followed by a 5,970 s `sim` phase; the
study's end states are the last frame of `sim`.

| Run | Start (Anion distribution) | End state |
|---|---|---|
| `full_grn_ic_bitmap`, `full_grn_ic_bitmap_rep` | The paper's bitmap gradient, and its exact repeat | Pattern A |
| `full_grn_ic_gradx`, `full_grn_ic_grady` | Linear gradients along x and along y | Pattern A |
| `full_grn_ic_uniform` | Perfectly uniform | The uniform saddle, −37.18 mV in every cell |
| `perturb_grn_ic_near_uniform_1pct`, `_0p1pct` | Uniform plus a 1 % or 0.1 % x gradient | Pattern B |

Read a file with `pandas.read_csv(path)`; the first two columns are the
frame index and time. Packed on 2026-11-08 from the run directory kept
outside the repository (`data/betse-results/multistability_grn/`).
