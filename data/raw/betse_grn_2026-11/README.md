# BETSE gene-network runs, November 2026

The per-cell membrane voltage of every run behind the perturbation, basins,
sweep and topology studies (`studies/`), packed from BETSE 1.5.0's
`Vmem2D_TextExport` output by `studies/pack_runs.py`. One seeded world of
254 cells (the 2018 patterns configuration at a 500 µm world) is shared by
every run; its cell positions are in `cells.csv`.

| File | Contents |
|---|---|
| `<run>__<phase>.csv.gz` | One row per sampled frame: `frame`, `time_s`, then `cell_0` … `cell_253` in mV to 4 decimals. The sampling interval is 30 s and the first frame is at 30 s. |
| `cells.csv` | `cell`, `x_um`, `y_um` |
| `manifest.csv` | Every file with its run, phase, frame count, end time and the final mean and spread |

Run names follow the study generators:

| Prefix | Study | Phase holding the result |
|---|---|---|
| `settle_A`, `settle_B` | perturbation: settling into patterns A and B | `init` (6,000 s), then `sim` (3,000 s control continuation) |
| `pert_<P>_<label>` | perturbation of pattern P | `sim` (3,000 s; 600 s for `vm_only`) |
| `cont_<run>` | continuation of a run from its final state | `sim` |
| `basin_L<len>_r<seed>` | basins: random Anion start | `init` (6,000 s) |
| `sweep_dgj<value>_<start>`, `sweep_gj<value>_<start>` | coupling sweep | `init` (6,000 s) |

The two `sweep_gj1em07` runs failed inside BETSE and have no file. Short
timing and test runs are left out. Read a file with
`pandas.read_csv(path)`; the first two columns are the frame index and time.
The uncompressed BETSE exports (about 270 MB) are not kept.
