# Module 3: CLI Quickstart — Generate → Analyze → Visualize

- Objectives
  - Use Mneme CLI to generate synthetic data and run the bioelectric pipeline
  - Explore topology backends and the (experimental, opt-in) attractor options
- Time: 45–60 minutes

## 3.1 Generate synthetic data
```bash
mneme generate -o data/synthetic/quickstart.npz -t bioelectric -s 64,64 --timesteps 10 --seed 7
```

## 3.2 Analyze (bioelectric defaults)
```bash
mneme analyze data/synthetic/quickstart.npz \
  --pipeline bioelectric \
  --topology-backend cubical \
  -o results_cli
```
Expected: `results_cli/analysis_results.hdf5`. The command prints the status of each stage and exits non-zero if any stage fails. Reconstruction is reported as skipped when the input has no sparse `observations`/`positions`; it is not faked.

## 3.3 Visualize dashboard
```bash
mneme visualize results_cli/analysis_results.hdf5 -o plots -f png
```
Expected: `plots/dashboard.png` with field, topology, and any attractor summaries (only if attractor detection was opted into; see 3.4).

If you prefer Python, you can also drive visualization programmatically:

```python
from mneme.utils.io import load_results
from mneme.analysis.visualization import FieldVisualizer
from mneme.types import AnalysisResult, Field

ar = load_results('results_cli/analysis_results.hdf5')  # returns AnalysisResult
FieldVisualizer().create_analysis_dashboard(ar)
```

## 3.4 Backend and attractor variations
- Rips (point-cloud):
```bash
mneme analyze data/synthetic/quickstart.npz \
  --pipeline bioelectric \
  --topology-backend rips \
  -o results_cli_rips
```
- Enable attractor detection (experimental; off by default, see [SCOPE.md](../SCOPE.md)):
```bash
mneme analyze data/synthetic/quickstart.npz --attractor-method recurrence -o results_attr
```
The detectors locate dense or recurrent regions of the trajectory but cannot classify them; every region is reported as `undetermined`. Without the flag (or an `attractors` section in a config file) the stage does not run; `--attractor-method none` removes an `attractors` section supplied by a config file. Attractor detection only runs when the input is a time series (3D).

## Exercises
1) Compare cubical vs rips outputs (feature vector length; diagram counts)
2) With `--attractor-method recurrence`, increase `--attractor-threshold` and note changes in the number of regions found
3) Try `--attractor-method clustering` with `--attractor-min-samples 20`

Solutions (outline)
- Cubical operates directly on grids; Rips requires point-cloud conversion and may produce different diagram sparsity
- Higher thresholds reduce recurrence connections → fewer regions
- Clustering groups dense regions; raising min_samples filters small clusters