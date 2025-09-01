# Module 3: CLI Quickstart — Generate → Analyze → Visualize

- Objectives
  - Use Mneme CLI to generate synthetic data and run the bioelectric pipeline
  - Explore topology backends and attractor options
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
Expected: `results_cli/analysis_results.hdf5`

## 3.3 Visualize dashboard
```bash
mneme visualize results_cli/analysis_results.hdf5 -o plots -f png
```
Expected: `plots/dashboard.png` with field, topology, and any attractor summaries.

If the visualization CLI fails on your system, you can drive it from Python:

```python
from mneme.utils.io import load_results
from mneme.analysis.visualization import FieldVisualizer
from mneme.types import AnalysisResult, Field

raw = load_results('results_cli/analysis_results.hdf5', format='hdf5')
ar = AnalysisResult(
    experiment_id=raw.get('experiment_id','n/a'),
    timestamp=str(raw.get('timestamp','n/a')),
    raw_data=Field(data=raw['raw_data']['data']) if isinstance(raw.get('raw_data'), dict) else None,
    processed_data=Field(data=raw['processed_data']['data']) if isinstance(raw.get('processed_data'), dict) else None,
)
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
- Disable attractors:
```bash
mneme analyze data/synthetic/quickstart.npz --attractor-method none -o results_noattr
```

## Exercises
1) Compare cubical vs rips outputs (feature vector length; diagram counts)
2) Increase `--attractor-threshold` and note changes in detected attractors
3) Try `--attractor-method clustering` with `--attractor-min-samples 20`

Solutions (outline)
- Cubical operates directly on grids; Rips requires point-cloud conversion and may produce different diagram sparsity
- Higher thresholds reduce recurrence connections → fewer attractors
- Clustering groups dense regions; raising min_samples filters small basins