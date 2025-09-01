# Module 8: Visualization and Reporting

- Objectives
  - Create dashboards and pipeline plots
  - Read stage summaries and annotate findings
- Time: 45–60 minutes

## 8.1 Dashboard
```python
from mneme.analysis.visualization import FieldVisualizer
from mneme.types import AnalysisResult

# Suppose you already have `result: AnalysisResult`
viz = FieldVisualizer()
fig = viz.create_analysis_dashboard(result)
fig.savefig('dashboard.png', dpi=300)
```

## 8.2 Pipeline stage plots
```python
fig2 = viz.plot_pipeline_results(result.metadata.get('stage_results', {}) if isinstance(result.metadata, dict) else {})
fig2.savefig('pipeline.png', dpi=300)

Note: `mneme.utils.io.load_results` now returns an `AnalysisResult` directly for HDF5 paths (both `.h5` and `.hdf5`), so you can pass it straight into `create_analysis_dashboard`.
```

## Exercises
1) Add titles and annotations to highlight key topology features
2) Save a persistence image for H1 using `compute_persistence_image` and place it in your report

Run log (MVP)
- The HDF5 loader now returns an `AnalysisResult`; dashboards render without conversion.

Solutions (outline)
- Use Matplotlib annotations; summarize feature vector stats on the figure
- Derived images help visualize the distribution of persistence across birth/persistence space