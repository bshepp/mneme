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

Note: If you loaded results via `mneme.utils.io.load_results(..., format='hdf5')`, you will receive a dictionary; convert it to an `AnalysisResult` (as shown in Module 3) before calling `create_analysis_dashboard`.
```

## Exercises
1) Add titles and annotations to highlight key topology features
2) Save a persistence image for H1 using `compute_persistence_image` and place it in your report

Run log (MVP)
- The `mneme.utils.io.load_results` HDF5 loader returns a dict; `FieldVisualizer.create_analysis_dashboard` expects an `AnalysisResult`. Convert the dict to `AnalysisResult` first (see Module 3 snippet). We will improve this in a future release.

Solutions (outline)
- Use Matplotlib annotations; summarize feature vector stats on the figure
- Derived images help visualize the distribution of persistence across birth/persistence space