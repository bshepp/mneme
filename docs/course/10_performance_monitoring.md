# Module 10: Performance and Monitoring (MVP Tools)

- Objectives
  - Use basic parallel helpers and monitoring utilities
  - Measure stage timing and resource usage
- Time: 45–60 minutes

## 10.1 Monitoring
```python
from mneme.utils import monitoring
mon = monitoring.PipelineMonitor()
mon.start()
# with mon.track_stage('preprocessing'):
#     ...
metrics = mon.get_metrics()
print(metrics)
```

## 10.2 Parallel helpers
```python
from mneme.data.parallel import ParallelPipeline
# Example: wrap an existing pipeline for batch
pp = ParallelPipeline(pipeline=my_pipeline, backend='multiprocessing', n_workers=4)
pp.map(["data/a.npz", "data/b.npz"])  # pseudo-example
```

## 10.3 Exercises
1) Time topology vs no-topology runs on 128×128 vs 256×256 inputs; chart the difference
2) Profile preprocessing SNR improvements vs runtime; choose defaults for your data

Solutions (outline)
- TDA time grows with size; confirm subsampling behavior for cubical
- SNR/denoise trade-offs guide parameter choices for speed vs quality