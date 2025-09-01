# Module 4: Pipeline Anatomy and Configuration

- Objectives
  - Understand built-in stages and default config
  - Apply CLI overrides and read stage summaries
- Time: 60 minutes

## 4.1 Read the source
- Start at `src/mneme/analysis/pipeline.py` (class `MnemePipeline`)
- Components: Quality check → Preprocess (denoise/normalize/register/interpolate) → Topology → Attractors → Reconstruction (identity fallback unless sparse obs provided)

## 4.2 Defaults
- `create_bioelectric_pipeline()` sets light denoise, per-frame normalize, linear interpolate, IFT reconstructor, cubical TDA, recurrence attractors (threshold 0.1)

## 4.3 Config overrides via CLI
Examples:
```bash
# Change topology backend
mneme analyze path.npz --topology-backend alpha

# Override attractors
mneme analyze path.npz --attractor-method recurrence --attractor-min-persistence 0.2
```

## 4.4 Inspect stage results
- Stage summaries are included in the pipeline result and reflected in reports/visuals
- Quality report keys: `snr`, `resolution_adequacy`, `dynamic_range`, `coherence_quality`, etc.

## Exercises
1) Enable registration (for 3D time series) and observe changes in quality metrics
2) Downsampled TDA: increase your input size (e.g., 256×256) and confirm stride downsampling kicks in for cubical backend (see code around 128 limit)
3) Add a custom stage in Python that thresholds the processed field and records area; run it via a short script using `MnemePipeline.add_stage`

Solutions (outline)
- Registration computes per-frame shifts; coherence/diff metrics can change
- For large 2D arrays, the pipeline subsamples before TDA for speed; verify with logs/stage summary
- `add_stage(name='threshold', stage=..., inputs=['processed_field'], outputs=['mask'])` and append to pipeline before run