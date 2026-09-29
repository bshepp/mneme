# Module 4: Pipeline Anatomy and Configuration

- Objectives
  - Understand built-in stages and default config
  - Apply CLI overrides and read stage summaries
- Time: 60 minutes

## 4.1 Read the source
- Start at `src/mneme/analysis/pipeline.py` (class `MnemePipeline`)
- Components: Quality check (experimental; uncalibrated thresholds) → Preprocess (denoise/normalize/register/interpolate) → Topology → Attractors (experimental; only if configured, and only on 3D input) → Reconstruction (skipped, with `status: 'skipped'`, unless sparse `observations` and `positions` are provided)
- `PipelineResult.success` is False when any stage fails; the stage names are in `failed_stages` and the messages in `errors`

## 4.2 Defaults
- `create_bioelectric_pipeline()` sets light denoise, per-frame normalize, linear interpolate, `gp_subset` reconstructor, cubical TDA (sublevel filtration). Attractor detection is not run by default; add an `attractors` section to the config or pass `--attractor-method` to opt in.

## 4.3 Config overrides via CLI
Examples:
```bash
# Change topology backend
mneme analyze path.npz --topology-backend alpha

# Opt in to (experimental) attractor detection
mneme analyze path.npz --attractor-method recurrence --attractor-min-persistence 0.2
```

## 4.4 Inspect stage results
- Stage summaries are included in the pipeline result and reflected in reports/visuals; `mneme analyze` prints each stage's status (`completed`, `skipped` or `failed`) and exits non-zero on failure
- Quality report keys: `snr`, `resolution_adequacy`, `dynamic_range`, `coherence_quality`, etc.

## Exercises
1) Enable registration (for 3D time series) and observe changes in quality metrics
2) Downsampled TDA: increase your input size (e.g., 256×256) and confirm stride downsampling kicks in for cubical backend (see code around 128 limit)
3) Add a custom stage in Python that thresholds the processed field and records area; run it via a short script using `MnemePipeline.add_stage`

Solutions (outline)
- Registration computes per-frame shifts; coherence/diff metrics can change
- For large 2D arrays, the pipeline subsamples before TDA for speed; verify with logs/stage summary
- `add_stage(name='threshold', stage_func=..., inputs=['processed_field'], outputs=['mask'])` and append to pipeline before run