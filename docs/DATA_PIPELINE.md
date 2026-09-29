# Mneme Data Pipeline Documentation

> Accuracy note: The quality checker (`mneme.data.validation` / `mneme.data.quality`, experimental), a basic parallel helper (`mneme.data.parallel`), a feature extractor (`mneme.analysis.features`), a results facade (`mneme.analysis.results_generator`) and monitoring utilities (`mneme.utils.monitoring`) exist in `src/`. Caching and recovery/checkpointing are roadmap examples and do not exist. Sections marked "(roadmap)" describe intent, not code. Component tiers are in [SCOPE.md](SCOPE.md).

## Overview

The Mneme data pipeline handles the flow of data from raw bioelectric measurements and synthetic generation through preprocessing, analysis, and final results. The pipeline is designed to be modular, reproducible, and scalable.

## Data Flow Architecture

```
Raw Data Sources          Preprocessing           Analysis              Results
================          =============           ========              =======
                                                                        
Bioelectric Images   -->  Denoising          -->  Field               --> Persistence
Voltage Maps        -->  Registration       -->  Reconstruction      --> Diagrams
BETSE Output        -->  Normalization      -->  Topology Analysis   --> 
Synthetic Fields    -->  Interpolation      -->  Steady-State        --> Reports
                                                 Analysis              Visualizations
```

The default pipelines run quality check, preprocessing, reconstruction and topology. Attractor detection, symbolic regression and autoencoding are experimental and are not run unless asked for.

## Data Formats and Standards

### 1. Raw Data Formats

#### Bioelectric Imaging Data
```python
# Standard format: HDF5 with structured metadata
{
    'voltage_fields': np.ndarray,  # Shape: (time, height, width)
    'timestamps': np.ndarray,      # Shape: (time,)
    'metadata': {
        'specimen_id': str,
        'experiment_date': str,
        'sampling_rate_hz': float,
        'voltage_unit': str,
        'spatial_resolution_mm': float,
        'experimental_conditions': dict
    }
}
```

#### Gene Expression Data
```python
# Format: Spatial expression matrices
{
    'expression_matrix': np.ndarray,  # Shape: (genes, spatial_points)
    'gene_names': List[str],
    'spatial_coordinates': np.ndarray,  # Shape: (spatial_points, 2)
    'time_point': float
}
```

### 2. Processed Data Format

```python
# Standardized processed data structure
class ProcessedField:
    data: np.ndarray           # Normalized field values
    mask: np.ndarray          # Valid data mask
    coordinates: np.ndarray   # Spatial coordinates
    timestamp: float          # Time point
    metadata: Dict[str, Any]  # Processing metadata
    
    def to_hdf5(self, path: str): ...
    def from_hdf5(cls, path: str): ...
```

## Pipeline Stages

### Stage 1: Data Ingestion

```python
from mneme.data import loaders

# Bioelectric data loader
loader = loaders.BioelectricDataLoader(
    data_dir="data/raw/planarian/",
    file_pattern="*.h5",
    lazy_load=True  # Load data on demand
)

# Iterate through files; each is a dict with the voltage field and timestamps
for experiment in loader:
    voltage_field = experiment["voltage_field"]
    timestamps = experiment["timestamps"]
```

For BETSE simulation output use `mneme.data.betse_loader.load_betse_cells()` (cell values in time order, no interpolation) or `betse_to_field()` (interpolated to a grid; the `inside_hull` mask in the metadata marks grid points that are fill, not simulation output).

### Stage 2: Quality Control (experimental)

```python
from mneme.data import quality

# Quality assessment. Verdicts come from fixed thresholds that have not
# been calibrated; the checker emits ExperimentalWarning.
qc = quality.QualityChecker()
report = qc.check_field(voltage_field)  # dict

# Checks: resolution, noise level, dynamic range, spatial coherence,
# temporal consistency

if report["overall_quality"] != "poor":
    processed_field = preprocess(voltage_field)
else:
    logger.warning(f"Poor data quality: {report}")
```

### Stage 3: Preprocessing

```python
from mneme.data import preprocessors

# Create preprocessing pipeline: steps are names, or (name, params) pairs
preprocessor = preprocessors.FieldPreprocessor([
    ('denoise', {'method': 'wavelet', 'threshold': 'soft'}),
    ('register', {'reference': 'first'}),          # temporal (3D) data only
    ('normalize', {'method': 'z_score', 'per_frame': True}),
    ('interpolate', {'target_shape': (256, 256)}),
])

# Apply preprocessing
processed = preprocessor.fit_transform(voltage_field)
```

#### Preprocessing Steps:

1. **Denoising** (`Denoiser`, method `gaussian`, `median` or `wavelet`)
   - Wavelet denoising for preserving edges
   - Gaussian filtering for smooth fields
   - Median filtering for impulse noise

2. **Registration** (`Registrator`, reference `first`, `mean` or `median`)
   - Align temporal sequences
   - Correct for specimen movement
   - Maintain spatial correspondence

3. **Normalization** (`Normalizer`, method `z_score`, `min_max` or `robust`)
   - Z-score normalization
   - Min-max scaling
   - Robust (median/MAD) scaling

4. **Interpolation** (`Interpolator`, method `nearest`, `linear` or `bicubic`)
   - Resampling to a target shape

### Stage 4: Feature Extraction

```python
from mneme.analysis import features

# Extract basic scalar features from a 2D field or a (T, H, W) sequence
extractor = features.FieldFeatureExtractor(smoothing_sigma=0.0)
feature_dict = extractor.extract(processed_field)

# Features: mean, std, min, max, gradient magnitude mean/std,
# Laplacian magnitude mean/std, roughness (aggregated over time for 3D)
```

### Stage 5: Core Analysis

```python
from mneme.core import field_theory, topology, attractors
from mneme.analysis import steady_state

# 1. Field reconstruction from sparse observations (values at positions)
reconstructor = field_theory.FieldReconstructor(method='gp_subset', resolution=(128, 128))
result = reconstructor.fit_reconstruct(observations, positions)
continuous_field = result.field.data
uncertainty = result.uncertainty

# 2. Topology analysis
# Cubical for 2D fields (default). 'sublevel' tracks pits, 'superlevel' peaks.
# H1 and above need GUDHI; without it only H0 is computed and a RuntimeWarning is emitted.
tda = topology.PersistentHomology(max_dimension=1, filtration='sublevel')
persistence_diagrams = tda.compute_persistence(continuous_field)

# Point-cloud backends
pc = topology.field_to_point_cloud(continuous_field, method='peaks', percentile=95.0)
rips = topology.RipsComplex(max_dimension=1)
rips_diagrams = rips.compute_persistence(pc)

# 3. Steady-state analysis of several relaxing runs (core), each of shape (n_times, n_cells)
report = steady_state.assess_steady_state(vmem, times, rate_tolerance=1e-4, drift_tolerance=0.1)
distinct = steady_state.count_distinct_states([run[-1] for run in runs], threshold=1.0)

# 4. Attractor detection (experimental, off by default). Detectors locate
# dense or recurrent regions; the type of every region is UNDETERMINED.
detector = attractors.AttractorDetector(method='recurrence')
regions = detector.detect(trajectory)
```

The VAE (`mneme.models.autoencoders`) and symbolic regression (`mneme.models.symbolic`) are experimental and are not part of the default pipeline.

### Stage 6: Results Generation

```python
from mneme.analysis import results_generator

# Bundle outputs and save with mneme.utils.io.save_results
generator = results_generator.ResultGenerator()
bundle = generator.compile({
    'raw_data': voltage_field,
    'processed_data': processed_field,
    'reconstruction': continuous_field,
    'topology': persistence_diagrams,
})

# Save results
generator.save(bundle, "experiments/results/exp_001/results.h5", format='hdf5')
```

## Data Pipeline Configuration

### Configuration Keys

The keys the pipeline reads are those returned by `default_config('standard')` and `default_config('bioelectric')`. This is the bioelectric default:

```yaml
preprocessing:
  denoise:     {enabled: true, method: gaussian, sigma: 1.0}
  normalize:   {enabled: true, method: z_score, per_frame: true}
  register:    {enabled: false}          # needs temporal (3D) data
  interpolate: {enabled: true, target_shape: [256, 256], method: linear}

reconstruction:
  method: gp_subset                       # gp_subset | wiener_filter | gaussian_process | neural_field
  resolution: [256, 256]
  parameters: {}

topology:
  backend: cubical                        # cubical | rips | alpha
  max_dimension: 2
  filtration: sublevel                    # sublevel (pits) | superlevel (peaks)
  persistence_threshold: 0.05

# Attractor detection is experimental and is not run by default.
# Add this section (or pass --attractor-method on the CLI) to opt in.
# attractors:
#   method: recurrence                    # recurrence | lyapunov | clustering
#   threshold: 0.1
#   parameters: {}
```

### Running the Pipeline

```python
from mneme.analysis.pipeline import MnemePipeline, default_config, merge_config

# Start from the defaults and overlay changes
config = merge_config(default_config('bioelectric'),
                      {'reconstruction': {'resolution': (128, 128)}})

pipe = MnemePipeline(config)

# Input: a dict with 'field' (and optionally 'observations' and
# 'positions' for reconstruction), a Field, or a numpy array
result = pipe.run({'field': voltage_field})

# A failed stage does not raise. Check the result.
if not result.success:
    print(result.failed_stages, result.errors)
result.stage_results      # per-stage summaries, including 'status' for failed stages
result.analysis_result    # AnalysisResult with whatever the other stages produced
```

Reconstruction is skipped, not faked, when there are no sparse observations. On the command line, `mneme analyze` prints each stage's status and exits non-zero if a stage fails; `--config` overlays a YAML file on the pipeline defaults.

## Parallel Processing

```python
from mneme.data import parallel

# Parallel pipeline for large datasets
parallel_pipeline = parallel.ParallelPipeline(
    pipeline=pipe,
    backend='multiprocessing',
    n_workers=8
)

# Process multiple files
results = parallel_pipeline.map(file_list)
```

## Data Validation

```python
from mneme.data import validation
from mneme.types import FieldDataSchema

# Define validation schema
schema = FieldDataSchema(
    shape=(None, 256, 256),  # Time dimension can vary
    dtype=np.float32,
    value_range=(-100, 100),  # mV
    required_metadata=['specimen_id', 'timestamp']
)

# Validate data
validator = validation.DataValidator(schema)
result = validator.validate(data)  # ValidationResult
result.is_valid, result.errors, result.warnings
```

## Caching and Optimization (roadmap)

```python
from mneme.data import cache

# Enable caching for expensive operations
@cache.memoize(cache_dir="cache/preprocessing/")
def expensive_preprocessing(field):
    return heavy_computation(field)

# LRU cache for frequent access
field_cache = cache.FieldCache(max_size="10GB")
field_cache.put("exp_001", processed_field)
```

## Monitoring and Logging

```python
from mneme.utils import monitoring

# Pipeline monitoring
monitor = monitoring.PipelineMonitor()
monitor.start()

with monitor.track_stage("preprocessing"):
    processed = preprocessor.transform(data)

# Get performance metrics
metrics = monitor.get_metrics()
print(f"Preprocessing durations: {metrics['stage_durations_s']}")
```

## Error Handling and Recovery (roadmap)

```python
from mneme.data import recovery

# Checkpoint-based recovery
pipeline_with_checkpoints = pipeline.DataPipeline(
    config=config,
    checkpoint_dir="checkpoints/",
    checkpoint_frequency=10  # Every 10 samples
)

try:
    results = pipeline_with_checkpoints.run(data)
except Exception as e:
    # Resume from last checkpoint
    results = pipeline_with_checkpoints.resume()
```

## Best Practices

1. **Data Versioning**: Track data and pipeline versions
2. **Reproducibility**: Set random seeds, log parameters
3. **Validation**: Validate data at each stage
4. **Documentation**: Document data sources and transformations
5. **Testing**: Unit test each pipeline component
6. **Monitoring**: Track performance and resource usage
7. **Error Handling**: Implement graceful failure and recovery