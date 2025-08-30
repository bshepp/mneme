# Mneme Data Pipeline Documentation

> Accuracy note (MVP): Several sections below (quality module, feature extractor, parallel pipeline, monitoring, recovery) are illustrative/roadmap and not yet implemented in `src/`. Where code references non-existent modules, treat them as examples for future work.

## Overview

The Mneme data pipeline handles the flow of data from raw bioelectric measurements and synthetic generation through preprocessing, analysis, and final results. The pipeline is designed to be modular, reproducible, and scalable.

## Data Flow Architecture

```
Raw Data Sources          Preprocessing           Analysis              Results
================          =============           ========              =======
                                                                        
Bioelectric Images   -->  Denoising          -->  Field               --> Attractor
Voltage Maps        -->  Registration       -->  Reconstruction      --> Patterns
Gene Expression     -->  Normalization      -->  Topology Analysis   --> 
Synthetic Fields    -->  Interpolation      -->  Symbolic Regression --> Reports
                         Augmentation           Autoencoding          Visualizations
```

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
loader = loaders.BioelectricLoader(
    data_dir="data/raw/planarian/",
    file_pattern="*.h5",
    lazy_load=True  # Load data on demand
)

# Iterate through experiments
for experiment in loader:
    voltage_field = experiment.voltage_field
    metadata = experiment.metadata
```

### Stage 2: Quality Control (roadmap)

```python
from mneme.data import quality

# Quality assessment
qc = quality.QualityChecker()
report = qc.check_field(voltage_field)

# Check for:
# - Missing values
# - Outliers
# - Signal-to-noise ratio
# - Spatial resolution adequacy

if report.passed:
    processed_field = preprocess(voltage_field)
else:
    logger.warning(f"Quality check failed: {report.issues}")
```

### Stage 3: Preprocessing

```python
from mneme.data import preprocessors

# Create preprocessing pipeline
preprocessor = preprocessors.FieldPreprocessor([
    preprocessors.Denoiser(method='wavelet', threshold='soft'),
    preprocessors.Registrator(reference='first_frame'),
    preprocessors.Normalizer(method='z_score', per_frame=True),
    preprocessors.Interpolator(target_resolution=(256, 256))
])

# Apply preprocessing
processed = preprocessor.fit_transform(voltage_field)
```

#### Preprocessing Steps:

1. **Denoising**
   - Wavelet denoising for preserving edges
   - Gaussian filtering for smooth fields
   - Median filtering for impulse noise

2. **Registration**
   - Align temporal sequences
   - Correct for specimen movement
   - Maintain spatial correspondence

3. **Normalization**
   - Z-score normalization
   - Min-max scaling
   - Histogram equalization

4. **Interpolation**
   - Bicubic interpolation for upsampling
   - Gaussian process interpolation for missing data

### Stage 4: Feature Extraction (roadmap)

```python
from mneme.analysis import features

# Extract multi-scale features
extractor = features.FieldFeatureExtractor()
feature_dict = extractor.extract(processed_field)

# Features include:
# - Spatial gradients
# - Laplacian values
# - Local curvature
# - Texture descriptors
# - Frequency components
```

### Stage 5: Core Analysis

```python
from mneme.core import field_theory, topology
from mneme.models import autoencoders

# 1. Field reconstruction
reconstructor = field_theory.FieldReconstructor(method='ift')
continuous_field = reconstructor.fit_reconstruct(processed_field)

# 2. Topology analysis
# Cubical for 2D fields (default), or use Rips/Alpha with adapter
tda = topology.PersistentHomology()
persistence_diagrams = tda.compute_persistence(continuous_field)

# Point-cloud backends
pc = topology.field_to_point_cloud(continuous_field, method='peaks', percentile=95.0)
rips = topology.RipsComplex(max_dimension=1)
rips_diagrams = rips.compute_persistence(pc)

# 3. Latent space embedding
autoencoder = autoencoders.FieldAutoencoder(latent_dim=32)
latent_representation = autoencoder.encode(continuous_field)

# 4. Attractor detection (recurrence default; lyapunov/clustering also available)
detector = topology.AttractorDetector(method='recurrence')
attractors = detector.detect(latent_trajectory)
```

### Stage 6: Results Generation (roadmap)

```python
from mneme.analysis import results

# Generate comprehensive results
result_generator = results.ResultGenerator()
results = result_generator.compile({
    'raw_data': voltage_field,
    'processed_data': processed_field,
    'reconstruction': continuous_field,
    'topology': persistence_diagrams,
    'attractors': attractors,
    'latent_space': latent_representation
})

# Save results
results.save("experiments/results/exp_001/")
```

## Data Pipeline Configuration

### Configuration File (`config/pipeline.yaml`)

```yaml
pipeline:
  name: "standard_bioelectric_pipeline"
  version: "1.0"
  
  stages:
    ingestion:
      loader: "BioelectricLoader"
      params:
        lazy_load: true
        cache_size: "2GB"
    
    quality_control:
      checks:
        - missing_values
        - outlier_detection
        - snr_threshold: 10.0
    
    preprocessing:
      steps:
        - name: "denoise"
          method: "wavelet"
          params:
            wavelet: "db4"
            level: 3
        
        - name: "normalize"
          method: "z_score"
          params:
            per_frame: true
        
        - name: "interpolate"
          method: "bicubic"
          params:
            target_shape: [256, 256]
    
    analysis:
      field_reconstruction:
        method: "ift"
        resolution: [512, 512]
      
      topology:
        max_dimension: 2
        filtration: "sublevel"
      
      attractors:
        method: "recurrence"
        threshold: 0.1
    
  output:
    format: "hdf5"
    compression: "gzip"
    save_intermediate: true
```

### Running the Pipeline

```python
from mneme.analysis import pipeline

# Load configuration
config = pipeline.load_config("config/pipeline.yaml")

# Create pipeline
pipe = pipeline.DataPipeline(config)

# Run on single dataset
results = pipe.run("data/raw/experiment_001.h5")

# Batch processing
results = pipe.run_batch(
    input_pattern="data/raw/*.h5",
    output_dir="results/",
    parallel=True,
    n_workers=4
)
```

## Parallel Processing (MVP)

```python
from mneme.data import parallel

# Parallel pipeline for large datasets
parallel_pipeline = parallel.ParallelPipeline(
    pipeline=pipe,
    backend='multiprocessing',  # MVP
    n_workers=8
)

# Process multiple files
results = parallel_pipeline.map(file_list)
```

## Data Validation

```python
from mneme.data import validation

# Define validation schema
schema = validation.FieldDataSchema(
    shape=(None, 256, 256),  # Time dimension can vary
    dtype=np.float32,
    value_range=(-100, 100),  # mV
    required_metadata=['specimen_id', 'timestamp']
)

# Validate data
validator = validation.DataValidator(schema)
is_valid, errors = validator.validate(data)
```

## Caching and Optimization

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

## Monitoring and Logging (MVP)

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