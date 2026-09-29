# Mneme API Design Documentation

## Core API Philosophy

The Mneme API follows these principles:
- **Composability**: Small, focused functions that combine into complex pipelines
- **Type Safety**: Clear type hints and validation
- **Configurability**: Flexible parameters with sensible defaults
- **Reproducibility**: Deterministic operations with seed control

## Module APIs

> Note: The signatures below are abbreviated; the generated [API reference](api/index.md) is authoritative. Every component belongs to a tier (core, frozen or experimental) listed in [SCOPE.md](SCOPE.md). Experimental components emit `mneme.ExperimentalWarning` when constructed, and their output should not be used as evidence for a scientific claim.

### 1. Field Theory Module (`mneme.core.field_theory`) — core

```python
from mneme.core import field_theory

class FieldReconstructor:
    """Reconstruct continuous fields from discrete observations."""
    
    def __init__(self, method='gp_subset', resolution=(256, 256), **kwargs):
        """
        Parameters:
            method: 'gp_subset' (default: a GP on a random subset of the observations),
                    'wiener_filter' (dense, small grids), 'gaussian_process' (every
                    observation), 'neural_field' (experimental). The old names 'ift',
                    'sparse_gp' and 'dense_ift' still work with a DeprecationWarning.
            resolution: Output field resolution
        """
    
    def fit(self, observations: np.ndarray, positions: np.ndarray) -> 'FieldReconstructor':
        """Fit the reconstructor to observations."""
    
    def reconstruct(self, grid_points: Optional[np.ndarray] = None) -> np.ndarray:
        """Reconstruct the continuous field."""
    
    def uncertainty(self) -> np.ndarray:
        """Return reconstruction uncertainty estimates (NotImplementedError for 'neural_field')."""

# Usage example
reconstructor = FieldReconstructor(method='gp_subset')
reconstructor.fit(voltage_measurements, electrode_positions)
field = reconstructor.reconstruct()
uncertainty = reconstructor.uncertainty()

# Or the factory, which defaults to 'gp_subset'
from mneme.core import create_reconstructor
reconstructor = create_reconstructor('gp_subset', resolution=(128, 128), n_subset=500)
```

### 2. Topology Module (`mneme.core.topology`) — core

```python
from mneme.core import topology

class PersistentHomology:
    """Compute persistent homology of fields."""
    
    def __init__(self, max_dimension=2, filtration='sublevel',
                 persistence_threshold=0.05, compute_cycles=False):
        """
        Parameters:
            max_dimension: Maximum homological dimension
            filtration: 'sublevel' tracks pits as the threshold rises;
                        'superlevel' tracks peaks (diagrams in units of the negated field)
            persistence_threshold: Minimum persistence to keep
            compute_cycles: Not implemented; True raises NotImplementedError
        """
    
    def compute_persistence(self, field: np.ndarray) -> List[PersistenceDiagram]:
        """Compute persistence diagrams. NaN fields raise. Without GUDHI only H0
        is computed and a RuntimeWarning is emitted."""
    
    def extract_features(self, diagrams: List[PersistenceDiagram]) -> np.ndarray:
        """Extract topological features from diagrams."""
```

### 2b. Point-cloud topology backends — core

```python
from mneme.core.topology import RipsComplex, AlphaComplex, field_to_point_cloud

# Convert 2D field to point cloud and run Rips
pc = field_to_point_cloud(field2d, method='peaks', percentile=95.0)
tda = RipsComplex(max_dimension=1)
diagrams = tda.compute_persistence(pc)
features = tda.extract_features(diagrams)
```

### 2c. Attractor detectors (`mneme.core.attractors`) — experimental

```python
from mneme.core.attractors import AttractorDetector

class AttractorDetector:
    """Locate dense or recurrent regions of a trajectory.

    Detectors cannot determine what kind of attractor a region is. Every
    Attractor they return has type AttractorType.UNDETERMINED.
    """
    
    def __init__(self, method='recurrence', threshold=0.1, **kwargs):
        """
        Parameters:
            method: Detection method ('recurrence', 'lyapunov', 'clustering')
            threshold: Detection threshold
        """
    
    def detect(self, trajectory: np.ndarray) -> List[Attractor]:
        """Locate regions in a phase space trajectory."""
    
    def characterize(self, attractor: Attractor, trajectory: np.ndarray) -> Dict[str, Any]:
        """Compute descriptive properties of a region."""
```

### 2d. Lyapunov tools (`mneme.core`) — frozen

```python
from mneme.core import largest_lyapunov, surrogate_test, classify_attractor
from mneme.core import lyapunov_spectrum, kaplan_yorke_dimension

res = largest_lyapunov(series, dt=0.01)                    # LyapunovResult; res.lambda1
sur = surrogate_test(series, statistic="lambda1", n=200, dt=0.01)  # n >= 39 at alpha 0.05
label = classify_attractor(res.lambda1, surrogate=sur, oscillatory=None)
# STRANGE only when sur.significant; UNDETERMINED for near-zero or negative
# estimates unless the caller asserts oscillatory=True/False.
spectrum = lyapunov_spectrum(trajectory, dt=0.01)          # exploratory; RuntimeWarning below 1000 points
d_ky = kaplan_yorke_dimension(spectrum)
```

Read [LYAPUNOV_OPERATING_RANGE.md](LYAPUNOV_OPERATING_RANGE.md) before using a number from these: the surrogate test needs about 4,000 points to have power, and λ₁ was off by tens of percent away from the conditions it was tuned on.

### 2e. Steady-state analysis (`mneme.analysis.steady_state`) — core

```python
from mneme.analysis.steady_state import assess_steady_state, count_distinct_states

# values: shape (n_times, n_cells), one value per cell per sample
report = assess_steady_state(values, times, rate_tolerance=1e-4, drift_tolerance=0.1)
report.settled                     # bool

# end states of several runs, each shape (n_cells,)
distinct = count_distinct_states([run[-1] for run in runs], threshold=1.0)
distinct.n_distinct
```

### 3. Models Module (`mneme.models`) — experimental

```python
from mneme.models import autoencoders, symbolic

class FieldAutoencoder(nn.Module):
    """Convolutional VAE for 2D field data. Experimental."""
    
    def __init__(self, input_shape, latent_dim=32, in_channels=1,
                 base_channels=32, architecture='standard', beta=1.0):
        """
        Parameters:
            input_shape: Shape of input fields (height, width), each divisible by 16
            latent_dim: Latent space dimensionality
            architecture: 'standard', 'deep' or 'residual'
            beta: β-VAE weight
        """
    
    def encode(self, field: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode field to latent representation (mean, log_var)."""
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent representation to field."""
    
    def forward(self, field: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass returning reconstruction, mean, log_var."""

class SymbolicRegressor:
    """PySR wrapper; falls back to linear regression without PySR. Experimental."""
    
    def __init__(self, operators=None, complexity_penalty=0.001,
                 niterations=100, random_state=None, **kwargs):
        """
        Parameters:
            operators: Allowed mathematical operators
            complexity_penalty: Penalty for equation complexity
        """
    
    def fit(self, X: np.ndarray, y: np.ndarray, 
            variable_names: Optional[List[str]] = None) -> 'SymbolicRegressor':
        """Fit symbolic equations to data."""
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using discovered equations."""
    
    def get_equations(self) -> List[str]:
        """Return discovered equations as strings."""
```

### 4. Data Module (`mneme.data`) — core

```python
from mneme.data import loaders, generators, preprocessors

"""
Note: Loading goes through `mneme.data.loaders.create_data_loader` and the
BETSE loader (`mneme.data.betse_loader.load_betse_cells`, `betse_to_field`).
There is no `BioelectricDataset` class.
"""

class SyntheticFieldGenerator:
    """Generate synthetic field data for testing."""
    
    def __init__(self, field_type='gaussian_random', seed=None):
        """
        Parameters:
            field_type: Type of field to generate
            seed: Random seed for reproducibility
        """
    
    def generate_static(self, shape: Tuple[int, ...], 
                       parameters: Dict[str, Any]) -> np.ndarray:
        """Generate static field."""
    
    def generate_dynamic(self, shape: Tuple[int, ...], 
                        timesteps: int, 
                        parameters: Dict[str, Any]) -> np.ndarray:
        """Generate time-evolving field."""
    
    def add_noise(self, field: np.ndarray, noise_level: float) -> np.ndarray:
        """Add realistic noise to field."""

class FieldPreprocessor:
    """Preprocess field data for analysis."""
    
    def __init__(self, steps=['denoise', 'normalize', 'register']):
        """
        Parameters:
            steps: Step names ('denoise', 'normalize', 'register', 'interpolate'),
                   or (name, params) tuples
        """
    
    def fit(self, data: np.ndarray) -> 'FieldPreprocessor':
        """Fit preprocessing parameters."""
    
    def transform(self, data: np.ndarray) -> np.ndarray:
        """Apply preprocessing to field."""
    
    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """Fit and apply."""
```

### 5. Analysis Pipeline (`mneme.analysis.pipeline`) — core

```python
from mneme.analysis import pipeline

class MnemePipeline:
    """Complete analysis pipeline for field memory detection."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Parameters:
            config: Pipeline configuration dictionary (keys: 'preprocessing',
                    'reconstruction', 'topology', and optionally 'attractors')
        """
    
    def add_stage(self, name: str, stage_func: Callable, inputs: List[str],
                  outputs: List[str], enabled: bool = True) -> 'MnemePipeline':
        """Add a custom stage. When any custom stages exist they replace the
        default preprocessing stage; the configured topology, reconstruction
        and attractor components still run on the resulting data."""
    
    def run(self, data: Union[Dict[str, Any], Field, np.ndarray]) -> PipelineResult:
        """Execute full pipeline on data."""

@dataclass
class PipelineResult:
    success: bool                 # False when any stage failed
    execution_time: float
    stage_results: Dict[str, Any] # per-stage summaries; failed stages carry 'status': 'failed'
    analysis_result: Optional[AnalysisResult]  # whatever the other stages produced
    errors: Optional[List[str]]
    failed_stages: List[str]

# Predefined configurations
def default_config(pipeline: str = 'standard') -> Dict[str, Any]:
    """Fresh copy of the 'standard' or 'bioelectric' defaults. Both run
    preprocessing, reconstruction and topology; neither runs attractor detection."""

def merge_config(base: Dict[str, Any], overrides: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Recursively overlay overrides on base, returning a new dict."""

def create_standard_pipeline(config=None) -> MnemePipeline:
    """MnemePipeline(default_config('standard')) unless a non-empty config is given."""

def create_bioelectric_pipeline(config=None) -> MnemePipeline:
    """MnemePipeline(default_config('bioelectric')) unless a non-empty config is given."""
```

Reconstruction runs only when the input dict has `'observations'` and `'positions'`; otherwise the stage is reported as skipped. Attractor detection runs only when an `'attractors'` section is present and the field is a temporal (3D) sequence.

### 6. Visualization Module (`mneme.analysis.visualization`)

```python
from mneme.analysis import visualization

class FieldVisualizer:
    """Visualize fields and analysis results."""
    
    def __init__(self, style='publication', figsize=(10, 8)):
        """
        Parameters:
            style: Plotting style preset
            figsize: Default figure size
        """
    
    def plot_field(self, field: np.ndarray, title: str = None, 
                   colormap: str = 'viridis', **kwargs) -> plt.Figure:
        """Plot 2D field with customizable appearance."""
    
    def plot_field_sequence(self, fields: List[np.ndarray], 
                           fps: int = 10) -> animation.FuncAnimation:
        """Create animation of field evolution."""
    
    def plot_persistence_diagram(self, diagram: Diagram, 
                                ax: Optional[plt.Axes] = None) -> plt.Figure:
        """Plot topological persistence diagram."""
    
    def plot_attractor_portrait(self, trajectory: np.ndarray, 
                               attractors: List[Attractor]) -> plt.Figure:
        """Plot phase space with detected regions."""
    
    def create_analysis_dashboard(self, result: AnalysisResult,
                                  save_path: Optional[str] = None) -> plt.Figure:
        """Create a dashboard figure from an AnalysisResult."""
```

## Usage Patterns

### Basic Field Analysis

```python
from mneme.data.betse_loader import betse_to_field
from mneme.analysis import pipeline, visualization

# Load data (BETSE output interpolated to a grid)
field = betse_to_field("path/to/Vmem2D_TextExport/", resolution=(64, 64))

# Create and run pipeline
pipe = pipeline.create_bioelectric_pipeline()
result = pipe.run({'field': field.data[-1]})   # one 2D frame
if not result.success:
    print(result.failed_stages, result.errors)

# Visualize results
viz = visualization.FieldVisualizer()
viz.create_analysis_dashboard(result.analysis_result)
```

### Custom Pipeline

```python
# Custom stages replace the default preprocessing stage
pipe = MnemePipeline(config={'topology': {'max_dimension': 1}})

pipe.add_stage(
    name='custom_filter',
    stage_func=lambda x: custom_filter_function(x['field']),
    inputs=['field'],
    outputs=['processed_field']   # 'processed_field' is what topology analyses
)

# Run pipeline
result = pipe.run({'field': my_field_data})
```

### Batch Processing

```python
from mneme.data.loaders import create_data_loader
from mneme.data.parallel import ParallelPipeline

# Iterate a directory of files
loader = create_data_loader("data/planarian/", loader_type="bioelectric")
for item in loader:
    result = pipe.run({'field': item['voltage_field']})

# Or process a list of files in parallel
results = ParallelPipeline(pipe, backend='multiprocessing', n_workers=4).map(file_list)
```

## Error Handling

Constructors validate their arguments:

```python
try:
    reconstructor = FieldReconstructor(method='invalid_method')
except ValueError as e:
    print(f"Invalid method: {e}")
```

Inside a pipeline, a failing stage does not raise: `PipelineResult.success` is False, the stage is named in `failed_stages`, and its message is in `errors`.

## Configuration Management

```python
from mneme.utils import Config

# Load configuration
config = Config.from_yaml("config/experiment.yaml")

# Access nested values
reconstruction_method = config.get("reconstruction.method", default="gp_subset")

# Update configuration
config.set("analysis.threshold", 0.15)
config.save("config/modified.yaml")
```