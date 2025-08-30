# Mneme API Design Documentation

## Core API Philosophy

The Mneme API follows these principles:
- **Composability**: Small, focused functions that combine into complex pipelines
- **Type Safety**: Clear type hints and validation
- **Configurability**: Flexible parameters with sensible defaults
- **Reproducibility**: Deterministic operations with seed control

## Module APIs

> MVP note: Some classes shown below (e.g., rich attractor characterization, full models) are placeholders or partially implemented. Methods explicitly marked with `NotImplementedError` are roadmap.

### 1. Field Theory Module (`mneme.core.field_theory`) — MVP

```python
from mneme.core import field_theory

class FieldReconstructor:
    """Reconstruct continuous fields from discrete observations."""
    
    def __init__(self, method='ift', resolution=(256, 256)):
        """
        Parameters:
            method: Reconstruction method ('gaussian_process', 'ift', 'neural_field')
            resolution: Output field resolution
        """
    
    def fit(self, observations: np.ndarray, positions: np.ndarray) -> 'FieldReconstructor':
        """Fit the reconstructor to observations."""
    
    def reconstruct(self, grid_points: Optional[np.ndarray] = None) -> np.ndarray:
        """Reconstruct the continuous field."""
    
    def uncertainty(self) -> np.ndarray:
        """Return reconstruction uncertainty estimates."""

# Usage example
reconstructor = FieldReconstructor(method='ift')
reconstructor.fit(voltage_measurements, electrode_positions)
field = reconstructor.reconstruct()
uncertainty = reconstructor.uncertainty()
```

### 2. Topology Module (`mneme.core.topology`) — MVP

```python
from mneme.core import topology

class PersistentHomology:
    """Compute persistent homology of fields."""
    
    def __init__(self, max_dimension=2, filtration='sublevel'):
        """
        Parameters:
            max_dimension: Maximum homological dimension
            filtration: Type of filtration to use
        """
    
    def compute_persistence(self, field: np.ndarray) -> List[Diagram]:
        """Compute persistence diagrams."""
    
    def extract_features(self, diagrams: List[Diagram]) -> np.ndarray:
        """Extract topological features from diagrams."""

class AttractorDetector:
    """Detect and characterize attractors in dynamical fields (basic recurrence)."""
    
    def __init__(self, method='recurrence', threshold=0.1):
        """
        Parameters:
            method: Detection method ('recurrence', 'lyapunov', 'clustering')
            threshold: Detection threshold
        """
    
    def detect(self, trajectory: np.ndarray) -> List[Attractor]:
        """Detect attractors in phase space trajectory."""
    
    def characterize(self, attractor: Attractor) -> Dict[str, Any]:
        """Compute attractor properties (dimension, stability, basin)."""

### 2b. Point-cloud topology backends — MVP

```python
from mneme.core.topology import RipsComplex, AlphaComplex, field_to_point_cloud

# Convert 2D field to point cloud and run Rips
pc = field_to_point_cloud(field2d, method='peaks', percentile=95.0)
tda = RipsComplex(max_dimension=1)
diagrams = tda.compute_persistence(pc)
features = tda.extract_features(diagrams)
```
```

### 3. Models Module (`mneme.models`) — placeholders

```python
from mneme.models import autoencoders, symbolic

class FieldAutoencoder(nn.Module):
    """Placeholder VAE for field data (minimal)."""
    
    def __init__(self, input_shape, latent_dim=32, architecture='convolutional'):
        """
        Parameters:
            input_shape: Shape of input fields
            latent_dim: Latent space dimensionality
            architecture: Network architecture type
        """
    
    def encode(self, field: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode field to latent representation (mean, log_var)."""
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent representation to field."""
    
    def forward(self, field: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass returning reconstruction, mean, log_var."""

class SymbolicRegressor:
    """Placeholder symbolic regression interface."""
    
    def __init__(self, operators=['+', '-', '*', '/', 'sin', 'cos'], 
                 complexity_penalty=0.001):
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

### 4. Data Module (`mneme.data`) — MVP

```python
from mneme.data import loaders, generators, preprocessors

class BioelectricDataset(Dataset):
    """PyTorch dataset for bioelectric imaging data."""
    
    def __init__(self, data_dir: str, transform=None, 
                 time_window=None, normalize=True):
        """
        Parameters:
            data_dir: Directory containing bioelectric data
            transform: Optional data transformations
            time_window: Time range to load
            normalize: Whether to normalize values
        """
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Return sample containing voltage field, gene expression, metadata."""
    
    def get_temporal_sequence(self, idx: int, length: int) -> torch.Tensor:
        """Get temporal sequence of fields."""

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
            steps: Preprocessing steps to apply
        """
    
    def fit(self, fields: List[np.ndarray]) -> 'FieldPreprocessor':
        """Fit preprocessing parameters."""
    
    def transform(self, field: np.ndarray) -> np.ndarray:
        """Apply preprocessing to field."""
    
    def inverse_transform(self, field: np.ndarray) -> np.ndarray:
        """Reverse preprocessing (where possible)."""
```

### 5. Analysis Pipeline (`mneme.analysis.pipeline`) — MVP

```python
from mneme.analysis import pipeline

class MnemePipeline:
    """Complete analysis pipeline for field memory detection."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Parameters:
            config: Pipeline configuration dictionary
        """
    
    def add_stage(self, name: str, stage: Callable, 
                  inputs: List[str], outputs: List[str]) -> 'MnemePipeline':
        """Add processing stage to pipeline."""
    
    def run(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute full pipeline on data."""
    
    def run_stage(self, stage_name: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Run specific pipeline stage."""
    
    def visualize_flow(self) -> None:
        """Visualize pipeline structure."""

# Predefined pipeline configurations
def create_standard_pipeline() -> MnemePipeline:
    """Create standard analysis pipeline."""
    pipeline = MnemePipeline(config={
        'preprocessing': {'normalize': True, 'denoise': True},
        'reconstruction': {'method': 'ift', 'resolution': (256, 256)},
        'analysis': {'compute_topology': True, 'detect_attractors': True},
        'modeling': {'use_autoencoder': True, 'symbolic_regression': True}
    })
    return pipeline

def create_bioelectric_pipeline() -> MnemePipeline:
    """Bioelectric-focused defaults; thin wrapper over standard."""
    return MnemePipeline({
        'preprocessing': {'denoise': {'enabled': True}, 'normalize': {'enabled': True}, 'register': {'enabled': True}, 'interpolate': {'enabled': True}},
        'reconstruction': {'method': 'ift', 'resolution': (256, 256)},
        'topology': {'max_dimension': 2, 'filtration': 'sublevel'},
        'attractors': {'method': 'recurrence', 'threshold': 0.1}
    })
```

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
        """Plot phase space with detected attractors."""
    
    def create_dashboard(self, results: Dict[str, Any]) -> None:
        """Create interactive dashboard of results."""
```

## Usage Patterns

### Basic Field Analysis

```python
import mneme
from mneme.core import field_theory, topology
from mneme.analysis import pipeline, visualization

# Load data
data = mneme.data.load_bioelectric("path/to/data")

# Create and run pipeline
pipe = pipeline.create_standard_pipeline()
results = pipe.run(data)

# Visualize results
viz = visualization.FieldVisualizer()
viz.create_dashboard(results)
```

### Custom Pipeline

```python
# Define custom pipeline
pipe = MnemePipeline(config={'seed': 42})

# Add custom stages
pipe.add_stage(
    name='custom_filter',
    stage=lambda x: custom_filter_function(x['field']),
    inputs=['field'],
    outputs=['filtered_field']
)

pipe.add_stage(
    name='extract_features',
    stage=lambda x: extract_spatial_features(x['filtered_field']),
    inputs=['filtered_field'],
    outputs=['features']
)

# Run pipeline
results = pipe.run({'field': my_field_data})
```

### Batch Processing

```python
from mneme.data import BioelectricDataset
from torch.utils.data import DataLoader

# Create dataset and dataloader
dataset = BioelectricDataset("data/planarian/")
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Process batches
for batch in dataloader:
    fields = batch['voltage_field']
    results = pipe.run_batch(fields)
    # Save or aggregate results
```

## Error Handling

All API functions include proper error handling:

```python
try:
    reconstructor = FieldReconstructor(method='invalid_method')
except ValueError as e:
    print(f"Invalid method: {e}")

# Or with validation
from mneme.utils import validate_parameters

@validate_parameters
def process_field(field: np.ndarray, threshold: float = 0.1) -> np.ndarray:
    """Process field with automatic parameter validation."""
    return field[field > threshold]
```

## Configuration Management

```python
from mneme.utils import Config

# Load configuration
config = Config.from_yaml("config/experiment.yaml")

# Access nested values
reconstruction_method = config.get("reconstruction.method", default="ift")

# Update configuration
config.set("analysis.threshold", 0.15)
config.save("config/modified.yaml")
```