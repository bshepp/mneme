"""Type definitions and data schemas for Mneme."""

from typing import Dict, List, Tuple, Optional, Union, Any, Callable, Protocol
from dataclasses import dataclass
from enum import Enum
import numpy as np
import torch
from pydantic import BaseModel, Field, field_validator
import numpy.typing as npt

# Type aliases
ArrayLike = Union[np.ndarray, torch.Tensor, List[float]]
Shape = Tuple[int, ...]
Coordinates = npt.NDArray[np.float64]  # Shape: (n_points, n_dims)
FieldData = npt.NDArray[np.float64]  # Shape: (height, width) or (time, height, width)
TimeSeries = npt.NDArray[np.float64]  # Shape: (time_steps, ...)

# Enums
class ReconstructionMethod(str, Enum):
    """Available field reconstruction methods."""
    IFT = "ift"
    GAUSSIAN_PROCESS = "gaussian_process"
    NEURAL_FIELD = "neural_field"

class PreprocessingStep(str, Enum):
    """Preprocessing pipeline steps."""
    DENOISE = "denoise"
    NORMALIZE = "normalize"
    REGISTER = "register"
    INTERPOLATE = "interpolate"

class FiltrationMethod(str, Enum):
    """Topological filtration methods."""
    SUBLEVEL = "sublevel"
    SUPERLEVEL = "superlevel"

class AttractorType(str, Enum):
    """Types of dynamical attractors."""
    FIXED_POINT = "fixed_point"
    LIMIT_CYCLE = "limit_cycle"
    STRANGE = "strange"
    QUASI_PERIODIC = "quasi_periodic"

# Data classes
@dataclass
class BioelectricMeasurement:
    """Single bioelectric measurement."""
    voltage: float
    position: Tuple[float, float]
    timestamp: float
    electrode_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class Field:
    """Continuous or discrete field representation."""
    data: np.ndarray
    coordinates: Optional[np.ndarray] = None
    resolution: Optional[Tuple[int, int]] = None
    bounds: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None
    metadata: Optional[Dict[str, Any]] = None
    
    @property
    def shape(self) -> Shape:
        return self.data.shape
    
    @property
    def ndim(self) -> int:
        return self.data.ndim

@dataclass
class PersistenceDiagram:
    """Topological persistence diagram."""
    points: np.ndarray  # Shape: (n_points, 2) with columns [birth, death]
    dimension: int
    threshold: Optional[float] = None
    
    @property
    def persistence(self) -> np.ndarray:
        """Compute persistence (death - birth) for each feature."""
        return self.points[:, 1] - self.points[:, 0]

@dataclass
class Attractor:
    """Dynamical attractor characterization."""
    type: AttractorType
    center: np.ndarray
    basin_size: float
    dimension: Optional[float] = None
    lyapunov_exponents: Optional[np.ndarray] = None
    stability: Optional[float] = None
    trajectory_indices: Optional[List[int]] = None

# Pydantic models for validation
class FieldDataSchema(BaseModel):
    """Schema for validating field data."""
    shape: Tuple[Optional[int], ...]
    dtype: str = "float64"
    value_range: Optional[Tuple[float, float]] = None
    required_metadata: List[str] = []
    
    @field_validator("shape")
    @classmethod
    def validate_shape(cls, v):
        if len(v) not in [2, 3]:
            raise ValueError("Field must be 2D or 3D (with time)")
        return v

class ExperimentConfig(BaseModel):
    """Experiment configuration schema."""
    name: str
    description: Optional[str] = None
    data_path: str
    output_dir: str
    preprocessing: Dict[str, Any]
    reconstruction: Dict[str, Any]
    analysis: Dict[str, Any]
    random_seed: int = 42
    
    class Config:
        extra = "allow"  # Allow additional fields

class PipelineStage(BaseModel):
    """Pipeline stage configuration."""
    name: str
    enabled: bool = True
    inputs: List[str]
    outputs: List[str]
    parameters: Dict[str, Any] = {}

# Protocol classes for type checking
class Reconstructor(Protocol):
    """Protocol for field reconstruction methods."""
    
    def fit(self, observations: np.ndarray, positions: np.ndarray) -> None:
        """Fit the reconstructor to observations."""
        ...
    
    def reconstruct(self, grid_points: Optional[np.ndarray] = None) -> np.ndarray:
        """Reconstruct the continuous field."""
        ...

class Preprocessor(Protocol):
    """Protocol for preprocessing steps."""
    
    def fit(self, data: np.ndarray) -> None:
        """Fit preprocessing parameters."""
        ...
    
    def transform(self, data: np.ndarray) -> np.ndarray:
        """Apply preprocessing transformation."""
        ...

class TopologyAnalyzer(Protocol):
    """Protocol for topology analysis methods."""
    
    def compute_persistence(self, field: np.ndarray) -> List[PersistenceDiagram]:
        """Compute persistence diagrams."""
        ...

# Result types
@dataclass
class ReconstructionResult:
    """Result from field reconstruction."""
    field: Field
    uncertainty: Optional[np.ndarray] = None
    method: Optional[ReconstructionMethod] = None
    parameters: Optional[Dict[str, Any]] = None
    computation_time: Optional[float] = None

@dataclass
class TopologyResult:
    """Result from topology analysis."""
    diagrams: List[PersistenceDiagram]
    features: Optional[np.ndarray] = None
    cycles: Optional[List[np.ndarray]] = None
    computation_time: Optional[float] = None

@dataclass
class AnalysisResult:
    """Complete analysis result."""
    experiment_id: str
    timestamp: str
    raw_data: Field
    processed_data: Optional[Field] = None
    reconstruction: Optional[ReconstructionResult] = None
    topology: Optional[TopologyResult] = None
    attractors: Optional[List[Attractor]] = None
    symbolic_equations: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None

# Type guards
def is_2d_field(data: ArrayLike) -> bool:
    """Check if data represents a 2D field."""
    return hasattr(data, "ndim") and data.ndim == 2

def is_temporal_field(data: ArrayLike) -> bool:
    """Check if data represents a temporal field sequence."""
    return hasattr(data, "ndim") and data.ndim == 3

def validate_field_data(data: ArrayLike, schema: FieldDataSchema) -> Tuple[bool, List[str]]:
    """Validate field data against schema."""
    errors = []
    
    # Check shape
    if hasattr(data, "shape"):
        if len(data.shape) != len(schema.shape):
            errors.append(f"Expected {len(schema.shape)}D data, got {len(data.shape)}D")
        else:
            for i, (actual, expected) in enumerate(zip(data.shape, schema.shape)):
                if expected is not None and actual != expected:
                    errors.append(f"Dimension {i}: expected {expected}, got {actual}")
    
    # Check dtype
    if hasattr(data, "dtype") and str(data.dtype) != schema.dtype:
        errors.append(f"Expected dtype {schema.dtype}, got {data.dtype}")
    
    # Check value range
    if schema.value_range is not None and hasattr(data, "min") and hasattr(data, "max"):
        data_min, data_max = float(data.min()), float(data.max())
        range_min, range_max = schema.value_range
        if data_min < range_min or data_max > range_max:
            errors.append(f"Values outside range [{range_min}, {range_max}]: [{data_min}, {data_max}]")
    
    return len(errors) == 0, errors