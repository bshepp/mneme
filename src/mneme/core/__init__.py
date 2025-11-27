"""Core computational modules for field analysis."""

from . import field_theory
from . import topology
from . import attractors

# Convenience exports for common classes
from .field_theory import (
    FieldReconstructor,
    SparseGPReconstructor,
    DenseIFTReconstructor,
    GaussianProcessReconstructor,
    NeuralFieldReconstructor,
    create_reconstructor,
    create_grid_points,
)

__all__ = [
    # Modules
    "field_theory",
    "topology", 
    "attractors",
    # Reconstructors
    "FieldReconstructor",
    "SparseGPReconstructor",
    "DenseIFTReconstructor",
    "GaussianProcessReconstructor",
    "NeuralFieldReconstructor",
    "create_reconstructor",
    "create_grid_points",
]