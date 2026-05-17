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

from .embedding import embed_trajectory, estimate_embedding_parameters
from .lyapunov import LyapunovResult, largest_lyapunov, lyapunov_spectrum
from .surrogates import SurrogateResult, iaaft_surrogates, surrogate_test
from .classify import classify_attractor, kaplan_yorke_dimension

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
    # Embedding
    "embed_trajectory",
    "estimate_embedding_parameters",
    # Lyapunov analysis
    "LyapunovResult",
    "largest_lyapunov",
    "lyapunov_spectrum",
    # Surrogate testing
    "SurrogateResult",
    "iaaft_surrogates",
    "surrogate_test",
    # Classification
    "classify_attractor",
    "kaplan_yorke_dimension",
]
