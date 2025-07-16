"""Information Field Theory implementations for field reconstruction."""

from typing import Optional, Tuple, Dict, Any, Union
import numpy as np
from abc import ABC, abstractmethod

from ..types import (
    Field, ReconstructionMethod, ReconstructionResult,
    Coordinates, FieldData
)


class BaseFieldReconstructor(ABC):
    """Abstract base class for field reconstruction methods."""
    
    def __init__(self, resolution: Tuple[int, int] = (256, 256)):
        """
        Initialize field reconstructor.
        
        Parameters
        ----------
        resolution : Tuple[int, int]
            Output field resolution (height, width)
        """
        self.resolution = resolution
        self.is_fitted = False
        self.observations = None
        self.positions = None
        
    @abstractmethod
    def fit(self, observations: np.ndarray, positions: np.ndarray) -> 'BaseFieldReconstructor':
        """
        Fit the reconstructor to observations.
        
        Parameters
        ----------
        observations : np.ndarray
            Observed field values, shape (n_observations,)
        positions : np.ndarray
            Observation positions, shape (n_observations, 2)
            
        Returns
        -------
        self : BaseFieldReconstructor
            Fitted reconstructor
        """
        raise NotImplementedError
        
    @abstractmethod
    def reconstruct(self, grid_points: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Reconstruct the continuous field.
        
        Parameters
        ----------
        grid_points : np.ndarray, optional
            Points at which to evaluate field, shape (n_points, 2)
            If None, uses regular grid based on resolution
            
        Returns
        -------
        field : np.ndarray
            Reconstructed field values
        """
        raise NotImplementedError
        
    @abstractmethod
    def uncertainty(self) -> np.ndarray:
        """
        Return reconstruction uncertainty estimates.
        
        Returns
        -------
        uncertainty : np.ndarray
            Uncertainty values at each grid point
        """
        raise NotImplementedError


class FieldReconstructor(BaseFieldReconstructor):
    """Main field reconstruction class with multiple backend methods."""
    
    def __init__(
        self, 
        method: Union[str, ReconstructionMethod] = ReconstructionMethod.IFT,
        resolution: Tuple[int, int] = (256, 256),
        **kwargs
    ):
        """
        Initialize field reconstructor with specified method.
        
        Parameters
        ----------
        method : str or ReconstructionMethod
            Reconstruction method to use
        resolution : Tuple[int, int]
            Output field resolution
        **kwargs
            Additional method-specific parameters
        """
        super().__init__(resolution)
        self.method = ReconstructionMethod(method) if isinstance(method, str) else method
        self.method_params = kwargs
        self._backend = None
        self._initialize_backend()
        
    def _initialize_backend(self):
        """Initialize the appropriate backend reconstructor."""
        if self.method == ReconstructionMethod.IFT:
            self._backend = IFTReconstructor(self.resolution, **self.method_params)
        elif self.method == ReconstructionMethod.GAUSSIAN_PROCESS:
            self._backend = GaussianProcessReconstructor(self.resolution, **self.method_params)
        elif self.method == ReconstructionMethod.NEURAL_FIELD:
            self._backend = NeuralFieldReconstructor(self.resolution, **self.method_params)
        else:
            raise ValueError(f"Unknown reconstruction method: {self.method}")
            
    def fit(self, observations: np.ndarray, positions: np.ndarray) -> 'FieldReconstructor':
        """Fit the reconstructor to observations."""
        self._backend.fit(observations, positions)
        self.is_fitted = True
        self.observations = observations
        self.positions = positions
        return self
        
    def reconstruct(self, grid_points: Optional[np.ndarray] = None) -> np.ndarray:
        """Reconstruct the continuous field."""
        if not self.is_fitted:
            raise RuntimeError("Reconstructor must be fitted before reconstruction")
        return self._backend.reconstruct(grid_points)
        
    def uncertainty(self) -> np.ndarray:
        """Return reconstruction uncertainty estimates."""
        if not self.is_fitted:
            raise RuntimeError("Reconstructor must be fitted before computing uncertainty")
        return self._backend.uncertainty()
        
    def fit_reconstruct(
        self, 
        observations: np.ndarray, 
        positions: np.ndarray,
        grid_points: Optional[np.ndarray] = None
    ) -> ReconstructionResult:
        """
        Convenience method to fit and reconstruct in one call.
        
        Parameters
        ----------
        observations : np.ndarray
            Observed field values
        positions : np.ndarray
            Observation positions
        grid_points : np.ndarray, optional
            Points at which to evaluate field
            
        Returns
        -------
        result : ReconstructionResult
            Complete reconstruction result
        """
        import time
        start_time = time.time()
        
        self.fit(observations, positions)
        field_data = self.reconstruct(grid_points)
        uncertainty = self.uncertainty()
        
        computation_time = time.time() - start_time
        
        field = Field(
            data=field_data,
            resolution=self.resolution,
            metadata={"method": self.method.value}
        )
        
        return ReconstructionResult(
            field=field,
            uncertainty=uncertainty,
            method=self.method,
            parameters=self.method_params,
            computation_time=computation_time
        )


class IFTReconstructor(BaseFieldReconstructor):
    """Information Field Theory based reconstruction."""
    
    def __init__(
        self,
        resolution: Tuple[int, int] = (256, 256),
        power_spectrum_model: str = "power_law",
        correlation_length: float = 10.0,
        smoothness_prior: float = 2.0
    ):
        """
        Initialize IFT reconstructor.
        
        Parameters
        ----------
        resolution : Tuple[int, int]
            Output field resolution
        power_spectrum_model : str
            Power spectrum model ('power_law', 'gaussian', 'exponential')
        correlation_length : float
            Correlation length scale
        smoothness_prior : float
            Smoothness parameter
        """
        super().__init__(resolution)
        self.power_spectrum_model = power_spectrum_model
        self.correlation_length = correlation_length
        self.smoothness_prior = smoothness_prior
        
    def fit(self, observations: np.ndarray, positions: np.ndarray) -> 'IFTReconstructor':
        """Fit IFT model to observations."""
        self.observations = observations
        self.positions = positions
        self.n_observations = len(observations)
        
        # Set up response operator matrix
        self._setup_response_operator()
        
        # Compute prior covariance
        self._compute_prior_covariance()
        
        # Compute posterior parameters
        self._compute_posterior()
        
        self.is_fitted = True
        return self
    
    def _setup_response_operator(self):
        """Set up the response operator matrix."""
        # Create regular grid for field reconstruction
        self.grid_points = create_grid_points(self.resolution)
        self.n_grid = len(self.grid_points)
        
        # Compute response matrix R[i,j] = response at observation i due to grid point j
        self.R = np.zeros((self.n_observations, self.n_grid))
        
        for i, obs_pos in enumerate(self.positions):
            for j, grid_pos in enumerate(self.grid_points):
                # Simple interpolation kernel (can be improved)
                dist = np.linalg.norm(obs_pos - grid_pos)
                if dist < self.correlation_length:
                    self.R[i, j] = np.exp(-dist**2 / (2 * self.correlation_length**2))
    
    def _compute_prior_covariance(self):
        """Compute prior covariance matrix."""
        # Compute prior covariance matrix for grid points
        self.S = np.zeros((self.n_grid, self.n_grid))
        
        for i, pos_i in enumerate(self.grid_points):
            for j, pos_j in enumerate(self.grid_points):
                dist = np.linalg.norm(pos_i - pos_j)
                
                if self.power_spectrum_model == "power_law":
                    # Power-law covariance
                    self.S[i, j] = np.exp(-dist**2 / (2 * self.correlation_length**2))
                elif self.power_spectrum_model == "gaussian":
                    # Gaussian covariance
                    self.S[i, j] = np.exp(-dist**2 / (2 * self.correlation_length**2))
                elif self.power_spectrum_model == "exponential":
                    # Exponential covariance
                    self.S[i, j] = np.exp(-dist / self.correlation_length)
        
        # Add regularization
        self.S += 1e-6 * np.eye(self.n_grid)
    
    def _compute_posterior(self):
        """Compute posterior mean and covariance."""
        # Observation noise covariance
        noise_var = 0.1  # Can be estimated or set as parameter
        N = noise_var * np.eye(self.n_observations)
        
        # Information matrix (inverse of posterior covariance)
        try:
            S_inv = np.linalg.inv(self.S)
        except np.linalg.LinAlgError:
            # Use pseudo-inverse if singular
            S_inv = np.linalg.pinv(self.S)
        
        # Posterior covariance: (S^-1 + R^T N^-1 R)^-1
        info_matrix = S_inv + self.R.T @ np.linalg.inv(N) @ self.R
        
        try:
            self.posterior_cov = np.linalg.inv(info_matrix)
        except np.linalg.LinAlgError:
            self.posterior_cov = np.linalg.pinv(info_matrix)
        
        # Posterior mean: D * R^T * N^-1 * d
        self.posterior_mean = self.posterior_cov @ self.R.T @ np.linalg.inv(N) @ self.observations
        
    def reconstruct(self, grid_points: Optional[np.ndarray] = None) -> np.ndarray:
        """Reconstruct field using IFT."""
        if not self.is_fitted:
            raise RuntimeError("IFT reconstructor must be fitted first")
        
        if grid_points is None:
            # Use regular grid
            field_1d = self.posterior_mean
            field_2d = field_1d.reshape(self.resolution)
        else:
            # Interpolate to new grid points
            # For simplicity, use nearest neighbor interpolation
            from scipy.spatial import cKDTree
            tree = cKDTree(self.grid_points)
            _, indices = tree.query(grid_points)
            field_1d = self.posterior_mean[indices]
            field_2d = field_1d.reshape(self.resolution)
        
        return field_2d
        
    def uncertainty(self) -> np.ndarray:
        """Compute IFT uncertainty estimates."""
        if not self.is_fitted:
            raise RuntimeError("IFT reconstructor must be fitted first")
        
        # Uncertainty is the diagonal of posterior covariance
        uncertainty_1d = np.sqrt(np.diag(self.posterior_cov))
        uncertainty_2d = uncertainty_1d.reshape(self.resolution)
        
        return uncertainty_2d


class GaussianProcessReconstructor(BaseFieldReconstructor):
    """Gaussian Process based field reconstruction."""
    
    def __init__(
        self,
        resolution: Tuple[int, int] = (256, 256),
        kernel: str = "rbf",
        length_scale: float = 10.0,
        noise_level: float = 0.1
    ):
        """
        Initialize GP reconstructor.
        
        Parameters
        ----------
        resolution : Tuple[int, int]
            Output field resolution
        kernel : str
            Kernel type ('rbf', 'matern', 'periodic')
        length_scale : float
            Kernel length scale
        noise_level : float
            Observation noise level
        """
        super().__init__(resolution)
        self.kernel = kernel
        self.length_scale = length_scale
        self.noise_level = noise_level
        
    def fit(self, observations: np.ndarray, positions: np.ndarray) -> 'GaussianProcessReconstructor':
        """Fit Gaussian Process to observations."""
        self.observations = observations
        self.positions = positions
        self.n_observations = len(observations)
        
        # Compute kernel matrix
        self.K = self._compute_kernel_matrix(positions, positions)
        
        # Add noise to diagonal
        self.K += self.noise_level**2 * np.eye(self.n_observations)
        
        # Compute inverse (with regularization)
        try:
            self.K_inv = np.linalg.inv(self.K)
        except np.linalg.LinAlgError:
            # Add regularization if singular
            self.K += 1e-6 * np.eye(self.n_observations)
            self.K_inv = np.linalg.inv(self.K)
        
        # Precompute alpha for efficiency
        self.alpha = self.K_inv @ observations
        
        self.is_fitted = True
        return self
    
    def _compute_kernel_matrix(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """Compute kernel matrix between two sets of points."""
        from scipy.spatial.distance import cdist
        
        # Compute pairwise distances
        distances = cdist(X1, X2)
        
        if self.kernel == "rbf":
            # RBF kernel
            K = np.exp(-distances**2 / (2 * self.length_scale**2))
        elif self.kernel == "matern":
            # Matérn kernel (ν = 1.5)
            scaled_dist = distances * np.sqrt(3) / self.length_scale
            K = (1 + scaled_dist) * np.exp(-scaled_dist)
        elif self.kernel == "periodic":
            # Periodic kernel
            K = np.exp(-2 * np.sin(np.pi * distances / self.length_scale)**2)
        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")
        
        return K
        
    def reconstruct(self, grid_points: Optional[np.ndarray] = None) -> np.ndarray:
        """Reconstruct field using Gaussian Process."""
        if not self.is_fitted:
            raise RuntimeError("GP reconstructor must be fitted first")
        
        if grid_points is None:
            grid_points = create_grid_points(self.resolution)
        
        # Compute kernel matrix between grid points and observations
        K_star = self._compute_kernel_matrix(grid_points, self.positions)
        
        # Compute posterior mean
        mu = K_star @ self.alpha
        
        # Reshape to 2D if using regular grid
        if grid_points.shape[0] == self.resolution[0] * self.resolution[1]:
            mu = mu.reshape(self.resolution)
        
        # Store for uncertainty computation
        self._last_grid_points = grid_points
        self._last_K_star = K_star
        self._last_mu = mu
        
        return mu
        
    def uncertainty(self) -> np.ndarray:
        """Compute GP uncertainty (posterior variance)."""
        if not self.is_fitted:
            raise RuntimeError("GP reconstructor must be fitted first")
        
        if not hasattr(self, '_last_grid_points'):
            raise RuntimeError("Must call reconstruct() before uncertainty()")
        
        # Compute kernel matrix for grid points
        K_star_star = self._compute_kernel_matrix(self._last_grid_points, self._last_grid_points)
        
        # Compute posterior variance
        var = np.diag(K_star_star - self._last_K_star @ self.K_inv @ self._last_K_star.T)
        
        # Ensure non-negative variance
        var = np.maximum(var, 0)
        
        # Reshape to 2D if using regular grid
        if var.shape[0] == self.resolution[0] * self.resolution[1]:
            var = var.reshape(self.resolution)
        
        return np.sqrt(var)


class NeuralFieldReconstructor(BaseFieldReconstructor):
    """Neural field based reconstruction."""
    
    def __init__(
        self,
        resolution: Tuple[int, int] = (256, 256),
        hidden_dims: Tuple[int, ...] = (256, 128, 64),
        activation: str = "relu",
        positional_encoding_dims: int = 32
    ):
        """
        Initialize neural field reconstructor.
        
        Parameters
        ----------
        resolution : Tuple[int, int]
            Output field resolution
        hidden_dims : Tuple[int, ...]
            Hidden layer dimensions
        activation : str
            Activation function
        positional_encoding_dims : int
            Positional encoding dimensions
        """
        super().__init__(resolution)
        self.hidden_dims = hidden_dims
        self.activation = activation
        self.positional_encoding_dims = positional_encoding_dims
        
    def fit(self, observations: np.ndarray, positions: np.ndarray) -> 'NeuralFieldReconstructor':
        """Train neural field on observations."""
        import torch
        import torch.nn as nn
        import torch.optim as optim
        
        self.observations = observations
        self.positions = positions
        
        # Create neural network
        self.network = self._create_network()
        
        # Convert to torch tensors
        pos_tensor = torch.FloatTensor(positions)
        obs_tensor = torch.FloatTensor(observations)
        
        # Training
        optimizer = optim.Adam(self.network.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        for epoch in range(1000):  # Can be made configurable
            optimizer.zero_grad()
            
            # Forward pass
            pred = self.network(pos_tensor).squeeze()
            loss = criterion(pred, obs_tensor)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item():.6f}")
        
        self.is_fitted = True
        return self
    
    def _create_network(self):
        """Create neural network for field reconstruction."""
        import torch.nn as nn
        
        # Input dimension (2D positions)
        input_dim = 2
        
        # Add positional encoding
        if self.positional_encoding_dims > 0:
            input_dim += 2 * self.positional_encoding_dims
        
        # Create network layers
        layers = []
        
        # Input layer
        layers.append(nn.Linear(input_dim, self.hidden_dims[0]))
        layers.append(self._get_activation())
        
        # Hidden layers
        for i in range(len(self.hidden_dims) - 1):
            layers.append(nn.Linear(self.hidden_dims[i], self.hidden_dims[i + 1]))
            layers.append(self._get_activation())
        
        # Output layer
        layers.append(nn.Linear(self.hidden_dims[-1], 1))
        
        return nn.Sequential(*layers)
    
    def _get_activation(self):
        """Get activation function."""
        import torch.nn as nn
        
        if self.activation == "relu":
            return nn.ReLU()
        elif self.activation == "tanh":
            return nn.Tanh()
        elif self.activation == "sigmoid":
            return nn.Sigmoid()
        else:
            return nn.ReLU()
    
    def _positional_encoding(self, positions):
        """Apply positional encoding to positions."""
        import torch
        
        if self.positional_encoding_dims == 0:
            return positions
        
        # Create frequency bands
        freqs = 2.0 ** torch.arange(self.positional_encoding_dims).float()
        
        # Apply encoding
        encoded = []
        for i in range(positions.shape[1]):
            pos = positions[:, i:i+1]
            for freq in freqs:
                encoded.append(torch.sin(freq * pos))
                encoded.append(torch.cos(freq * pos))
        
        return torch.cat([positions] + encoded, dim=1)
        
    def reconstruct(self, grid_points: Optional[np.ndarray] = None) -> np.ndarray:
        """Reconstruct field using trained neural network."""
        if not self.is_fitted:
            raise RuntimeError("Neural field must be fitted first")
        
        import torch
        
        if grid_points is None:
            grid_points = create_grid_points(self.resolution)
        
        # Convert to tensor
        pos_tensor = torch.FloatTensor(grid_points)
        
        # Apply positional encoding
        if self.positional_encoding_dims > 0:
            pos_tensor = self._positional_encoding(pos_tensor)
        
        # Predict
        with torch.no_grad():
            pred = self.network(pos_tensor).squeeze().numpy()
        
        # Reshape to 2D if using regular grid
        if grid_points.shape[0] == self.resolution[0] * self.resolution[1]:
            pred = pred.reshape(self.resolution)
        
        return pred
        
    def uncertainty(self) -> np.ndarray:
        """Estimate uncertainty (e.g., via dropout or ensemble)."""
        if not self.is_fitted:
            raise RuntimeError("Neural field must be fitted first")
        
        # Simple uncertainty estimation by computing multiple forward passes
        # with dropout (if available) or by adding noise to weights
        
        # For now, return zeros as placeholder
        # This could be improved with proper uncertainty quantification
        return np.zeros(self.resolution)


# Utility functions
def create_grid_points(
    resolution: Tuple[int, int],
    bounds: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None
) -> np.ndarray:
    """
    Create regular grid points for field evaluation.
    
    Parameters
    ----------
    resolution : Tuple[int, int]
        Grid resolution (height, width)
    bounds : Tuple[Tuple[float, float], Tuple[float, float]], optional
        Spatial bounds ((x_min, x_max), (y_min, y_max))
        If None, uses unit square [0, 1] x [0, 1]
        
    Returns
    -------
    grid_points : np.ndarray
        Grid points, shape (height * width, 2)
    """
    if bounds is None:
        bounds = ((0.0, 1.0), (0.0, 1.0))
        
    x_range = np.linspace(bounds[0][0], bounds[0][1], resolution[1])
    y_range = np.linspace(bounds[1][0], bounds[1][1], resolution[0])
    
    xx, yy = np.meshgrid(x_range, y_range)
    grid_points = np.column_stack([xx.ravel(), yy.ravel()])
    
    return grid_points