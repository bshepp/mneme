"""Information Field Theory implementations for field reconstruction.

This module provides multiple approaches for reconstructing continuous fields
from sparse observations, with scalable defaults for large field sizes.
"""

from typing import Optional, Tuple, Dict, Any, Union
import warnings
import numpy as np
from abc import ABC, abstractmethod

from ..types import (
    Field, ReconstructionMethod, ReconstructionResult,
    Coordinates, FieldData
)

# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------

#: Number of grid points to predict per batch in SparseGPReconstructor.
#: Controls the memory/speed trade-off when evaluating the GP on a dense grid.
GP_PREDICTION_BATCH_SIZE: int = 10_000

#: Maximum grid size (height*width) before DenseIFTReconstructor emits a
#: memory warning.  64x64 = 4096 points → covariance matrix is ~128 MB.
DENSE_IFT_MAX_RECOMMENDED_SIZE: int = 64 * 64


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
    """Main field reconstruction class with multiple backend methods.
    
    This is the primary interface for field reconstruction. It dispatches
    to specialized backend implementations based on the chosen method.
    
    Parameters
    ----------
    method : str or ReconstructionMethod
        Reconstruction method to use:
        - 'ift': Sparse GP-based IFT (default, scalable)
        - 'dense_ift': Dense matrix IFT (exact but O(n³), for small fields only)
        - 'gaussian_process': Standard GP reconstruction
        - 'neural_field': Neural network-based reconstruction
    resolution : Tuple[int, int]
        Output field resolution (height, width)
    **kwargs
        Additional method-specific parameters
        
    Examples
    --------
    >>> from mneme.core.field_theory import FieldReconstructor
    >>> import numpy as np
    >>> 
    >>> # Create reconstructor (uses scalable Sparse GP by default)
    >>> reconstructor = FieldReconstructor(resolution=(128, 128))
    >>> 
    >>> # Fit to sparse observations
    >>> positions = np.random.rand(100, 2)  # 100 observation points
    >>> observations = np.sin(2 * np.pi * positions[:, 0])  # Some field values
    >>> reconstructor.fit(observations, positions)
    >>> 
    >>> # Reconstruct full field
    >>> field = reconstructor.reconstruct()
    >>> uncertainty = reconstructor.uncertainty()
    """
    
    def __init__(
        self, 
        method: Union[str, ReconstructionMethod] = ReconstructionMethod.IFT,
        resolution: Tuple[int, int] = (256, 256),
        **kwargs
    ):
        super().__init__(resolution)
        
        # Handle string method names
        if isinstance(method, str):
            method_lower = method.lower()
            if method_lower == 'dense_ift':
                self.method = ReconstructionMethod.IFT
                self._use_dense = True
            else:
                self.method = ReconstructionMethod(method_lower)
                self._use_dense = False
        else:
            self.method = method
            self._use_dense = kwargs.pop('use_dense', False)
        
        self.method_params = kwargs
        self._backend = None
        self._initialize_backend()
        
    def _initialize_backend(self):
        """Initialize the appropriate backend reconstructor."""
        if self.method == ReconstructionMethod.IFT:
            if self._use_dense:
                # Use original dense IFT (for small fields or exact computation)
                self._backend = DenseIFTReconstructor(self.resolution, **self.method_params)
            else:
                # Use scalable Sparse GP (default)
                self._backend = SparseGPReconstructor(self.resolution, **self.method_params)
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
        uncertainty_data = self.uncertainty()
        
        computation_time = time.time() - start_time
        
        field = Field(
            data=field_data,
            resolution=self.resolution,
            metadata={"method": self.method.value}
        )
        
        return ReconstructionResult(
            field=field,
            uncertainty=uncertainty_data,
            method=self.method,
            parameters=self.method_params,
            computation_time=computation_time
        )


class SparseGPReconstructor(BaseFieldReconstructor):
    """Sparse Gaussian Process reconstructor using inducing points.
    
    This is a scalable approximation to full GP regression that uses
    a subset of inducing points to approximate the full covariance
    structure. Complexity is O(nm²) instead of O(n³) where m << n.
    
    This is the DEFAULT method for IFT reconstruction as it scales
    to large field sizes while maintaining good accuracy.
    
    Parameters
    ----------
    resolution : Tuple[int, int]
        Output field resolution
    n_inducing : int
        Number of inducing points. More points = better accuracy but slower.
        Default 500 works well for most bioelectric fields.
    kernel : str
        Kernel type: 'rbf', 'matern', 'exponential'
    length_scale : float
        Kernel length scale (correlation length)
    noise_level : float
        Observation noise level
    optimize_hyperparameters : bool
        Whether to optimize kernel hyperparameters during fitting
    random_state : int, optional
        Random seed for inducing point selection
    """
    
    def __init__(
        self,
        resolution: Tuple[int, int] = (256, 256),
        n_inducing: int = 500,
        kernel: str = "rbf",
        length_scale: float = 0.1,
        noise_level: float = 0.1,
        optimize_hyperparameters: bool = True,
        random_state: Optional[int] = None,
        # Legacy parameter mapping from IFT
        correlation_length: Optional[float] = None,
        power_spectrum_model: Optional[str] = None,
    ):
        super().__init__(resolution)
        
        self.n_inducing = n_inducing
        self.optimize_hyperparameters = optimize_hyperparameters
        self.random_state = random_state
        
        # Map legacy IFT parameters
        if correlation_length is not None:
            # Convert from pixel-space to normalized [0,1] space
            length_scale = correlation_length / max(resolution)
        self.length_scale = length_scale
        
        if power_spectrum_model is not None:
            # Map power spectrum model to kernel type
            kernel_map = {
                'power_law': 'rbf',
                'gaussian': 'rbf',
                'exponential': 'matern',
            }
            kernel = kernel_map.get(power_spectrum_model, 'rbf')
        self.kernel = kernel
        
        self.noise_level = noise_level
        
        # Internal state
        self._gp = None
        self._grid_points = None
        self._last_predictions = None
        self._last_std = None
        
    def _create_kernel(self):
        """Create sklearn kernel based on settings."""
        from sklearn.gaussian_process.kernels import (
            RBF, Matern, ExpSineSquared, WhiteKernel, ConstantKernel
        )
        
        # Base kernel
        if self.kernel == "rbf":
            base_kernel = RBF(length_scale=self.length_scale)
        elif self.kernel == "matern":
            base_kernel = Matern(length_scale=self.length_scale, nu=1.5)
        elif self.kernel == "exponential":
            base_kernel = Matern(length_scale=self.length_scale, nu=0.5)
        elif self.kernel == "periodic":
            base_kernel = ExpSineSquared(length_scale=self.length_scale, periodicity=1.0)
        else:
            base_kernel = RBF(length_scale=self.length_scale)
        
        # Add amplitude and noise
        kernel = ConstantKernel(1.0) * base_kernel + WhiteKernel(noise_level=self.noise_level**2)
        
        return kernel
        
    def fit(self, observations: np.ndarray, positions: np.ndarray) -> 'SparseGPReconstructor':
        """Fit Sparse GP to observations using inducing points.
        
        For efficiency, we subsample the observations to create inducing points
        when the number of observations exceeds n_inducing.
        """
        from sklearn.gaussian_process import GaussianProcessRegressor
        
        self.observations = observations
        self.positions = positions
        n_obs = len(observations)
        
        # Normalize positions to [0, 1] for numerical stability
        self._pos_min = positions.min(axis=0)
        self._pos_max = positions.max(axis=0)
        self._pos_range = self._pos_max - self._pos_min
        self._pos_range[self._pos_range == 0] = 1.0  # Avoid division by zero
        
        positions_norm = (positions - self._pos_min) / self._pos_range
        
        # Select inducing points if we have more observations than n_inducing
        if n_obs > self.n_inducing:
            rng = np.random.RandomState(self.random_state)
            inducing_idx = rng.choice(n_obs, size=self.n_inducing, replace=False)
            X_train = positions_norm[inducing_idx]
            y_train = observations[inducing_idx]
        else:
            X_train = positions_norm
            y_train = observations
        
        # Normalize observations
        self._y_mean = y_train.mean()
        self._y_std = y_train.std()
        if self._y_std == 0:
            self._y_std = 1.0
        y_train_norm = (y_train - self._y_mean) / self._y_std
        
        # Create and fit GP
        kernel = self._create_kernel()
        
        self._gp = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=5 if self.optimize_hyperparameters else 0,
            normalize_y=False,  # We already normalized
            random_state=self.random_state,
        )
        
        self._gp.fit(X_train, y_train_norm)
        
        self.is_fitted = True
        return self
    
    def reconstruct(self, grid_points: Optional[np.ndarray] = None) -> np.ndarray:
        """Reconstruct field using the fitted Sparse GP."""
        if not self.is_fitted:
            raise RuntimeError("SparseGP reconstructor must be fitted first")
        
        if grid_points is None:
            grid_points = create_grid_points(self.resolution)
        
        self._grid_points = grid_points
        
        # Normalize grid points to same space as training data
        grid_norm = (grid_points - self._pos_min) / self._pos_range
        
        # Predict in batches for memory efficiency
        batch_size = GP_PREDICTION_BATCH_SIZE
        n_points = len(grid_norm)
        
        predictions = np.zeros(n_points)
        stds = np.zeros(n_points)
        
        for i in range(0, n_points, batch_size):
            batch = grid_norm[i:i+batch_size]
            pred, std = self._gp.predict(batch, return_std=True)
            predictions[i:i+batch_size] = pred
            stds[i:i+batch_size] = std
        
        # Denormalize predictions
        predictions = predictions * self._y_std + self._y_mean
        stds = stds * self._y_std
        
        self._last_predictions = predictions
        self._last_std = stds
        
        # Reshape to 2D if using regular grid
        if n_points == self.resolution[0] * self.resolution[1]:
            return predictions.reshape(self.resolution)
        
        return predictions
    
    def uncertainty(self) -> np.ndarray:
        """Return uncertainty estimates from Sparse GP."""
        if not self.is_fitted:
            raise RuntimeError("SparseGP reconstructor must be fitted first")
        
        if self._last_std is None:
            # Need to run reconstruct first
            self.reconstruct()
        
        n_points = len(self._last_std)
        
        if n_points == self.resolution[0] * self.resolution[1]:
            return self._last_std.reshape(self.resolution)
        
        return self._last_std


class DenseIFTReconstructor(BaseFieldReconstructor):
    """Dense Information Field Theory based reconstruction.
    
    WARNING: This implementation uses full dense matrices and has O(n³)
    complexity. It will be very slow or run out of memory for large fields
    (e.g., 256×256 = 65K points requires ~34GB for covariance matrix).
    
    Use this only for:
    - Small fields (< 64×64)
    - When you need exact IFT computation
    - Educational/debugging purposes
    
    For production use with larger fields, use the default SparseGPReconstructor.
    
    Parameters
    ----------
    resolution : Tuple[int, int]
        Output field resolution
    power_spectrum_model : str
        Power spectrum model ('power_law', 'gaussian', 'exponential')
    correlation_length : float
        Correlation length scale in pixels
    noise_var : float
        Observation noise variance
    """
    
    MAX_RECOMMENDED_SIZE = DENSE_IFT_MAX_RECOMMENDED_SIZE
    
    def __init__(
        self,
        resolution: Tuple[int, int] = (256, 256),
        power_spectrum_model: str = "gaussian",
        correlation_length: float = 10.0,
        noise_var: float = 0.1,
    ):
        super().__init__(resolution)
        self.power_spectrum_model = power_spectrum_model
        self.correlation_length = correlation_length
        self.noise_var = noise_var
        
        # Warn if resolution is too large
        n_grid = resolution[0] * resolution[1]
        if n_grid > self.MAX_RECOMMENDED_SIZE:
            warnings.warn(
                f"DenseIFTReconstructor with resolution {resolution} ({n_grid} points) "
                f"will require {n_grid**2 * 8 / 1e9:.1f} GB of memory and be very slow. "
                f"Consider using the default SparseGPReconstructor instead (method='ift').",
                UserWarning
            )
        
    def fit(self, observations: np.ndarray, positions: np.ndarray) -> 'DenseIFTReconstructor':
        """Fit IFT model to observations using dense matrices."""
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
        self.grid_points = create_grid_points(self.resolution)
        self.n_grid = len(self.grid_points)
        
        # Compute response matrix R[i,j] = response at observation i due to grid point j
        self.R = np.zeros((self.n_observations, self.n_grid))
        
        for i, obs_pos in enumerate(self.positions):
            for j, grid_pos in enumerate(self.grid_points):
                dist = np.linalg.norm(obs_pos - grid_pos)
                if dist < self.correlation_length:
                    self.R[i, j] = np.exp(-dist**2 / (2 * self.correlation_length**2))
    
    def _compute_prior_covariance(self):
        """Compute prior covariance matrix."""
        self.S = np.zeros((self.n_grid, self.n_grid))
        
        for i, pos_i in enumerate(self.grid_points):
            for j, pos_j in enumerate(self.grid_points):
                dist = np.linalg.norm(pos_i - pos_j)
                
                if self.power_spectrum_model == "exponential":
                    self.S[i, j] = np.exp(-dist / self.correlation_length)
                else:
                    # Gaussian/power_law
                    self.S[i, j] = np.exp(-dist**2 / (2 * self.correlation_length**2))
        
        # Add regularization
        self.S += 1e-6 * np.eye(self.n_grid)
    
    def _compute_posterior(self):
        """Compute posterior mean and covariance."""
        N = self.noise_var * np.eye(self.n_observations)
        
        try:
            S_inv = np.linalg.inv(self.S)
        except np.linalg.LinAlgError:
            S_inv = np.linalg.pinv(self.S)
        
        # Posterior covariance: (S^-1 + R^T N^-1 R)^-1
        N_inv = np.linalg.inv(N)
        info_matrix = S_inv + self.R.T @ N_inv @ self.R
        
        try:
            self.posterior_cov = np.linalg.inv(info_matrix)
        except np.linalg.LinAlgError:
            self.posterior_cov = np.linalg.pinv(info_matrix)
        
        # Posterior mean: D * R^T * N^-1 * d
        self.posterior_mean = self.posterior_cov @ self.R.T @ N_inv @ self.observations
        
    def reconstruct(self, grid_points: Optional[np.ndarray] = None) -> np.ndarray:
        """Reconstruct field using dense IFT."""
        if not self.is_fitted:
            raise RuntimeError("DenseIFT reconstructor must be fitted first")
        
        if grid_points is None:
            field_1d = self.posterior_mean
            return field_1d.reshape(self.resolution)
        else:
            from scipy.spatial import cKDTree
            tree = cKDTree(self.grid_points)
            _, indices = tree.query(grid_points)
            field_1d = self.posterior_mean[indices]
            return field_1d.reshape(self.resolution)
        
    def uncertainty(self) -> np.ndarray:
        """Compute IFT uncertainty estimates."""
        if not self.is_fitted:
            raise RuntimeError("DenseIFT reconstructor must be fitted first")
        
        uncertainty_1d = np.sqrt(np.diag(self.posterior_cov))
        return uncertainty_1d.reshape(self.resolution)


# Keep the old name as an alias for backwards compatibility
IFTReconstructor = SparseGPReconstructor


class GaussianProcessReconstructor(BaseFieldReconstructor):
    """Standard Gaussian Process based field reconstruction.
    
    This uses sklearn's GaussianProcessRegressor directly without
    inducing point approximation. Good for moderate-sized datasets.
    
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
    
    def __init__(
        self,
        resolution: Tuple[int, int] = (256, 256),
        kernel: str = "rbf",
        length_scale: float = 10.0,
        noise_level: float = 0.1
    ):
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
            self.K += 1e-6 * np.eye(self.n_observations)
            self.K_inv = np.linalg.inv(self.K)
        
        # Precompute alpha for efficiency
        self.alpha = self.K_inv @ observations
        
        self.is_fitted = True
        return self
    
    def _compute_kernel_matrix(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """Compute kernel matrix between two sets of points."""
        from scipy.spatial.distance import cdist
        
        distances = cdist(X1, X2)
        
        if self.kernel == "rbf":
            K = np.exp(-distances**2 / (2 * self.length_scale**2))
        elif self.kernel == "matern":
            scaled_dist = distances * np.sqrt(3) / self.length_scale
            K = (1 + scaled_dist) * np.exp(-scaled_dist)
        elif self.kernel == "periodic":
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
        
        K_star = self._compute_kernel_matrix(grid_points, self.positions)
        mu = K_star @ self.alpha
        
        if grid_points.shape[0] == self.resolution[0] * self.resolution[1]:
            mu = mu.reshape(self.resolution)
        
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
        
        K_star_star = self._compute_kernel_matrix(self._last_grid_points, self._last_grid_points)
        var = np.diag(K_star_star - self._last_K_star @ self.K_inv @ self._last_K_star.T)
        var = np.maximum(var, 0)
        
        if var.shape[0] == self.resolution[0] * self.resolution[1]:
            var = var.reshape(self.resolution)
        
        return np.sqrt(var)


class NeuralFieldReconstructor(BaseFieldReconstructor):
    """Neural field based reconstruction using coordinate networks.
    
    This uses a neural network to learn a continuous field representation
    from sparse observations. Includes positional encoding for better
    high-frequency detail capture.
    
    Parameters
    ----------
    resolution : Tuple[int, int]
        Output field resolution
    hidden_dims : Tuple[int, ...]
        Hidden layer dimensions
    activation : str
        Activation function ('relu', 'tanh', 'sigmoid')
    positional_encoding_dims : int
        Number of positional encoding frequencies
    n_epochs : int
        Number of training epochs
    learning_rate : float
        Learning rate for optimization
    """
    
    def __init__(
        self,
        resolution: Tuple[int, int] = (256, 256),
        hidden_dims: Tuple[int, ...] = (256, 128, 64),
        activation: str = "relu",
        positional_encoding_dims: int = 32,
        n_epochs: int = 1000,
        learning_rate: float = 0.001,
        verbose: bool = False,
    ):
        super().__init__(resolution)
        self.hidden_dims = hidden_dims
        self.activation = activation
        self.positional_encoding_dims = positional_encoding_dims
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.verbose = verbose
        
    def fit(self, observations: np.ndarray, positions: np.ndarray) -> 'NeuralFieldReconstructor':
        """Train neural field on observations."""
        import torch
        import torch.nn as nn
        import torch.optim as optim
        
        self.observations = observations
        self.positions = positions
        
        # Normalize positions to [-1, 1]
        self._pos_min = positions.min(axis=0)
        self._pos_max = positions.max(axis=0)
        positions_norm = 2 * (positions - self._pos_min) / (self._pos_max - self._pos_min + 1e-8) - 1
        
        # Normalize observations
        self._y_mean = observations.mean()
        self._y_std = observations.std()
        if self._y_std == 0:
            self._y_std = 1.0
        obs_norm = (observations - self._y_mean) / self._y_std
        
        # Create neural network
        self.network = self._create_network()
        
        # Convert to torch tensors
        pos_tensor = torch.FloatTensor(positions_norm)
        obs_tensor = torch.FloatTensor(obs_norm)
        
        # Apply positional encoding
        if self.positional_encoding_dims > 0:
            pos_tensor = self._positional_encoding(pos_tensor)
        
        # Training
        optimizer = optim.Adam(self.network.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()
        
        for epoch in range(self.n_epochs):
            optimizer.zero_grad()
            pred = self.network(pos_tensor).squeeze()
            loss = criterion(pred, obs_tensor)
            loss.backward()
            optimizer.step()
            
            if self.verbose and (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch+1}/{self.n_epochs}, Loss: {loss.item():.6f}")
        
        self.is_fitted = True
        return self
    
    def _create_network(self):
        """Create neural network for field reconstruction."""
        import torch.nn as nn
        
        input_dim = 2
        if self.positional_encoding_dims > 0:
            input_dim += 4 * self.positional_encoding_dims
        
        layers = []
        layers.append(nn.Linear(input_dim, self.hidden_dims[0]))
        layers.append(self._get_activation())
        
        for i in range(len(self.hidden_dims) - 1):
            layers.append(nn.Linear(self.hidden_dims[i], self.hidden_dims[i + 1]))
            layers.append(self._get_activation())
        
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
        
        freqs = 2.0 ** torch.arange(self.positional_encoding_dims).float()
        
        encoded = [positions]
        for i in range(positions.shape[1]):
            pos = positions[:, i:i+1]
            for freq in freqs:
                encoded.append(torch.sin(freq * np.pi * pos))
                encoded.append(torch.cos(freq * np.pi * pos))
        
        return torch.cat(encoded, dim=1)
        
    def reconstruct(self, grid_points: Optional[np.ndarray] = None) -> np.ndarray:
        """Reconstruct field using trained neural network."""
        if not self.is_fitted:
            raise RuntimeError("Neural field must be fitted first")
        
        import torch
        
        if grid_points is None:
            grid_points = create_grid_points(self.resolution)
        
        # Normalize grid points
        grid_norm = 2 * (grid_points - self._pos_min) / (self._pos_max - self._pos_min + 1e-8) - 1
        
        pos_tensor = torch.FloatTensor(grid_norm)
        
        if self.positional_encoding_dims > 0:
            pos_tensor = self._positional_encoding(pos_tensor)
        
        with torch.no_grad():
            pred = self.network(pos_tensor).squeeze().numpy()
        
        # Denormalize
        pred = pred * self._y_std + self._y_mean
        
        if grid_points.shape[0] == self.resolution[0] * self.resolution[1]:
            pred = pred.reshape(self.resolution)
        
        return pred
        
    def uncertainty(self) -> np.ndarray:
        """Estimate uncertainty (placeholder - returns zeros)."""
        if not self.is_fitted:
            raise RuntimeError("Neural field must be fitted first")
        
        # Neural fields don't provide uncertainty by default
        # Could be improved with MC dropout or ensemble methods
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


def create_reconstructor(
    method: str = "ift",
    resolution: Tuple[int, int] = (256, 256),
    **kwargs
) -> BaseFieldReconstructor:
    """
    Factory function to create field reconstructors.
    
    Parameters
    ----------
    method : str
        Reconstruction method:
        - 'ift' or 'sparse_gp': Sparse GP (scalable, default)
        - 'dense_ift': Dense matrix IFT (exact, slow)
        - 'gp' or 'gaussian_process': Standard GP
        - 'neural' or 'neural_field': Neural network
    resolution : Tuple[int, int]
        Output field resolution
    **kwargs
        Method-specific parameters
        
    Returns
    -------
    reconstructor : BaseFieldReconstructor
        Configured reconstructor
    """
    method_lower = method.lower()
    
    if method_lower in ('ift', 'sparse_gp', 'sparse'):
        return SparseGPReconstructor(resolution, **kwargs)
    elif method_lower == 'dense_ift':
        return DenseIFTReconstructor(resolution, **kwargs)
    elif method_lower in ('gp', 'gaussian_process'):
        return GaussianProcessReconstructor(resolution, **kwargs)
    elif method_lower in ('neural', 'neural_field'):
        return NeuralFieldReconstructor(resolution, **kwargs)
    else:
        raise ValueError(f"Unknown reconstruction method: {method}")
