"""Field reconstruction from sparse observations.

Four backends: a Gaussian process fitted to a random subset of the
observations (default), a dense Wiener filter, a standard Gaussian process,
and a coordinate neural network.

The module keeps its historical name. Earlier versions described the
default as "Information Field Theory" and "sparse GP"; it was neither, and
the classes have been renamed to say what they do. The old names still
resolve, with a DeprecationWarning.
"""

from typing import Optional, Tuple, Dict, Any, Union
import warnings
import numpy as np
from abc import ABC, abstractmethod

from .._status import warn_experimental
from ..types import (
    Field, ReconstructionMethod, ReconstructionResult,
    Coordinates, FieldData
)

# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------

#: Number of grid points to predict per batch in SubsetGPReconstructor.
#: Controls the memory/speed trade-off when evaluating the GP on a dense grid.
GP_PREDICTION_BATCH_SIZE: int = 10_000

#: Maximum grid size (height*width) before WienerFilterReconstructor emits a
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
        - 'gp_subset': Gaussian process fitted to a random subset of the
          observations (default, scalable)
        - 'wiener_filter': dense Wiener filter (O(n³), small fields only)
        - 'gaussian_process': Standard GP reconstruction
        - 'neural_field': Neural network-based reconstruction
        The older names 'ift', 'sparse_gp' and 'dense_ift' still work and
        emit a DeprecationWarning.
    resolution : Tuple[int, int]
        Output field resolution (height, width)
    **kwargs
        Additional method-specific parameters
        
    Examples
    --------
    >>> from mneme.core.field_theory import FieldReconstructor
    >>> import numpy as np
    >>> 
    >>> # Create reconstructor (subset-of-data GP by default)
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
        method: Union[str, ReconstructionMethod] = ReconstructionMethod.GP_SUBSET,
        resolution: Tuple[int, int] = (256, 256),
        **kwargs
    ):
        super().__init__(resolution)

        use_dense = kwargs.pop('use_dense', False)
        self.method = _resolve_method(method, use_dense=use_dense)
        self.method_params = kwargs
        self._backend = None
        self._initialize_backend()

    def _initialize_backend(self):
        """Initialize the appropriate backend reconstructor."""
        if self.method == ReconstructionMethod.GP_SUBSET:
            self._backend = SubsetGPReconstructor(self.resolution, **self.method_params)
        elif self.method == ReconstructionMethod.WIENER_FILTER:
            self._backend = WienerFilterReconstructor(self.resolution, **self.method_params)
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
        try:
            uncertainty_data = self.uncertainty()
        except NotImplementedError:
            # The backend has no uncertainty estimate; report none rather
            # than a placeholder.
            uncertainty_data = None
        
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


class SubsetGPReconstructor(BaseFieldReconstructor):
    """Gaussian process regression on a random subset of the observations.

    When there are more than `n_subset` observations, a random `n_subset`
    of them are used to fit an exact GP and THE REST ARE DISCARDED. This
    is the "subset of data" approximation. It is not a sparse GP in the
    inducing-point sense (no FITC/VFE), and it is not Information Field
    Theory. Cost is O(m³) in the subset size m.

    The reported uncertainty is the posterior standard deviation of the
    subset model, so it does not reflect the discarded observations.

    Parameters
    ----------
    resolution : Tuple[int, int]
        Output field resolution
    n_subset : int
        Maximum number of observations used to fit the GP.
    kernel : str
        Kernel type: 'rbf', 'matern', 'exponential'
    length_scale : float
        Kernel length scale, as a fraction of the observation extent
    noise_level : float
        Observation noise level
    optimize_hyperparameters : bool
        Whether to optimize kernel hyperparameters during fitting. When
        False the kernel is used exactly as specified.
    random_state : int, optional
        Random seed for subset selection
    n_inducing : int, optional
        Deprecated name for `n_subset`.
    """

    def __init__(
        self,
        resolution: Tuple[int, int] = (256, 256),
        n_subset: int = 500,
        kernel: str = "rbf",
        length_scale: float = 0.1,
        noise_level: float = 0.1,
        optimize_hyperparameters: bool = True,
        random_state: Optional[int] = None,
        # Legacy parameter mapping
        correlation_length: Optional[float] = None,
        power_spectrum_model: Optional[str] = None,
        n_inducing: Optional[int] = None,
    ):
        super().__init__(resolution)

        if n_inducing is not None:
            warnings.warn(
                "n_inducing is deprecated; use n_subset. The reconstructor "
                "fits a GP to a random subset and has no inducing points.",
                DeprecationWarning,
                stacklevel=2,
            )
            n_subset = n_inducing
        self.n_subset = n_subset
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
        
    @property
    def n_inducing(self) -> int:
        """Deprecated name for `n_subset`."""
        return self.n_subset

    def fit(self, observations: np.ndarray, positions: np.ndarray) -> 'SubsetGPReconstructor':
        """Fit a GP to the observations, or to a random subset of them.

        When there are more than `n_subset` observations a random subset is
        used and ``n_discarded_`` records how many were left out.
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
        
        # Use a random subset when there are more observations than n_subset
        if n_obs > self.n_subset:
            rng = np.random.RandomState(self.random_state)
            subset_idx = rng.choice(n_obs, size=self.n_subset, replace=False)
            X_train = positions_norm[subset_idx]
            y_train = observations[subset_idx]
        else:
            X_train = positions_norm
            y_train = observations
        self.n_used_ = len(y_train)
        self.n_discarded_ = n_obs - len(y_train)
        
        # Normalize observations
        self._y_mean = y_train.mean()
        self._y_std = y_train.std()
        if self._y_std == 0:
            self._y_std = 1.0
        y_train_norm = (y_train - self._y_mean) / self._y_std
        
        # Create and fit GP
        kernel = self._create_kernel()
        
        if self.optimize_hyperparameters:
            self._gp = GaussianProcessRegressor(
                kernel=kernel,
                n_restarts_optimizer=5,
                normalize_y=False,  # We already normalized
                random_state=self.random_state,
            )
        else:
            # optimizer=None keeps the kernel exactly as specified.
            self._gp = GaussianProcessRegressor(
                kernel=kernel,
                optimizer=None,
                normalize_y=False,
                random_state=self.random_state,
            )

        self._gp.fit(X_train, y_train_norm)

        self.is_fitted = True
        return self

    def reconstruct(self, grid_points: Optional[np.ndarray] = None) -> np.ndarray:
        """Reconstruct field using the fitted GP."""
        if not self.is_fitted:
            raise RuntimeError("Reconstructor must be fitted first")
        
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
        """Posterior standard deviation of the fitted (subset) GP."""
        if not self.is_fitted:
            raise RuntimeError("Reconstructor must be fitted first")
        
        if self._last_std is None:
            # Need to run reconstruct first
            self.reconstruct()
        
        n_points = len(self._last_std)
        
        if n_points == self.resolution[0] * self.resolution[1]:
            return self._last_std.reshape(self.resolution)
        
        return self._last_std


class WienerFilterReconstructor(BaseFieldReconstructor):
    """Dense Wiener-filter reconstruction on the grid.

    Computes the posterior mean and covariance of a Gaussian field with a
    fixed stationary prior, observed through a Gaussian-blur response with
    white noise. This is the free-theory (linear, Gaussian) case of
    Information Field Theory, which is the classical Wiener filter; nothing
    beyond that is implemented.

    WARNING: uses full dense matrices, O(n³) in the number of grid points.
    256×256 = 65K points needs ~34GB for the covariance matrix. Use only
    for small fields (< 64×64).

    Parameters
    ----------
    resolution : Tuple[int, int]
        Output field resolution
    power_spectrum_model : str
        Prior covariance shape ('power_law', 'gaussian', 'exponential')
    correlation_length : float
        Correlation length in pixels. Converted to grid coordinates by
        dividing by the larger grid dimension, since the grid spans the
        unit square.
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
        self.correlation_length_pixels = correlation_length
        # The grid spans the unit square, so a length in pixels must be
        # converted before it is compared with grid distances.
        self.correlation_length = correlation_length / max(resolution)
        self.noise_var = noise_var
        
        # Warn if resolution is too large
        n_grid = resolution[0] * resolution[1]
        if n_grid > self.MAX_RECOMMENDED_SIZE:
            warnings.warn(
                f"WienerFilterReconstructor with resolution {resolution} ({n_grid} points) "
                f"will require {n_grid**2 * 8 / 1e9:.1f} GB of memory and be very slow. "
                f"Consider the default SubsetGPReconstructor (method='gp_subset').",
                UserWarning
            )
        
    def fit(self, observations: np.ndarray, positions: np.ndarray) -> 'WienerFilterReconstructor':
        """Compute the Wiener-filter posterior using dense matrices."""
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

        # Each observation is a weighted AVERAGE of nearby grid values, so
        # the weights of a row must sum to one. Left unnormalised, a row sums
        # to roughly the number of grid points within a correlation length
        # and the reconstruction is scaled down by that factor.
        row_sums = self.R.sum(axis=1, keepdims=True)
        if np.any(row_sums == 0):
            raise ValueError(
                "An observation lies further than one correlation length from "
                "every grid point; increase correlation_length or resolution."
            )
        self.R /= row_sums
    
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
        """Return the posterior mean field."""
        if not self.is_fitted:
            raise RuntimeError("Reconstructor must be fitted first")
        
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
        """Posterior standard deviation on the grid."""
        if not self.is_fitted:
            raise RuntimeError("Reconstructor must be fitted first")
        
        uncertainty_1d = np.sqrt(np.diag(self.posterior_cov))
        return uncertainty_1d.reshape(self.resolution)


#: Old class names, kept so existing code keeps working. They resolve to
#: the SAME class objects (so isinstance checks still hold) through the
#: module-level ``__getattr__`` below, which warns on access.
_DEPRECATED_CLASSES = {
    "SparseGPReconstructor": "SubsetGPReconstructor",
    "IFTReconstructor": "SubsetGPReconstructor",
    "DenseIFTReconstructor": "WienerFilterReconstructor",
}


def __getattr__(name: str):
    if name in _DEPRECATED_CLASSES:
        new_name = _DEPRECATED_CLASSES[name]
        warnings.warn(
            f"{name} is deprecated; use {new_name}.",
            DeprecationWarning,
            stacklevel=2,
        )
        return globals()[new_name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

#: Deprecated method names and what they now select.
_DEPRECATED_METHODS = {
    "ift": ReconstructionMethod.GP_SUBSET,
    "sparse_gp": ReconstructionMethod.GP_SUBSET,
    "sparse": ReconstructionMethod.GP_SUBSET,
    "dense_ift": ReconstructionMethod.WIENER_FILTER,
}

_METHOD_ALIASES = {
    "gp": ReconstructionMethod.GAUSSIAN_PROCESS,
    "neural": ReconstructionMethod.NEURAL_FIELD,
    "wiener": ReconstructionMethod.WIENER_FILTER,
}


def _resolve_method(
    method: Union[str, ReconstructionMethod], use_dense: bool = False
) -> ReconstructionMethod:
    """Map a method name (current or deprecated) to a ReconstructionMethod."""
    if isinstance(method, ReconstructionMethod) and method is not ReconstructionMethod.IFT:
        return method
    name = method.value if isinstance(method, ReconstructionMethod) else str(method).lower()
    if name in _DEPRECATED_METHODS:
        resolved = _DEPRECATED_METHODS[name]
        if name == "ift" and use_dense:
            resolved = ReconstructionMethod.WIENER_FILTER
        warnings.warn(
            f"Reconstruction method {name!r} is deprecated; use "
            f"{resolved.value!r}. The method is unchanged, only the name.",
            DeprecationWarning,
            stacklevel=3,
        )
        return resolved
    if name in _METHOD_ALIASES:
        return _METHOD_ALIASES[name]
    try:
        return ReconstructionMethod(name)
    except ValueError:
        raise ValueError(f"Unknown reconstruction method: {method}") from None


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
        warn_experimental("NeuralFieldReconstructor")
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
        """Not available: a single trained network gives no uncertainty."""
        raise NotImplementedError(
            "NeuralFieldReconstructor does not estimate uncertainty"
        )


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
    method: str = "gp_subset",
    resolution: Tuple[int, int] = (256, 256),
    **kwargs
) -> BaseFieldReconstructor:
    """
    Factory function to create field reconstructors.

    Parameters
    ----------
    method : str
        Reconstruction method:
        - 'gp_subset': GP on a random subset of observations (default)
        - 'wiener_filter' or 'wiener': dense Wiener filter (slow)
        - 'gp' or 'gaussian_process': Standard GP
        - 'neural' or 'neural_field': Neural network
        Deprecated names 'ift', 'sparse_gp', 'sparse' and 'dense_ift'
        still work and emit a DeprecationWarning.
    resolution : Tuple[int, int]
        Output field resolution
    **kwargs
        Method-specific parameters
        
    Returns
    -------
    reconstructor : BaseFieldReconstructor
        Configured reconstructor
    """
    resolved = _resolve_method(method)
    backends = {
        ReconstructionMethod.GP_SUBSET: SubsetGPReconstructor,
        ReconstructionMethod.WIENER_FILTER: WienerFilterReconstructor,
        ReconstructionMethod.GAUSSIAN_PROCESS: GaussianProcessReconstructor,
        ReconstructionMethod.NEURAL_FIELD: NeuralFieldReconstructor,
    }
    return backends[resolved](resolution, **kwargs)
