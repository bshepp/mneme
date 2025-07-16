"""Data preprocessing utilities for field data."""

import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Union
from abc import ABC, abstractmethod
from scipy import ndimage
from scipy.ndimage import gaussian_filter, median_filter
from scipy.interpolate import griddata
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
import warnings

from ..types import Field, PreprocessingStep


class BasePreprocessor(ABC):
    """Abstract base class for preprocessing steps."""
    
    def __init__(self):
        self.is_fitted = False
        self.parameters = {}
    
    @abstractmethod
    def fit(self, data: np.ndarray) -> 'BasePreprocessor':
        """
        Fit preprocessing parameters to data.
        
        Parameters
        ----------
        data : np.ndarray
            Input data to fit to
            
        Returns
        -------
        self : BasePreprocessor
            Fitted preprocessor
        """
        pass
    
    @abstractmethod
    def transform(self, data: np.ndarray) -> np.ndarray:
        """
        Apply preprocessing transformation.
        
        Parameters
        ----------
        data : np.ndarray
            Input data to transform
            
        Returns
        -------
        transformed : np.ndarray
            Transformed data
        """
        pass
    
    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """Fit and transform data in one step."""
        return self.fit(data).transform(data)
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get fitted parameters."""
        return self.parameters.copy()


class Denoiser(BasePreprocessor):
    """Denoise field data using various methods."""
    
    def __init__(
        self,
        method: str = "gaussian",
        sigma: float = 1.0,
        threshold: Optional[str] = None,
        wavelet: str = "db4",
        levels: int = 3
    ):
        """
        Initialize denoiser.
        
        Parameters
        ----------
        method : str
            Denoising method ('gaussian', 'median', 'wavelet')
        sigma : float
            Standard deviation for Gaussian filter
        threshold : str, optional
            Wavelet threshold method ('soft', 'hard')
        wavelet : str
            Wavelet type for wavelet denoising
        levels : int
            Number of wavelet decomposition levels
        """
        super().__init__()
        self.method = method
        self.sigma = sigma
        self.threshold = threshold
        self.wavelet = wavelet
        self.levels = levels
    
    def fit(self, data: np.ndarray) -> 'Denoiser':
        """Fit denoiser parameters."""
        if self.method == "gaussian":
            # Estimate optimal sigma if not provided
            if self.sigma == "auto":
                # Simple noise estimation using high-frequency content
                if data.ndim == 2:
                    laplacian = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
                    noise_est = np.abs(ndimage.convolve(data, laplacian)).mean()
                    self.sigma = max(0.5, min(3.0, noise_est / 2))
                else:
                    self.sigma = 1.0
            
            self.parameters["sigma"] = self.sigma
        
        elif self.method == "median":
            # Median filter doesn't need fitting
            self.parameters["kernel_size"] = 3
        
        elif self.method == "wavelet":
            try:
                import pywt
                # Estimate noise threshold
                coeffs = pywt.wavedec2(data, self.wavelet, level=self.levels)
                sigma_est = np.median(np.abs(coeffs[-1])) / 0.6745
                self.parameters["threshold_value"] = sigma_est * np.sqrt(2 * np.log(data.size))
                self.parameters["sigma_est"] = sigma_est
            except ImportError:
                warnings.warn("PyWavelets not available, falling back to Gaussian")
                self.method = "gaussian"
                self.sigma = 1.0
                self.parameters["sigma"] = self.sigma
        
        self.is_fitted = True
        return self
    
    def transform(self, data: np.ndarray) -> np.ndarray:
        """Apply denoising transformation."""
        if not self.is_fitted:
            raise RuntimeError("Denoiser must be fitted before transformation")
        
        if self.method == "gaussian":
            return gaussian_filter(data, sigma=self.parameters["sigma"])
        
        elif self.method == "median":
            return median_filter(data, size=self.parameters["kernel_size"])
        
        elif self.method == "wavelet":
            try:
                import pywt
                coeffs = pywt.wavedec2(data, self.wavelet, level=self.levels)
                
                # Apply threshold
                threshold_value = self.parameters["threshold_value"]
                if self.threshold == "soft":
                    coeffs_thresh = list(coeffs)
                    coeffs_thresh[1:] = [
                        pywt.threshold(detail, threshold_value, mode='soft') 
                        for detail in coeffs_thresh[1:]
                    ]
                elif self.threshold == "hard":
                    coeffs_thresh = list(coeffs)
                    coeffs_thresh[1:] = [
                        pywt.threshold(detail, threshold_value, mode='hard') 
                        for detail in coeffs_thresh[1:]
                    ]
                else:
                    coeffs_thresh = coeffs
                
                return pywt.waverec2(coeffs_thresh, self.wavelet)
            except ImportError:
                # Fallback to Gaussian if wavelet not available
                return gaussian_filter(data, sigma=1.0)
        
        else:
            raise ValueError(f"Unknown denoising method: {self.method}")


class Normalizer(BasePreprocessor):
    """Normalize field data."""
    
    def __init__(
        self,
        method: str = "z_score",
        per_frame: bool = False,
        clip_percentile: Optional[float] = None
    ):
        """
        Initialize normalizer.
        
        Parameters
        ----------
        method : str
            Normalization method ('z_score', 'min_max', 'robust')
        per_frame : bool
            Whether to normalize each frame independently
        clip_percentile : float, optional
            Percentile for clipping outliers before normalization
        """
        super().__init__()
        self.method = method
        self.per_frame = per_frame
        self.clip_percentile = clip_percentile
    
    def fit(self, data: np.ndarray) -> 'Normalizer':
        """Fit normalization parameters."""
        if self.per_frame and data.ndim == 3:
            # Fit parameters for each frame
            self.parameters["frame_params"] = []
            for i in range(data.shape[0]):
                frame_data = data[i]
                params = self._fit_frame(frame_data)
                self.parameters["frame_params"].append(params)
        else:
            # Fit parameters for entire dataset
            self.parameters.update(self._fit_frame(data))
        
        self.is_fitted = True
        return self
    
    def _fit_frame(self, data: np.ndarray) -> Dict[str, Any]:
        """Fit parameters for a single frame."""
        params = {}
        
        # Clip outliers if requested
        if self.clip_percentile is not None:
            lower = np.percentile(data, self.clip_percentile)
            upper = np.percentile(data, 100 - self.clip_percentile)
            params["clip_lower"] = lower
            params["clip_upper"] = upper
            clipped_data = np.clip(data, lower, upper)
        else:
            clipped_data = data
        
        if self.method == "z_score":
            params["mean"] = np.mean(clipped_data)
            params["std"] = np.std(clipped_data)
        elif self.method == "min_max":
            params["min"] = np.min(clipped_data)
            params["max"] = np.max(clipped_data)
        elif self.method == "robust":
            params["median"] = np.median(clipped_data)
            params["mad"] = np.median(np.abs(clipped_data - params["median"]))
        else:
            raise ValueError(f"Unknown normalization method: {self.method}")
        
        return params
    
    def transform(self, data: np.ndarray) -> np.ndarray:
        """Apply normalization transformation."""
        if not self.is_fitted:
            raise RuntimeError("Normalizer must be fitted before transformation")
        
        if self.per_frame and data.ndim == 3:
            # Transform each frame
            result = np.zeros_like(data)
            for i in range(data.shape[0]):
                if i < len(self.parameters["frame_params"]):
                    params = self.parameters["frame_params"][i]
                else:
                    # Use last frame's parameters for extra frames
                    params = self.parameters["frame_params"][-1]
                result[i] = self._transform_frame(data[i], params)
            return result
        else:
            # Transform entire dataset
            return self._transform_frame(data, self.parameters)
    
    def _transform_frame(self, data: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Transform a single frame."""
        # Clip outliers if parameters exist
        if "clip_lower" in params:
            data = np.clip(data, params["clip_lower"], params["clip_upper"])
        
        if self.method == "z_score":
            if params["std"] > 0:
                return (data - params["mean"]) / params["std"]
            else:
                return data - params["mean"]
        elif self.method == "min_max":
            if params["max"] > params["min"]:
                return (data - params["min"]) / (params["max"] - params["min"])
            else:
                return np.zeros_like(data)
        elif self.method == "robust":
            if params["mad"] > 0:
                return (data - params["median"]) / params["mad"]
            else:
                return data - params["median"]
        else:
            raise ValueError(f"Unknown normalization method: {self.method}")


class Registrator(BasePreprocessor):
    """Register field data to align sequences."""
    
    def __init__(
        self,
        reference: str = "first",
        method: str = "rigid",
        max_shift: int = 10
    ):
        """
        Initialize registrator.
        
        Parameters
        ----------
        reference : str
            Reference frame ('first', 'mean', 'median')
        method : str
            Registration method ('rigid', 'translation_only')
        max_shift : int
            Maximum allowed shift in pixels
        """
        super().__init__()
        self.reference = reference
        self.method = method
        self.max_shift = max_shift
    
    def fit(self, data: np.ndarray) -> 'Registrator':
        """Fit registration parameters."""
        if data.ndim != 3:
            raise ValueError("Registration requires 3D data (time, height, width)")
        
        # Compute reference frame
        if self.reference == "first":
            self.parameters["reference_frame"] = data[0].copy()
        elif self.reference == "mean":
            self.parameters["reference_frame"] = np.mean(data, axis=0)
        elif self.reference == "median":
            self.parameters["reference_frame"] = np.median(data, axis=0)
        else:
            raise ValueError(f"Unknown reference type: {self.reference}")
        
        # Compute shifts for each frame
        shifts = []
        for i in range(data.shape[0]):
            shift = self._compute_shift(data[i], self.parameters["reference_frame"])
            shifts.append(shift)
        
        self.parameters["shifts"] = shifts
        self.is_fitted = True
        return self
    
    def _compute_shift(self, frame: np.ndarray, reference: np.ndarray) -> Tuple[float, float]:
        """Compute optimal shift using cross-correlation."""
        # Use normalized cross-correlation
        correlation = ndimage.correlate(reference, frame, mode='constant')
        
        # Find peak
        peak_coords = np.unravel_index(np.argmax(correlation), correlation.shape)
        
        # Convert to shift
        shift_y = peak_coords[0] - reference.shape[0] // 2
        shift_x = peak_coords[1] - reference.shape[1] // 2
        
        # Limit shift magnitude
        shift_y = np.clip(shift_y, -self.max_shift, self.max_shift)
        shift_x = np.clip(shift_x, -self.max_shift, self.max_shift)
        
        return (shift_y, shift_x)
    
    def transform(self, data: np.ndarray) -> np.ndarray:
        """Apply registration transformation."""
        if not self.is_fitted:
            raise RuntimeError("Registrator must be fitted before transformation")
        
        if data.ndim != 3:
            raise ValueError("Registration requires 3D data (time, height, width)")
        
        result = np.zeros_like(data)
        for i in range(data.shape[0]):
            if i < len(self.parameters["shifts"]):
                shift = self.parameters["shifts"][i]
            else:
                shift = (0, 0)  # No shift for extra frames
            
            # Apply shift
            if shift != (0, 0):
                result[i] = ndimage.shift(data[i], shift, mode='constant', cval=0)
            else:
                result[i] = data[i]
        
        return result


class Interpolator(BasePreprocessor):
    """Interpolate field data to different resolutions."""
    
    def __init__(
        self,
        target_shape: Tuple[int, int],
        method: str = "bicubic",
        preserve_range: bool = True
    ):
        """
        Initialize interpolator.
        
        Parameters
        ----------
        target_shape : Tuple[int, int]
            Target shape (height, width)
        method : str
            Interpolation method ('nearest', 'linear', 'bicubic')
        preserve_range : bool
            Whether to preserve original value range
        """
        super().__init__()
        self.target_shape = target_shape
        self.method = method
        self.preserve_range = preserve_range
    
    def fit(self, data: np.ndarray) -> 'Interpolator':
        """Fit interpolation parameters."""
        self.parameters["original_shape"] = data.shape[-2:]
        
        if self.preserve_range:
            if data.ndim == 2:
                self.parameters["value_range"] = (np.min(data), np.max(data))
            else:
                self.parameters["value_range"] = (np.min(data), np.max(data))
        
        self.is_fitted = True
        return self
    
    def transform(self, data: np.ndarray) -> np.ndarray:
        """Apply interpolation transformation."""
        if not self.is_fitted:
            raise RuntimeError("Interpolator must be fitted before transformation")
        
        original_shape = data.shape
        
        if data.ndim == 2:
            result = self._interpolate_2d(data)
        elif data.ndim == 3:
            result = np.zeros((data.shape[0], *self.target_shape))
            for i in range(data.shape[0]):
                result[i] = self._interpolate_2d(data[i])
        else:
            raise ValueError("Interpolation only supports 2D or 3D data")
        
        # Preserve value range if requested
        if self.preserve_range and "value_range" in self.parameters:
            min_val, max_val = self.parameters["value_range"]
            result = np.clip(result, min_val, max_val)
        
        return result
    
    def _interpolate_2d(self, data: np.ndarray) -> np.ndarray:
        """Interpolate 2D data."""
        h, w = data.shape
        th, tw = self.target_shape
        
        # Create coordinate grids
        y_old = np.linspace(0, 1, h)
        x_old = np.linspace(0, 1, w)
        y_new = np.linspace(0, 1, th)
        x_new = np.linspace(0, 1, tw)
        
        # Create meshgrids
        X_old, Y_old = np.meshgrid(x_old, y_old)
        X_new, Y_new = np.meshgrid(x_new, y_new)
        
        # Flatten for griddata
        points = np.column_stack([X_old.ravel(), Y_old.ravel()])
        values = data.ravel()
        xi = np.column_stack([X_new.ravel(), Y_new.ravel()])
        
        # Interpolate
        try:
            if self.method == "nearest":
                interpolated = griddata(points, values, xi, method='nearest')
            elif self.method == "linear":
                interpolated = griddata(points, values, xi, method='linear')
            elif self.method == "bicubic":
                interpolated = griddata(points, values, xi, method='cubic')
            else:
                raise ValueError(f"Unknown interpolation method: {self.method}")
        except Exception:
            # Fallback to linear if cubic fails
            interpolated = griddata(points, values, xi, method='linear')
        
        # Handle NaN values
        if np.any(np.isnan(interpolated)):
            # Fill NaN with nearest neighbor
            interpolated_nn = griddata(points, values, xi, method='nearest')
            interpolated = np.where(np.isnan(interpolated), interpolated_nn, interpolated)
        
        return interpolated.reshape(self.target_shape)


class FieldPreprocessor:
    """Combined preprocessing pipeline for field data."""
    
    def __init__(self, steps: List[Union[str, Tuple[str, Dict[str, Any]]]]):
        """
        Initialize preprocessing pipeline.
        
        Parameters
        ----------
        steps : List[Union[str, Tuple[str, Dict[str, Any]]]]
            List of preprocessing steps
        """
        self.steps = []
        self.step_names = []
        
        for step in steps:
            if isinstance(step, str):
                name = step
                params = {}
            else:
                name, params = step
            
            self.step_names.append(name)
            
            if name == "denoise":
                self.steps.append(Denoiser(**params))
            elif name == "normalize":
                self.steps.append(Normalizer(**params))
            elif name == "register":
                self.steps.append(Registrator(**params))
            elif name == "interpolate":
                self.steps.append(Interpolator(**params))
            else:
                raise ValueError(f"Unknown preprocessing step: {name}")
    
    def fit(self, data: np.ndarray) -> 'FieldPreprocessor':
        """Fit all preprocessing steps."""
        current_data = data
        
        for step in self.steps:
            step.fit(current_data)
            current_data = step.transform(current_data)
        
        return self
    
    def transform(self, data: np.ndarray) -> np.ndarray:
        """Apply all preprocessing steps."""
        current_data = data
        
        for step in self.steps:
            current_data = step.transform(current_data)
        
        return current_data
    
    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """Fit and transform data."""
        return self.fit(data).transform(data)
    
    def get_step_parameters(self) -> Dict[str, Dict[str, Any]]:
        """Get parameters for all steps."""
        parameters = {}
        for name, step in zip(self.step_names, self.steps):
            parameters[name] = step.get_parameters()
        return parameters