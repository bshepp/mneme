"""Synthetic data generation for testing and validation."""

import numpy as np
from typing import Dict, Any, Optional, Tuple, Union, List
from scipy import ndimage
from scipy.ndimage import gaussian_filter
from scipy.spatial.distance import cdist
import warnings

from ..types import Field, FieldData


class SyntheticFieldGenerator:
    """Generate synthetic field data for testing."""
    
    def __init__(self, field_type: str = "gaussian_random", seed: Optional[int] = None):
        """
        Initialize synthetic field generator.
        
        Parameters
        ----------
        field_type : str
            Type of field to generate
        seed : int, optional
            Random seed for reproducibility
        """
        self.field_type = field_type
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)
    
    def generate_static(self, shape: Tuple[int, ...], parameters: Dict[str, Any]) -> np.ndarray:
        """
        Generate static field.
        
        Parameters
        ----------
        shape : Tuple[int, ...]
            Shape of field to generate
        parameters : Dict[str, Any]
            Parameters specific to field type
            
        Returns
        -------
        field : np.ndarray
            Generated field
        """
        if self.field_type == "gaussian_random":
            return self._generate_gaussian_random(shape, parameters)
        elif self.field_type == "gaussian_blob":
            return self._generate_gaussian_blob(shape, parameters)
        elif self.field_type == "sinusoidal":
            return self._generate_sinusoidal(shape, parameters)
        elif self.field_type == "turbulent":
            return self._generate_turbulent(shape, parameters)
        elif self.field_type == "reaction_diffusion":
            return self._generate_reaction_diffusion(shape, parameters)
        elif self.field_type == "bioelectric_gradient":
            return self._generate_bioelectric_gradient(shape, parameters)
        else:
            raise ValueError(f"Unknown field type: {self.field_type}")
    
    def generate_dynamic(
        self, 
        shape: Tuple[int, ...], 
        timesteps: int, 
        parameters: Dict[str, Any]
    ) -> np.ndarray:
        """
        Generate time-evolving field.
        
        Parameters
        ----------
        shape : Tuple[int, ...]
            Spatial shape of field
        timesteps : int
            Number of time steps
        parameters : Dict[str, Any]
            Parameters for evolution
            
        Returns
        -------
        field : np.ndarray
            Time-evolving field with shape (timesteps, *shape)
        """
        # Generate initial field
        initial_field = self.generate_static(shape, parameters)
        
        # Initialize output array
        dynamic_field = np.zeros((timesteps, *shape))
        dynamic_field[0] = initial_field
        
        # Get evolution parameters
        drift_velocity = parameters.get("drift_velocity", (0.0, 0.0))
        growth_rate = parameters.get("growth_rate", 0.0)
        diffusion_rate = parameters.get("diffusion_rate", 0.01)
        noise_level = parameters.get("noise_level", 0.0)
        
        # Evolve field over time
        for t in range(1, timesteps):
            current_field = dynamic_field[t-1].copy()
            
            # Apply drift
            if drift_velocity != (0.0, 0.0):
                current_field = self._apply_drift(current_field, drift_velocity)
            
            # Apply growth/decay
            if growth_rate != 0.0:
                current_field *= (1 + growth_rate)
            
            # Apply diffusion
            if diffusion_rate > 0.0:
                current_field = gaussian_filter(current_field, sigma=diffusion_rate)
            
            # Add noise
            if noise_level > 0.0:
                noise = np.random.normal(0, noise_level, current_field.shape)
                current_field += noise
            
            dynamic_field[t] = current_field
        
        return dynamic_field
    
    def add_noise(
        self, 
        field: np.ndarray, 
        noise_level: float, 
        noise_type: str = "gaussian"
    ) -> np.ndarray:
        """
        Add realistic noise to field.
        
        Parameters
        ----------
        field : np.ndarray
            Clean field
        noise_level : float
            Noise intensity
        noise_type : str
            Type of noise to add
            
        Returns
        -------
        noisy_field : np.ndarray
            Field with added noise
        """
        noisy_field = field.copy()
        
        if noise_type == "gaussian":
            noise = np.random.normal(0, noise_level, field.shape)
            noisy_field += noise
        elif noise_type == "poisson":
            # Scale to positive values for Poisson noise
            scaled_field = field - field.min() + 1e-6
            noisy_field = np.random.poisson(scaled_field * noise_level) / noise_level
            noisy_field += field.min()
        elif noise_type == "salt_pepper":
            mask = np.random.random(field.shape) < noise_level
            salt_pepper = np.random.choice([0, 1], size=field.shape, p=[0.5, 0.5])
            noisy_field[mask] = salt_pepper[mask] * (field.max() - field.min()) + field.min()
        elif noise_type == "speckle":
            noise = np.random.normal(1, noise_level, field.shape)
            noisy_field *= noise
        else:
            raise ValueError(f"Unknown noise type: {noise_type}")
        
        return noisy_field
    
    def _generate_gaussian_random(self, shape: Tuple[int, ...], parameters: Dict[str, Any]) -> np.ndarray:
        """Generate Gaussian random field."""
        correlation_length = parameters.get("correlation_length", 5.0)
        amplitude = parameters.get("amplitude", 1.0)
        
        # Generate white noise
        field = np.random.normal(0, 1, shape)
        
        # Apply spatial correlation
        if correlation_length > 0:
            field = gaussian_filter(field, sigma=correlation_length)
        
        return field * amplitude
    
    def _generate_gaussian_blob(self, shape: Tuple[int, ...], parameters: Dict[str, Any]) -> np.ndarray:
        """Generate field with Gaussian blobs."""
        n_centers = parameters.get("n_centers", 3)
        sigma = parameters.get("sigma", 10.0)
        amplitude = parameters.get("amplitude", 1.0)
        
        field = np.zeros(shape)
        
        # Generate random centers
        centers = []
        for _ in range(n_centers):
            center = tuple(np.random.randint(0, s) for s in shape)
            centers.append(center)
        
        # Create coordinate grids
        coords = np.meshgrid(*[np.arange(s) for s in shape], indexing='ij')
        
        # Add Gaussian blobs
        for center in centers:
            blob_amplitude = amplitude * np.random.uniform(0.5, 1.5)
            blob_sigma = sigma * np.random.uniform(0.7, 1.3)
            
            distances_sq = sum((coord - c)**2 for coord, c in zip(coords, center))
            blob = blob_amplitude * np.exp(-distances_sq / (2 * blob_sigma**2))
            field += blob
        
        return field
    
    def _generate_sinusoidal(self, shape: Tuple[int, ...], parameters: Dict[str, Any]) -> np.ndarray:
        """Generate sinusoidal field."""
        frequency = parameters.get("frequency", 0.1)
        angle = parameters.get("angle", 0.0)
        amplitude = parameters.get("amplitude", 1.0)
        phase = parameters.get("phase", 0.0)
        
        # Create coordinate grids
        coords = np.meshgrid(*[np.arange(s) for s in shape], indexing='ij')
        
        # Apply rotation
        angle_rad = np.radians(angle)
        if len(shape) == 2:
            rotated_x = coords[0] * np.cos(angle_rad) - coords[1] * np.sin(angle_rad)
            field = amplitude * np.sin(2 * np.pi * frequency * rotated_x + phase)
        else:
            # For higher dimensions, use first coordinate
            field = amplitude * np.sin(2 * np.pi * frequency * coords[0] + phase)
        
        return field
    
    def _generate_turbulent(self, shape: Tuple[int, ...], parameters: Dict[str, Any]) -> np.ndarray:
        """Generate turbulent field using fractal noise."""
        scale = parameters.get("scale", 10.0)
        intensity = parameters.get("intensity", 1.0)
        octaves = parameters.get("octaves", 4)
        
        field = np.zeros(shape)
        
        # Generate fractal noise
        for octave in range(octaves):
            frequency = 2**octave / scale
            amplitude = intensity / (2**octave)
            
            # Generate noise at this frequency
            noise = np.random.normal(0, 1, shape)
            if frequency > 0:
                noise = gaussian_filter(noise, sigma=1/frequency)
            
            field += amplitude * noise
        
        return field
    
    def _generate_reaction_diffusion(self, shape: Tuple[int, ...], parameters: Dict[str, Any]) -> np.ndarray:
        """Generate reaction-diffusion pattern."""
        a = parameters.get("a", 0.16)  # Feed rate
        b = parameters.get("b", 0.08)  # Kill rate
        timesteps = parameters.get("timesteps", 1000)
        
        if len(shape) != 2:
            raise ValueError("Reaction-diffusion only supports 2D fields")
        
        h, w = shape
        
        # Initialize with random perturbations
        u = np.ones((h, w))
        v = np.zeros((h, w))
        
        # Add initial perturbation
        center_h, center_w = h // 2, w // 2
        size = min(h, w) // 10
        u[center_h-size:center_h+size, center_w-size:center_w+size] += 0.25 * np.random.random((2*size, 2*size))
        v[center_h-size:center_h+size, center_w-size:center_w+size] += 0.25 * np.random.random((2*size, 2*size))
        
        # Diffusion coefficients
        Du, Dv = 0.16, 0.08
        dt = 1.0
        
        # Laplacian kernel
        laplacian = np.array([[0.05, 0.2, 0.05],
                             [0.2, -1, 0.2],
                             [0.05, 0.2, 0.05]])
        
        # Simulate reaction-diffusion
        for _ in range(timesteps):
            # Apply Laplacian
            lap_u = ndimage.convolve(u, laplacian, mode='wrap')
            lap_v = ndimage.convolve(v, laplacian, mode='wrap')
            
            # Reaction terms
            uvv = u * v * v
            
            # Update equations
            u += dt * (Du * lap_u - uvv + a * (1 - u))
            v += dt * (Dv * lap_v + uvv - (a + b) * v)
        
        return v  # Return the pattern-forming component
    
    def _generate_bioelectric_gradient(self, shape: Tuple[int, ...], parameters: Dict[str, Any]) -> np.ndarray:
        """Generate bioelectric gradient field."""
        anterior_voltage = parameters.get("anterior_voltage", -50.0)
        posterior_voltage = parameters.get("posterior_voltage", -20.0)
        lateral_variation = parameters.get("lateral_variation", 10.0)
        gradient_steepness = parameters.get("gradient_steepness", 0.1)
        
        if len(shape) != 2:
            raise ValueError("Bioelectric gradient only supports 2D fields")
        
        h, w = shape
        
        # Create coordinate grids
        y_coords = np.linspace(0, 1, h)
        x_coords = np.linspace(0, 1, w)
        X, Y = np.meshgrid(x_coords, y_coords)
        
        # Base gradient (anterior-posterior)
        gradient = anterior_voltage + (posterior_voltage - anterior_voltage) * Y
        
        # Add lateral variation
        if lateral_variation > 0:
            lateral = lateral_variation * np.sin(2 * np.pi * X)
            gradient += lateral
        
        # Add some random spatial variation
        noise = np.random.normal(0, 2, shape)
        gradient += gaussian_filter(noise, sigma=2.0)
        
        return gradient
    
    def _apply_drift(self, field: np.ndarray, velocity: Tuple[float, float]) -> np.ndarray:
        """Apply drift to field."""
        if len(field.shape) != 2:
            raise ValueError("Drift only supports 2D fields")
        
        vx, vy = velocity
        
        # Simple shift implementation
        if vx != 0 or vy != 0:
            # Use scipy's shift function for sub-pixel accuracy
            field = ndimage.shift(field, (vy, vx), mode='wrap')
        
        return field


def generate_planarian_bioelectric_sequence(
    shape: Tuple[int, int] = (256, 128),
    timesteps: int = 100,
    regeneration_event_time: int = 20,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Generate a sequence mimicking planarian bioelectric patterns during regeneration.
    
    Parameters
    ----------
    shape : Tuple[int, int]
        Field shape (height, width)
    timesteps : int
        Number of time steps
    regeneration_event_time : int
        Time step when regeneration event occurs
    seed : int, optional
        Random seed
        
    Returns
    -------
    sequence : np.ndarray
        Bioelectric sequence with shape (timesteps, height, width)
    """
    if seed is not None:
        np.random.seed(seed)
    
    generator = SyntheticFieldGenerator("bioelectric_gradient", seed=seed)
    
    # Pre-regeneration parameters
    pre_params = {
        "anterior_voltage": -50.0,
        "posterior_voltage": -20.0,
        "lateral_variation": 5.0,
        "gradient_steepness": 0.1
    }
    
    # Post-regeneration parameters (more dynamic)
    post_params = {
        "anterior_voltage": -60.0,
        "posterior_voltage": -15.0,
        "lateral_variation": 15.0,
        "gradient_steepness": 0.2
    }
    
    sequence = np.zeros((timesteps, *shape))
    
    for t in range(timesteps):
        # Gradually transition parameters after regeneration event
        if t < regeneration_event_time:
            params = pre_params.copy()
        else:
            # Smoothly transition parameters
            alpha = min(1.0, (t - regeneration_event_time) / 20.0)
            params = {}
            for key in pre_params:
                params[key] = pre_params[key] + alpha * (post_params[key] - pre_params[key])
        
        # Generate field for this time step
        field = generator.generate_static(shape, params)
        
        # Add temporal noise
        noise = np.random.normal(0, 1, shape)
        field += gaussian_filter(noise, sigma=1.0)
        
        sequence[t] = field
    
    return sequence


def generate_test_dataset(
    n_samples: int = 100,
    shape: Tuple[int, int] = (64, 64),
    field_types: Optional[List[str]] = None,
    seed: Optional[int] = None
) -> Dict[str, np.ndarray]:
    """
    Generate a test dataset with multiple field types.
    
    Parameters
    ----------
    n_samples : int
        Number of samples per field type
    shape : Tuple[int, int]
        Field shape
    field_types : List[str], optional
        Field types to generate
    seed : int, optional
        Random seed
        
    Returns
    -------
    dataset : Dict[str, np.ndarray]
        Dictionary with field types as keys and arrays as values
    """
    if seed is not None:
        np.random.seed(seed)
    
    if field_types is None:
        field_types = ["gaussian_blob", "sinusoidal", "turbulent", "bioelectric_gradient"]
    
    dataset = {}
    
    for field_type in field_types:
        generator = SyntheticFieldGenerator(field_type, seed=seed)
        samples = []
        
        for i in range(n_samples):
            # Generate random parameters for variety
            if field_type == "gaussian_blob":
                params = {
                    "n_centers": np.random.randint(2, 6),
                    "sigma": np.random.uniform(5, 15),
                    "amplitude": np.random.uniform(0.5, 2.0)
                }
            elif field_type == "sinusoidal":
                params = {
                    "frequency": np.random.uniform(0.05, 0.2),
                    "angle": np.random.uniform(0, 180),
                    "amplitude": np.random.uniform(0.5, 2.0)
                }
            elif field_type == "turbulent":
                params = {
                    "scale": np.random.uniform(5, 20),
                    "intensity": np.random.uniform(0.5, 2.0),
                    "octaves": np.random.randint(3, 6)
                }
            elif field_type == "bioelectric_gradient":
                params = {
                    "anterior_voltage": np.random.uniform(-60, -40),
                    "posterior_voltage": np.random.uniform(-30, -10),
                    "lateral_variation": np.random.uniform(5, 15)
                }
            else:
                params = {}
            
            sample = generator.generate_static(shape, params)
            samples.append(sample)
        
        dataset[field_type] = np.stack(samples)
    
    return dataset