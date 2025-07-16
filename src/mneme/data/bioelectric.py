"""Specialized handlers for bioelectric data."""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
from scipy.spatial import ConvexHull
from scipy.interpolate import griddata

from ..types import BioelectricMeasurement, Field


@dataclass
class BioelectricField:
    """Bioelectric field representation."""
    voltage_field: np.ndarray
    timestamps: Optional[np.ndarray] = None
    electrode_positions: Optional[np.ndarray] = None
    sampling_rate: Optional[float] = None
    spatial_resolution: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None
    
    @property
    def shape(self) -> Tuple[int, ...]:
        """Get field shape."""
        return self.voltage_field.shape
    
    @property
    def is_temporal(self) -> bool:
        """Check if field has temporal dimension."""
        return self.voltage_field.ndim == 3
    
    def get_spatial_extent(self) -> Tuple[float, float, float, float]:
        """Get spatial extent (min_x, max_x, min_y, max_y)."""
        if self.electrode_positions is not None:
            positions = self.electrode_positions
            return (
                positions[:, 0].min(),
                positions[:, 0].max(),
                positions[:, 1].min(),
                positions[:, 1].max()
            )
        else:
            # Use image coordinates
            h, w = self.voltage_field.shape[-2:]
            return (0, w, 0, h)
    
    def interpolate_to_grid(
        self,
        grid_resolution: Tuple[int, int] = (256, 256),
        method: str = 'linear'
    ) -> 'BioelectricField':
        """Interpolate field to regular grid."""
        if self.electrode_positions is None:
            # Already on grid
            return self
        
        # Create target grid
        extent = self.get_spatial_extent()
        x_grid = np.linspace(extent[0], extent[1], grid_resolution[1])
        y_grid = np.linspace(extent[2], extent[3], grid_resolution[0])
        X, Y = np.meshgrid(x_grid, y_grid)
        
        # Interpolate
        if self.is_temporal:
            # Interpolate each time frame
            interpolated = np.zeros((self.voltage_field.shape[0], *grid_resolution))
            
            for t in range(self.voltage_field.shape[0]):
                interpolated[t] = griddata(
                    self.electrode_positions,
                    self.voltage_field[t],
                    (X, Y),
                    method=method,
                    fill_value=0
                )
        else:
            # Single frame interpolation
            interpolated = griddata(
                self.electrode_positions,
                self.voltage_field,
                (X, Y),
                method=method,
                fill_value=0
            )
        
        return BioelectricField(
            voltage_field=interpolated,
            timestamps=self.timestamps,
            electrode_positions=None,  # Now on regular grid
            sampling_rate=self.sampling_rate,
            spatial_resolution=self.spatial_resolution,
            metadata=self.metadata
        )
    
    def compute_gradients(self) -> Dict[str, np.ndarray]:
        """Compute spatial gradients of voltage field."""
        if self.is_temporal:
            # Compute gradients for each time frame
            grad_x = np.zeros_like(self.voltage_field)
            grad_y = np.zeros_like(self.voltage_field)
            
            for t in range(self.voltage_field.shape[0]):
                grad_y[t], grad_x[t] = np.gradient(self.voltage_field[t])
        else:
            grad_y, grad_x = np.gradient(self.voltage_field)
        
        # Compute magnitude
        grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        return {
            'grad_x': grad_x,
            'grad_y': grad_y,
            'magnitude': grad_magnitude
        }
    
    def detect_features(self, threshold_percentile: float = 90) -> Dict[str, Any]:
        """Detect features in bioelectric field."""
        # Compute gradients
        gradients = self.compute_gradients()
        
        # Threshold for feature detection
        threshold = np.percentile(gradients['magnitude'], threshold_percentile)
        
        # Find features
        if self.is_temporal:
            features = []
            for t in range(self.voltage_field.shape[0]):
                feature_mask = gradients['magnitude'][t] > threshold
                feature_positions = np.where(feature_mask)
                
                if len(feature_positions[0]) > 0:
                    features.append({
                        'time': t,
                        'positions': np.column_stack(feature_positions),
                        'n_features': len(feature_positions[0])
                    })
        else:
            feature_mask = gradients['magnitude'] > threshold
            feature_positions = np.where(feature_mask)
            
            features = {
                'positions': np.column_stack(feature_positions),
                'n_features': len(feature_positions[0])
            }
        
        return features
    
    def extract_anterior_posterior_profile(self) -> np.ndarray:
        """Extract anterior-posterior profile."""
        if self.is_temporal:
            # Average over time
            field = np.mean(self.voltage_field, axis=0)
        else:
            field = self.voltage_field
        
        # Average across lateral dimension
        ap_profile = np.mean(field, axis=1)
        
        return ap_profile
    
    def extract_lateral_profile(self) -> np.ndarray:
        """Extract lateral profile."""
        if self.is_temporal:
            # Average over time
            field = np.mean(self.voltage_field, axis=0)
        else:
            field = self.voltage_field
        
        # Average across anterior-posterior dimension
        lateral_profile = np.mean(field, axis=0)
        
        return lateral_profile
    
    def compute_polarity_index(self) -> float:
        """Compute polarity index (difference between anterior and posterior)."""
        ap_profile = self.extract_anterior_posterior_profile()
        
        # Anterior (first 20%) vs posterior (last 20%)
        n_points = len(ap_profile)
        anterior_region = ap_profile[:n_points//5]
        posterior_region = ap_profile[-n_points//5:]
        
        polarity_index = np.mean(anterior_region) - np.mean(posterior_region)
        
        return polarity_index
    
    def to_field(self) -> Field:
        """Convert to generic Field object."""
        return Field(
            data=self.voltage_field,
            coordinates=self.electrode_positions,
            resolution=self.voltage_field.shape[-2:] if not self.is_temporal else None,
            metadata=self.metadata
        )


def create_bioelectric_field(
    voltage_data: np.ndarray,
    timestamps: Optional[np.ndarray] = None,
    electrode_positions: Optional[np.ndarray] = None,
    **kwargs
) -> BioelectricField:
    """
    Create BioelectricField from raw data.
    
    Parameters
    ----------
    voltage_data : np.ndarray
        Voltage measurements
    timestamps : np.ndarray, optional
        Timestamps for temporal data
    electrode_positions : np.ndarray, optional
        Electrode positions
    **kwargs
        Additional metadata
        
    Returns
    -------
    field : BioelectricField
        Bioelectric field object
    """
    return BioelectricField(
        voltage_field=voltage_data,
        timestamps=timestamps,
        electrode_positions=electrode_positions,
        **kwargs
    )


def analyze_bioelectric_patterns(field: BioelectricField) -> Dict[str, Any]:
    """
    Analyze patterns in bioelectric field.
    
    Parameters
    ----------
    field : BioelectricField
        Bioelectric field to analyze
        
    Returns
    -------
    analysis : Dict[str, Any]
        Analysis results
    """
    analysis = {}
    
    # Basic statistics
    analysis['mean_voltage'] = np.mean(field.voltage_field)
    analysis['std_voltage'] = np.std(field.voltage_field)
    analysis['min_voltage'] = np.min(field.voltage_field)
    analysis['max_voltage'] = np.max(field.voltage_field)
    
    # Polarity analysis
    analysis['polarity_index'] = field.compute_polarity_index()
    
    # Gradient analysis
    gradients = field.compute_gradients()
    analysis['mean_gradient'] = np.mean(gradients['magnitude'])
    analysis['max_gradient'] = np.max(gradients['magnitude'])
    
    # Feature detection
    features = field.detect_features()
    if field.is_temporal:
        analysis['n_features_per_frame'] = [f['n_features'] for f in features]
        analysis['total_features'] = sum(analysis['n_features_per_frame'])
    else:
        analysis['n_features'] = features['n_features']
    
    # Spatial extent
    analysis['spatial_extent'] = field.get_spatial_extent()
    
    # Temporal analysis (if applicable)
    if field.is_temporal:
        # Compute temporal variance
        temporal_variance = np.var(field.voltage_field, axis=0)
        analysis['temporal_variance'] = {
            'mean': np.mean(temporal_variance),
            'max': np.max(temporal_variance),
            'spatial_pattern': temporal_variance
        }
        
        # Compute temporal correlation
        if field.voltage_field.shape[0] > 1:
            # Correlation between consecutive frames
            correlations = []
            for t in range(field.voltage_field.shape[0] - 1):
                corr = np.corrcoef(
                    field.voltage_field[t].flatten(),
                    field.voltage_field[t + 1].flatten()
                )[0, 1]
                correlations.append(corr)
            
            analysis['temporal_correlation'] = {
                'mean': np.mean(correlations),
                'std': np.std(correlations),
                'values': correlations
            }
    
    return analysis


def detect_bioelectric_events(
    field: BioelectricField,
    threshold_type: str = 'gradient',
    threshold_value: Optional[float] = None
) -> List[Dict[str, Any]]:
    """
    Detect bioelectric events in field.
    
    Parameters
    ----------
    field : BioelectricField
        Bioelectric field
    threshold_type : str
        Type of threshold ('gradient', 'voltage', 'variance')
    threshold_value : float, optional
        Threshold value (auto-computed if None)
        
    Returns
    -------
    events : List[Dict[str, Any]]
        Detected events
    """
    if not field.is_temporal:
        raise ValueError("Event detection requires temporal data")
    
    events = []
    
    for t in range(field.voltage_field.shape[0]):
        frame = field.voltage_field[t]
        
        if threshold_type == 'gradient':
            gradients = field.compute_gradients()
            metric = gradients['magnitude'][t]
            if threshold_value is None:
                threshold_value = np.percentile(metric, 95)
        
        elif threshold_type == 'voltage':
            metric = np.abs(frame)
            if threshold_value is None:
                threshold_value = np.percentile(metric, 95)
        
        elif threshold_type == 'variance':
            if t > 0:
                metric = np.abs(frame - field.voltage_field[t-1])
                if threshold_value is None:
                    threshold_value = np.percentile(metric, 95)
            else:
                continue
        
        # Find events
        event_mask = metric > threshold_value
        event_positions = np.where(event_mask)
        
        if len(event_positions[0]) > 0:
            event = {
                'time': t,
                'timestamp': field.timestamps[t] if field.timestamps is not None else t,
                'positions': np.column_stack(event_positions),
                'intensities': metric[event_mask],
                'n_events': len(event_positions[0])
            }
            events.append(event)
    
    return events


def compute_bioelectric_connectivity(
    field: BioelectricField,
    correlation_threshold: float = 0.7
) -> np.ndarray:
    """
    Compute spatial connectivity based on temporal correlations.
    
    Parameters
    ----------
    field : BioelectricField
        Bioelectric field
    correlation_threshold : float
        Correlation threshold for connectivity
        
    Returns
    -------
    connectivity : np.ndarray
        Connectivity matrix
    """
    if not field.is_temporal:
        raise ValueError("Connectivity computation requires temporal data")
    
    # Flatten spatial dimensions
    h, w = field.voltage_field.shape[-2:]
    n_pixels = h * w
    
    # Reshape to (time, pixels)
    field_flat = field.voltage_field.reshape(field.voltage_field.shape[0], -1)
    
    # Compute correlation matrix
    connectivity = np.corrcoef(field_flat.T)
    
    # Apply threshold
    connectivity[connectivity < correlation_threshold] = 0
    
    return connectivity