"""Data validation utilities."""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass

from ..types import Field, FieldDataSchema, validate_field_data


@dataclass
class ValidationResult:
    """Result of data validation."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    metadata: Dict[str, Any]


class DataValidator:
    """Validate field data against schemas."""
    
    def __init__(self, schema: FieldDataSchema):
        """
        Initialize data validator.
        
        Parameters
        ----------
        schema : FieldDataSchema
            Schema to validate against
        """
        self.schema = schema
    
    def validate(self, data: Union[np.ndarray, Field]) -> ValidationResult:
        """
        Validate data against schema.
        
        Parameters
        ----------
        data : np.ndarray or Field
            Data to validate
            
        Returns
        -------
        result : ValidationResult
            Validation result
        """
        errors = []
        warnings = []
        metadata = {}
        
        # Extract array data
        if isinstance(data, Field):
            array_data = data.data
            field_metadata = data.metadata or {}
        else:
            array_data = data
            field_metadata = {}
        
        # Basic validation using existing function
        is_valid, basic_errors = validate_field_data(array_data, self.schema)
        errors.extend(basic_errors)
        
        # Additional validations
        additional_errors, additional_warnings, additional_metadata = self._additional_validations(
            array_data, field_metadata
        )
        errors.extend(additional_errors)
        warnings.extend(additional_warnings)
        metadata.update(additional_metadata)
        
        # Check required metadata
        for required_key in self.schema.required_metadata:
            if required_key not in field_metadata:
                errors.append(f"Required metadata key '{required_key}' is missing")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            metadata=metadata
        )
    
    def _additional_validations(self, data: np.ndarray, metadata: Dict[str, Any]) -> Tuple[List[str], List[str], Dict[str, Any]]:
        """
        Perform additional validations.
        
        Returns
        -------
        errors : List[str]
            Additional errors
        warnings : List[str]
            Additional warnings
        metadata : Dict[str, Any]
            Additional metadata
        """
        errors = []
        warnings = []
        result_metadata = {}
        
        # Check for NaN values
        if np.any(np.isnan(data)):
            nan_count = np.sum(np.isnan(data))
            nan_fraction = nan_count / data.size
            
            if nan_fraction > 0.1:
                errors.append(f"Too many NaN values: {nan_fraction:.2%} of data")
            else:
                warnings.append(f"Found {nan_count} NaN values ({nan_fraction:.2%} of data)")
            
            result_metadata['nan_count'] = nan_count
            result_metadata['nan_fraction'] = nan_fraction
        
        # Check for infinite values
        if np.any(np.isinf(data)):
            inf_count = np.sum(np.isinf(data))
            warnings.append(f"Found {inf_count} infinite values")
            result_metadata['inf_count'] = inf_count
        
        # Check data statistics
        data_stats = self._compute_data_statistics(data)
        result_metadata.update(data_stats)
        
        # Check for suspicious patterns
        suspicious_warnings = self._check_suspicious_patterns(data)
        warnings.extend(suspicious_warnings)
        
        return errors, warnings, result_metadata
    
    def _compute_data_statistics(self, data: np.ndarray) -> Dict[str, Any]:
        """Compute data statistics."""
        finite_data = data[np.isfinite(data)]
        
        if len(finite_data) == 0:
            return {'has_finite_data': False}
        
        stats = {
            'has_finite_data': True,
            'mean': float(np.mean(finite_data)),
            'std': float(np.std(finite_data)),
            'min': float(np.min(finite_data)),
            'max': float(np.max(finite_data)),
            'median': float(np.median(finite_data)),
            'percentile_1': float(np.percentile(finite_data, 1)),
            'percentile_99': float(np.percentile(finite_data, 99)),
        }
        
        return stats
    
    def _check_suspicious_patterns(self, data: np.ndarray) -> List[str]:
        """Check for suspicious patterns in data."""
        warnings = []
        
        # Check for constant values
        if np.all(data == data.flat[0]):
            warnings.append("All values are identical (constant field)")
        
        # Check for very low variance
        if np.var(data) < 1e-10:
            warnings.append("Very low variance, field may be nearly constant")
        
        # Check for outliers
        if len(data.shape) >= 2:
            q1 = np.percentile(data, 25)
            q3 = np.percentile(data, 75)
            iqr = q3 - q1
            
            if iqr > 0:
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                
                outliers = np.sum((data < lower_bound) | (data > upper_bound))
                outlier_fraction = outliers / data.size
                
                if outlier_fraction > 0.05:
                    warnings.append(f"High number of outliers: {outlier_fraction:.2%}")
        
        # Check for suspicious spatial patterns (for 2D data)
        if len(data.shape) == 2:
            # Check for periodic patterns
            if self._has_periodic_pattern(data):
                warnings.append("Detected periodic pattern, may be synthetic")
            
            # Check for grid artifacts
            if self._has_grid_artifacts(data):
                warnings.append("Detected grid artifacts")
        
        return warnings
    
    def _has_periodic_pattern(self, data: np.ndarray) -> bool:
        """Check if data has periodic patterns."""
        try:
            # Simple FFT-based check
            fft_data = np.fft.fft2(data)
            power_spectrum = np.abs(fft_data)**2
            
            # Check for dominant frequencies
            power_spectrum[0, 0] = 0  # Remove DC component
            max_power = np.max(power_spectrum)
            mean_power = np.mean(power_spectrum)
            
            return max_power > 10 * mean_power
        except:
            return False
    
    def _has_grid_artifacts(self, data: np.ndarray) -> bool:
        """Check for grid artifacts."""
        try:
            # Check for checkerboard patterns
            diff_h = np.abs(data[:, 1:] - data[:, :-1])
            diff_v = np.abs(data[1:, :] - data[:-1, :])
            
            # High variance in differences might indicate grid artifacts
            return np.var(diff_h) > 2 * np.var(data) or np.var(diff_v) > 2 * np.var(data)
        except:
            return False


class BioelectricDataValidator(DataValidator):
    """Specialized validator for bioelectric data."""
    
    def __init__(self, voltage_range: Tuple[float, float] = (-100, 50)):
        """
        Initialize bioelectric data validator.
        
        Parameters
        ----------
        voltage_range : Tuple[float, float]
            Expected voltage range in mV
        """
        self.voltage_range = voltage_range
        
        # Create schema
        schema = FieldDataSchema(
            shape=(None, None),  # Variable size
            dtype="float64",
            value_range=voltage_range,
            required_metadata=["sampling_rate_hz", "spatial_resolution_mm"]
        )
        
        super().__init__(schema)
    
    def _additional_validations(self, data: np.ndarray, metadata: Dict[str, Any]) -> Tuple[List[str], List[str], Dict[str, Any]]:
        """Additional validations for bioelectric data."""
        errors, warnings, result_metadata = super()._additional_validations(data, metadata)
        
        # Check voltage range
        if np.any(np.isfinite(data)):
            finite_data = data[np.isfinite(data)]
            
            if np.max(finite_data) > self.voltage_range[1]:
                warnings.append(f"Voltage values exceed expected maximum: {np.max(finite_data):.1f} mV")
            
            if np.min(finite_data) < self.voltage_range[0]:
                warnings.append(f"Voltage values below expected minimum: {np.min(finite_data):.1f} mV")
        
        # Check sampling rate
        if "sampling_rate_hz" in metadata:
            sampling_rate = metadata["sampling_rate_hz"]
            if sampling_rate < 1 or sampling_rate > 1000:
                warnings.append(f"Unusual sampling rate: {sampling_rate} Hz")
        
        # Check spatial resolution
        if "spatial_resolution_mm" in metadata:
            spatial_res = metadata["spatial_resolution_mm"]
            if spatial_res < 0.001 or spatial_res > 10:
                warnings.append(f"Unusual spatial resolution: {spatial_res} mm")
        
        # Check for bioelectric-specific patterns
        if self._has_bioelectric_artifacts(data):
            warnings.append("Detected potential bioelectric artifacts")
        
        return errors, warnings, result_metadata
    
    def _has_bioelectric_artifacts(self, data: np.ndarray) -> bool:
        """Check for bioelectric measurement artifacts."""
        try:
            # Check for electrode artifacts (sudden spikes)
            if data.ndim >= 2:
                # Compute gradients
                grad_x = np.gradient(data, axis=-1)
                grad_y = np.gradient(data, axis=-2)
                
                # Large gradients might indicate electrode artifacts
                gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
                
                # Check for extreme gradients
                threshold = 5 * np.std(gradient_magnitude)
                artifacts = np.sum(gradient_magnitude > threshold)
                
                return artifacts > 0.01 * data.size
            
            return False
        except:
            return False


class QualityChecker:
    """Check data quality and suggest improvements."""
    
    def __init__(self):
        """Initialize quality checker."""
        self.checks = [
            self._check_resolution,
            self._check_noise_level,
            self._check_dynamic_range,
            self._check_spatial_coherence,
            self._check_temporal_consistency,
        ]
    
    def check_field(self, field: Union[np.ndarray, Field]) -> Dict[str, Any]:
        """
        Check field data quality.
        
        Parameters
        ----------
        field : np.ndarray or Field
            Field data to check
            
        Returns
        -------
        report : Dict[str, Any]
            Quality report
        """
        if isinstance(field, Field):
            data = field.data
        else:
            data = field
        
        report = {
            'overall_quality': 'unknown',
            'issues': [],
            'recommendations': [],
            'metrics': {}
        }
        
        # Run all checks
        for check in self.checks:
            try:
                check_result = check(data)
                report['metrics'].update(check_result)
            except Exception as e:
                report['issues'].append(f"Check failed: {e}")
        
        # Determine overall quality
        report['overall_quality'] = self._determine_overall_quality(report['metrics'])
        
        # Generate recommendations
        report['recommendations'] = self._generate_recommendations(report['metrics'])
        
        return report
    
    def _check_resolution(self, data: np.ndarray) -> Dict[str, Any]:
        """Check spatial resolution adequacy."""
        metrics = {}
        
        if data.ndim >= 2:
            # Check if resolution is sufficient for features
            # Use gradient-based feature detection
            grad_x = np.gradient(data, axis=-1)
            grad_y = np.gradient(data, axis=-2)
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            
            # Estimate feature size
            feature_threshold = np.percentile(gradient_magnitude, 90)
            feature_pixels = np.sum(gradient_magnitude > feature_threshold)
            
            metrics['feature_pixel_ratio'] = feature_pixels / data.size
            metrics['resolution_adequacy'] = 'good' if feature_pixels > 100 else 'poor'
        
        return metrics
    
    def _check_noise_level(self, data: np.ndarray) -> Dict[str, Any]:
        """Check noise level in data."""
        metrics = {}
        
        if data.ndim >= 2:
            # Estimate noise using high-frequency content
            # Apply Laplacian filter
            laplacian = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
            
            if data.ndim == 2:
                from scipy.ndimage import convolve
                noise_estimate = np.abs(convolve(data, laplacian)).mean()
            else:
                noise_estimate = np.std(data)
            
            signal_strength = np.std(data)
            snr = signal_strength / (noise_estimate + 1e-10)
            
            metrics['noise_estimate'] = noise_estimate
            metrics['signal_strength'] = signal_strength
            metrics['snr'] = snr
            metrics['noise_level'] = 'low' if snr > 10 else 'high' if snr < 3 else 'medium'
        
        return metrics
    
    def _check_dynamic_range(self, data: np.ndarray) -> Dict[str, Any]:
        """Check dynamic range of data."""
        metrics = {}
        
        finite_data = data[np.isfinite(data)]
        if len(finite_data) > 0:
            data_range = np.max(finite_data) - np.min(finite_data)
            data_std = np.std(finite_data)
            
            # Dynamic range relative to standard deviation
            dynamic_range_ratio = data_range / (data_std + 1e-10)
            
            metrics['data_range'] = data_range
            metrics['dynamic_range_ratio'] = dynamic_range_ratio
            metrics['dynamic_range'] = 'good' if dynamic_range_ratio > 6 else 'poor'
        
        return metrics
    
    def _check_spatial_coherence(self, data: np.ndarray) -> Dict[str, Any]:
        """Check spatial coherence of field."""
        metrics = {}
        
        if data.ndim >= 2:
            # Compute spatial autocorrelation
            from scipy.signal import correlate2d
            
            # Use central region for autocorrelation
            center_h = data.shape[0] // 2
            center_w = data.shape[1] // 2
            size = min(32, data.shape[0] // 4, data.shape[1] // 4)
            
            if size > 0:
                region = data[center_h-size:center_h+size, center_w-size:center_w+size]
                autocorr = correlate2d(region, region, mode='same')
                
                # Coherence measure
                coherence = autocorr.max() / (autocorr.mean() + 1e-10)
                
                metrics['spatial_coherence'] = coherence
                metrics['coherence_quality'] = 'good' if coherence > 2 else 'poor'
        
        return metrics
    
    def _check_temporal_consistency(self, data: np.ndarray) -> Dict[str, Any]:
        """Check temporal consistency for time series data."""
        metrics = {}
        
        if data.ndim == 3:  # Time series
            # Check frame-to-frame consistency
            frame_diffs = []
            for i in range(data.shape[0] - 1):
                diff = np.mean(np.abs(data[i+1] - data[i]))
                frame_diffs.append(diff)
            
            if frame_diffs:
                mean_diff = np.mean(frame_diffs)
                std_diff = np.std(frame_diffs)
                
                # Consistency measure
                consistency = mean_diff / (std_diff + 1e-10)
                
                metrics['temporal_consistency'] = consistency
                metrics['consistency_quality'] = 'good' if consistency > 2 else 'poor'
        
        return metrics
    
    def _determine_overall_quality(self, metrics: Dict[str, Any]) -> str:
        """Determine overall data quality."""
        quality_indicators = []
        
        # Collect quality indicators
        for key, value in metrics.items():
            if key.endswith('_quality') or key.endswith('_adequacy') or key.endswith('_level'):
                quality_indicators.append(value)
        
        if not quality_indicators:
            return 'unknown'
        
        # Count good vs poor quality indicators
        good_count = quality_indicators.count('good')
        poor_count = quality_indicators.count('poor')
        
        if good_count > poor_count:
            return 'good'
        elif poor_count > good_count:
            return 'poor'
        else:
            return 'medium'
    
    def _generate_recommendations(self, metrics: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on metrics."""
        recommendations = []
        
        # SNR recommendations
        if 'snr' in metrics:
            snr = metrics['snr']
            if snr < 3:
                recommendations.append("Consider denoising the data (low SNR)")
            elif snr < 10:
                recommendations.append("Moderate noise level, denoising may help")
        
        # Resolution recommendations
        if 'resolution_adequacy' in metrics:
            if metrics['resolution_adequacy'] == 'poor':
                recommendations.append("Consider increasing spatial resolution")
        
        # Dynamic range recommendations
        if 'dynamic_range' in metrics:
            if metrics['dynamic_range'] == 'poor':
                recommendations.append("Dynamic range is limited, check measurement setup")
        
        # Coherence recommendations
        if 'coherence_quality' in metrics:
            if metrics['coherence_quality'] == 'poor':
                recommendations.append("Low spatial coherence, check for artifacts")
        
        # Temporal consistency recommendations
        if 'consistency_quality' in metrics:
            if metrics['consistency_quality'] == 'poor':
                recommendations.append("Inconsistent temporal evolution, check for drift")
        
        return recommendations