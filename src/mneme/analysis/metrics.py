"""Evaluation metrics for field analysis."""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr, spearmanr

from ..types import Field, PersistenceDiagram, Attractor, ReconstructionResult


def compute_reconstruction_metrics(
    original: np.ndarray,
    reconstructed: np.ndarray,
    uncertainty: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Compute reconstruction quality metrics.
    
    Parameters
    ----------
    original : np.ndarray
        Original field
    reconstructed : np.ndarray
        Reconstructed field
    uncertainty : np.ndarray, optional
        Reconstruction uncertainty
        
    Returns
    -------
    metrics : Dict[str, float]
        Reconstruction metrics
    """
    metrics = {}
    
    # Ensure same shape
    if original.shape != reconstructed.shape:
        raise ValueError("Original and reconstructed fields must have same shape")
    
    # Mean squared error
    mse = np.mean((original - reconstructed)**2)
    metrics['mse'] = mse
    
    # Root mean squared error
    metrics['rmse'] = np.sqrt(mse)
    
    # Mean absolute error
    metrics['mae'] = np.mean(np.abs(original - reconstructed))
    
    # Normalized MSE
    original_var = np.var(original)
    if original_var > 0:
        metrics['nmse'] = mse / original_var
    else:
        metrics['nmse'] = 0
    
    # Pearson correlation
    orig_flat = original.flatten()
    recon_flat = reconstructed.flatten()
    
    if len(orig_flat) > 1 and np.var(orig_flat) > 0 and np.var(recon_flat) > 0:
        correlation, _ = pearsonr(orig_flat, recon_flat)
        metrics['correlation'] = correlation
    else:
        metrics['correlation'] = 0
    
    # Signal-to-noise ratio
    signal_power = np.mean(original**2)
    noise_power = mse
    if noise_power > 0:
        metrics['snr'] = 10 * np.log10(signal_power / noise_power)
    else:
        metrics['snr'] = np.inf
    
    # Structural similarity index (simplified)
    metrics['ssim'] = compute_ssim(original, reconstructed)
    
    # Uncertainty-based metrics
    if uncertainty is not None:
        # Coverage probability (for Gaussian uncertainties)
        residuals = np.abs(original - reconstructed)
        coverage = np.mean(residuals <= 2 * uncertainty)  # ~95% for Gaussian
        metrics['coverage_95'] = coverage
        
        # Mean normalized residual
        mean_uncertainty = np.mean(uncertainty)
        if mean_uncertainty > 0:
            metrics['normalized_residual'] = np.mean(residuals) / mean_uncertainty
        else:
            metrics['normalized_residual'] = 0
    
    return metrics


def compute_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Compute structural similarity index (simplified version).
    
    Parameters
    ----------
    img1, img2 : np.ndarray
        Images to compare
        
    Returns
    -------
    ssim : float
        SSIM value
    """
    # Constants
    C1 = 0.01**2
    C2 = 0.03**2
    
    # Compute means
    mu1 = np.mean(img1)
    mu2 = np.mean(img2)
    
    # Compute variances and covariance
    var1 = np.var(img1)
    var2 = np.var(img2)
    cov = np.mean((img1 - mu1) * (img2 - mu2))
    
    # SSIM formula
    numerator = (2 * mu1 * mu2 + C1) * (2 * cov + C2)
    denominator = (mu1**2 + mu2**2 + C1) * (var1 + var2 + C2)
    
    if denominator > 0:
        ssim = numerator / denominator
    else:
        ssim = 0
    
    return ssim


def compute_topological_metrics(
    diagrams: List[PersistenceDiagram],
    reference_diagrams: Optional[List[PersistenceDiagram]] = None
) -> Dict[str, Any]:
    """
    Compute topological analysis metrics.
    
    Parameters
    ----------
    diagrams : List[PersistenceDiagram]
        Persistence diagrams
    reference_diagrams : List[PersistenceDiagram], optional
        Reference diagrams for comparison
        
    Returns
    -------
    metrics : Dict[str, Any]
        Topological metrics
    """
    metrics = {}
    
    # Per-dimension metrics
    for i, diagram in enumerate(diagrams):
        dim_metrics = {}
        
        # Number of features
        dim_metrics['n_features'] = len(diagram.points)
        
        if len(diagram.points) > 0:
            # Persistence statistics
            persistence = diagram.persistence
            dim_metrics['total_persistence'] = np.sum(persistence)
            dim_metrics['max_persistence'] = np.max(persistence)
            dim_metrics['mean_persistence'] = np.mean(persistence)
            dim_metrics['std_persistence'] = np.std(persistence)
            
            # Persistence entropy
            if np.sum(persistence) > 0:
                p_norm = persistence / np.sum(persistence)
                entropy = -np.sum(p_norm * np.log(p_norm + 1e-10))
                dim_metrics['entropy'] = entropy
            else:
                dim_metrics['entropy'] = 0
            
            # Birth and death statistics
            births = diagram.points[:, 0]
            deaths = diagram.points[:, 1]
            
            dim_metrics['birth_range'] = np.max(births) - np.min(births)
            dim_metrics['death_range'] = np.max(deaths) - np.min(deaths)
            
        else:
            # Empty diagram
            dim_metrics.update({
                'total_persistence': 0,
                'max_persistence': 0,
                'mean_persistence': 0,
                'std_persistence': 0,
                'entropy': 0,
                'birth_range': 0,
                'death_range': 0
            })
        
        metrics[f'dim_{i}'] = dim_metrics
    
    # Overall metrics
    total_features = sum(len(d.points) for d in diagrams)
    metrics['total_features'] = total_features
    
    # Comparison with reference diagrams
    if reference_diagrams is not None:
        from ..core.topology import compute_wasserstein_distance, compute_bottleneck_distance
        
        comparison_metrics = {}
        
        for i, (diag, ref_diag) in enumerate(zip(diagrams, reference_diagrams)):
            # Wasserstein distance
            try:
                w_dist = compute_wasserstein_distance(diag, ref_diag)
                comparison_metrics[f'wasserstein_dim_{i}'] = w_dist
            except:
                comparison_metrics[f'wasserstein_dim_{i}'] = np.nan
            
            # Bottleneck distance
            try:
                b_dist = compute_bottleneck_distance(diag, ref_diag)
                comparison_metrics[f'bottleneck_dim_{i}'] = b_dist
            except:
                comparison_metrics[f'bottleneck_dim_{i}'] = np.nan
        
        metrics['comparison'] = comparison_metrics
    
    return metrics


def compute_attractor_metrics(
    attractors: List[Attractor],
    trajectory: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Compute attractor analysis metrics.
    
    Parameters
    ----------
    attractors : List[Attractor]
        Detected attractors
    trajectory : np.ndarray, optional
        Original trajectory
        
    Returns
    -------
    metrics : Dict[str, Any]
        Attractor metrics
    """
    metrics = {}
    
    # Basic statistics
    metrics['n_attractors'] = len(attractors)
    
    if len(attractors) == 0:
        return metrics
    
    # Attractor types
    attractor_types = [a.type.value for a in attractors]
    unique_types, counts = np.unique(attractor_types, return_counts=True)
    metrics['attractor_types'] = dict(zip(unique_types, counts))
    
    # Basin size statistics
    basin_sizes = [a.basin_size for a in attractors]
    metrics['basin_sizes'] = {
        'mean': np.mean(basin_sizes),
        'std': np.std(basin_sizes),
        'min': np.min(basin_sizes),
        'max': np.max(basin_sizes),
        'total': np.sum(basin_sizes)
    }
    
    # Attractor dimensions (if available)
    dimensions = [a.dimension for a in attractors if a.dimension is not None]
    if dimensions:
        metrics['dimensions'] = {
            'mean': np.mean(dimensions),
            'std': np.std(dimensions),
            'min': np.min(dimensions),
            'max': np.max(dimensions)
        }
    
    # Lyapunov exponents (if available)
    lyapunov_exponents = []
    for a in attractors:
        if a.lyapunov_exponents is not None:
            lyapunov_exponents.extend(a.lyapunov_exponents)
    
    if lyapunov_exponents:
        metrics['lyapunov_exponents'] = {
            'mean': np.mean(lyapunov_exponents),
            'std': np.std(lyapunov_exponents),
            'min': np.min(lyapunov_exponents),
            'max': np.max(lyapunov_exponents)
        }
    
    # Attractor separation (distance between centers)
    if len(attractors) > 1:
        centers = np.array([a.center for a in attractors])
        distances = pdist(centers)
        
        metrics['separation'] = {
            'mean': np.mean(distances),
            'std': np.std(distances),
            'min': np.min(distances),
            'max': np.max(distances)
        }
    
    # Coverage metrics (if trajectory provided)
    if trajectory is not None:
        # Fraction of trajectory covered by attractors
        covered_points = 0
        for attractor in attractors:
            if attractor.trajectory_indices is not None:
                covered_points += len(attractor.trajectory_indices)
        
        metrics['trajectory_coverage'] = covered_points / len(trajectory)
    
    return metrics


def compute_field_statistics(field: Union[Field, np.ndarray]) -> Dict[str, Any]:
    """
    Compute basic field statistics.
    
    Parameters
    ----------
    field : Field or np.ndarray
        Field data
        
    Returns
    -------
    stats : Dict[str, Any]
        Field statistics
    """
    if isinstance(field, Field):
        data = field.data
    else:
        data = field
    
    finite_data = data[np.isfinite(data)]
    
    if len(finite_data) == 0:
        return {'has_data': False}
    
    stats = {
        'has_data': True,
        'shape': data.shape,
        'size': data.size,
        'n_finite': len(finite_data),
        'n_nan': np.sum(np.isnan(data)),
        'n_inf': np.sum(np.isinf(data)),
        'mean': np.mean(finite_data),
        'std': np.std(finite_data),
        'min': np.min(finite_data),
        'max': np.max(finite_data),
        'median': np.median(finite_data),
        'range': np.max(finite_data) - np.min(finite_data)
    }
    
    # Percentiles
    percentiles = [1, 5, 10, 25, 75, 90, 95, 99]
    for p in percentiles:
        stats[f'p{p}'] = np.percentile(finite_data, p)
    
    # Skewness and kurtosis
    from scipy.stats import skew, kurtosis
    stats['skewness'] = skew(finite_data)
    stats['kurtosis'] = kurtosis(finite_data)
    
    # Spatial statistics (for 2D fields)
    if len(data.shape) >= 2:
        # Gradient magnitude
        if data.ndim == 2:
            grad_y, grad_x = np.gradient(data)
            grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        else:
            # For 3D, use last frame
            grad_y, grad_x = np.gradient(data[-1] if data.ndim == 3 else data[0])
            grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        stats['gradient_mean'] = np.mean(grad_magnitude[np.isfinite(grad_magnitude)])
        stats['gradient_std'] = np.std(grad_magnitude[np.isfinite(grad_magnitude)])
        
        # Spatial correlation
        if data.ndim == 2:
            spatial_corr = compute_spatial_correlation(data)
            stats['spatial_correlation'] = spatial_corr
    
    # Temporal statistics (for 3D fields)
    if data.ndim == 3:
        # Temporal variance
        temporal_var = np.var(data, axis=0)
        stats['temporal_variance_mean'] = np.mean(temporal_var[np.isfinite(temporal_var)])
        
        # Frame-to-frame correlation
        correlations = []
        for t in range(data.shape[0] - 1):
            frame1 = data[t].flatten()
            frame2 = data[t + 1].flatten()
            
            mask = np.isfinite(frame1) & np.isfinite(frame2)
            if np.sum(mask) > 1:
                corr, _ = pearsonr(frame1[mask], frame2[mask])
                correlations.append(corr)
        
        if correlations:
            stats['temporal_correlation_mean'] = np.mean(correlations)
            stats['temporal_correlation_std'] = np.std(correlations)
    
    return stats


def compute_spatial_correlation(field: np.ndarray, max_distance: int = 50) -> Dict[str, Any]:
    """
    Compute spatial correlation function.
    
    Parameters
    ----------
    field : np.ndarray
        2D field
    max_distance : int
        Maximum distance for correlation
        
    Returns
    -------
    correlation : Dict[str, Any]
        Spatial correlation results
    """
    if field.ndim != 2:
        raise ValueError("Spatial correlation requires 2D field")
    
    h, w = field.shape
    correlations = []
    distances = []
    
    # Sample points for correlation calculation
    n_samples = min(1000, h * w // 10)
    y_coords = np.random.randint(0, h, n_samples)
    x_coords = np.random.randint(0, w, n_samples)
    
    for i in range(n_samples):
        y1, x1 = y_coords[i], x_coords[i]
        
        # Find points within max_distance
        y_range = slice(max(0, y1 - max_distance), min(h, y1 + max_distance + 1))
        x_range = slice(max(0, x1 - max_distance), min(w, x1 + max_distance + 1))
        
        # Calculate distances and correlations
        for y2 in range(y_range.start, y_range.stop):
            for x2 in range(x_range.start, x_range.stop):
                if y2 == y1 and x2 == x1:
                    continue
                
                distance = np.sqrt((y2 - y1)**2 + (x2 - x1)**2)
                if distance <= max_distance:
                    distances.append(distance)
                    correlations.append(field[y1, x1] * field[y2, x2])
    
    # Bin by distance
    distances = np.array(distances)
    correlations = np.array(correlations)
    
    distance_bins = np.arange(0, max_distance + 1, 1)
    binned_correlations = []
    
    for i in range(len(distance_bins) - 1):
        mask = (distances >= distance_bins[i]) & (distances < distance_bins[i + 1])
        if np.sum(mask) > 0:
            binned_correlations.append(np.mean(correlations[mask]))
        else:
            binned_correlations.append(0)
    
    return {
        'distances': distance_bins[:-1],
        'correlations': np.array(binned_correlations),
        'correlation_length': estimate_correlation_length(distance_bins[:-1], binned_correlations)
    }


def estimate_correlation_length(distances: np.ndarray, correlations: np.ndarray) -> float:
    """Estimate correlation length from correlation function."""
    # Find where correlation drops to 1/e of maximum
    max_corr = np.max(correlations)
    threshold = max_corr / np.e
    
    # Find first point below threshold
    below_threshold = np.where(correlations < threshold)[0]
    
    if len(below_threshold) > 0:
        return distances[below_threshold[0]]
    else:
        return distances[-1]  # Correlation length is longer than measured range


def compute_pipeline_metrics(
    pipeline_results: Dict[str, Any],
    reference_results: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Compute comprehensive pipeline metrics.
    
    Parameters
    ----------
    pipeline_results : Dict[str, Any]
        Pipeline stage results
    reference_results : Dict[str, Any], optional
        Reference results for comparison
        
    Returns
    -------
    metrics : Dict[str, Any]
        Pipeline metrics
    """
    metrics = {}
    
    # Execution time metrics
    execution_times = []
    for stage_name, stage_result in pipeline_results.items():
        if isinstance(stage_result, dict) and 'computation_time' in stage_result:
            execution_times.append(stage_result['computation_time'])
    
    if execution_times:
        metrics['execution_times'] = {
            'total': sum(execution_times),
            'mean': np.mean(execution_times),
            'max': np.max(execution_times),
            'stages': execution_times
        }
    
    # Stage-specific metrics
    for stage_name, stage_result in pipeline_results.items():
        if isinstance(stage_result, dict):
            stage_metrics = {}
            
            # Extract numerical metrics
            for key, value in stage_result.items():
                if isinstance(value, (int, float)):
                    stage_metrics[key] = value
                elif isinstance(value, np.ndarray) and value.size == 1:
                    stage_metrics[key] = value.item()
            
            if stage_metrics:
                metrics[f'{stage_name}_metrics'] = stage_metrics
    
    # Quality metrics
    if 'quality_check' in pipeline_results:
        quality_result = pipeline_results['quality_check']
        metrics['quality_score'] = compute_quality_score(quality_result)
    
    return metrics


def compute_quality_score(quality_result: Dict[str, Any]) -> float:
    """Compute overall quality score from quality check results."""
    if 'metrics' not in quality_result:
        return 0.0
    
    metrics = quality_result['metrics']
    score = 0.0
    n_metrics = 0
    
    # SNR contribution
    if 'snr' in metrics:
        snr = metrics['snr']
        score += min(1.0, snr / 10.0)  # Normalize to [0, 1]
        n_metrics += 1
    
    # Resolution contribution
    if 'resolution_adequacy' in metrics:
        if metrics['resolution_adequacy'] == 'good':
            score += 1.0
        elif metrics['resolution_adequacy'] == 'medium':
            score += 0.5
        n_metrics += 1
    
    # Dynamic range contribution
    if 'dynamic_range' in metrics:
        if metrics['dynamic_range'] == 'good':
            score += 1.0
        elif metrics['dynamic_range'] == 'medium':
            score += 0.5
        n_metrics += 1
    
    # Normalize score
    if n_metrics > 0:
        score /= n_metrics
    
    return score