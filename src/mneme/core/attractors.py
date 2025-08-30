"""Attractor detection and characterization methods."""

from typing import List, Optional, Dict, Any, Tuple
import numpy as np
from abc import ABC, abstractmethod

from ..types import Attractor, AttractorType, TimeSeries


class BaseAttractorDetector(ABC):
    """Abstract base class for attractor detection methods."""
    
    def __init__(self, threshold: float = 0.1):
        """
        Initialize attractor detector.
        
        Parameters
        ----------
        threshold : float
            Detection threshold
        """
        self.threshold = threshold
        
    @abstractmethod
    def detect(self, trajectory: np.ndarray) -> List[Attractor]:
        """
        Detect attractors in phase space trajectory.
        
        Parameters
        ----------
        trajectory : np.ndarray
            Phase space trajectory, shape (n_timesteps, n_dimensions)
            
        Returns
        -------
        attractors : List[Attractor]
            Detected attractors
        """
        raise NotImplementedError
        
    @abstractmethod
    def characterize(self, attractor: Attractor, trajectory: np.ndarray) -> Dict[str, Any]:
        """
        Compute detailed properties of an attractor.
        
        Parameters
        ----------
        attractor : Attractor
            Attractor to characterize
        trajectory : np.ndarray
            Full trajectory data
            
        Returns
        -------
        properties : Dict[str, Any]
            Attractor properties
        """
        raise NotImplementedError


class RecurrenceAnalysis(BaseAttractorDetector):
    """Detect attractors using recurrence analysis."""
    
    def __init__(
        self,
        threshold: float = 0.1,
        min_persistence: float = 0.1,
        embedding_dimension: int = 3,
        time_delay: int = 1
    ):
        """
        Initialize recurrence-based detector.
        
        Parameters
        ----------
        threshold : float
            Recurrence threshold
        min_persistence : float
            Minimum attractor persistence
        embedding_dimension : int
            Embedding dimension for delay embedding
        time_delay : int
            Time delay for embedding
        """
        super().__init__(threshold)
        self.min_persistence = min_persistence
        self.embedding_dimension = embedding_dimension
        self.time_delay = time_delay
        
    def detect(self, trajectory: np.ndarray) -> List[Attractor]:
        """Detect attractors using recurrence analysis."""
        # Embed trajectory if needed
        if trajectory.ndim == 1:
            embedded_trajectory = embed_trajectory(trajectory, self.embedding_dimension, self.time_delay)
        else:
            embedded_trajectory = trajectory
        
        # Compute recurrence matrix
        recurrence_matrix = self.compute_recurrence_matrix(embedded_trajectory)
        
        # Find recurrent regions
        recurrent_regions = self._find_recurrent_regions(recurrence_matrix)
        
        # Cluster trajectory points to find attractors
        attractors = self._cluster_attractors(embedded_trajectory, recurrent_regions)
        
        return attractors
    
    def _find_recurrent_regions(self, recurrence_matrix: np.ndarray) -> List[np.ndarray]:
        """Find recurrent regions in the recurrence matrix."""
        from scipy import ndimage
        
        # Apply morphological operations to find connected regions
        structure = np.ones((3, 3))
        labeled, num_features = ndimage.label(recurrence_matrix, structure=structure)
        
        # Extract regions that are large enough to be considered attractors
        min_size = int(self.min_persistence * recurrence_matrix.shape[0])
        regions = []
        
        for i in range(1, num_features + 1):
            region_mask = labeled == i
            if np.sum(region_mask) >= min_size:
                regions.append(region_mask)
        
        return regions
    
    def _cluster_attractors(self, trajectory: np.ndarray, recurrent_regions: List[np.ndarray]) -> List[Attractor]:
        """Cluster trajectory points to identify attractors."""
        from sklearn.cluster import DBSCAN
        
        attractors = []
        
        for region in recurrent_regions:
            # Get indices of points in this region
            indices = np.where(region)
            if len(indices[0]) == 0:
                continue
            
            # Extract trajectory points in this region
            region_points = trajectory[indices[0]]
            
            # Cluster points within the region
            clustering = DBSCAN(eps=self.threshold, min_samples=5)
            labels = clustering.fit_predict(region_points)
            
            # Create attractor for each cluster
            for label in np.unique(labels):
                if label == -1:  # Noise points
                    continue
                
                cluster_mask = labels == label
                cluster_points = region_points[cluster_mask]
                
                if len(cluster_points) < 5:  # Too small to be an attractor
                    continue
                
                # Compute attractor properties
                center = np.mean(cluster_points, axis=0)
                basin_size = len(cluster_points) / len(trajectory)
                
                # Classify attractor type (simplified)
                attractor_type = self._classify_attractor_simple(cluster_points)
                
                attractor = Attractor(
                    type=attractor_type,
                    center=center,
                    basin_size=basin_size,
                    trajectory_indices=indices[0][cluster_mask].tolist()
                )
                
                attractors.append(attractor)
        
        return attractors
    
    def _classify_attractor_simple(self, points: np.ndarray) -> AttractorType:
        """Simple attractor classification based on point distribution."""
        # Very simple classification based on variance
        variance = np.var(points, axis=0)
        total_variance = np.sum(variance)
        
        if total_variance < 0.01:
            return AttractorType.FIXED_POINT
        elif total_variance < 0.1:
            return AttractorType.LIMIT_CYCLE
        else:
            return AttractorType.STRANGE
        
    def characterize(self, attractor: Attractor, trajectory: np.ndarray) -> Dict[str, Any]:
        """Characterize attractor using recurrence quantification."""
        if attractor.trajectory_indices is None:
            return {}
        
        # Extract attractor trajectory
        attractor_trajectory = trajectory[attractor.trajectory_indices]
        
        # Compute recurrence matrix for attractor
        recurrence_matrix = self.compute_recurrence_matrix(attractor_trajectory)
        
        # Recurrence quantification analysis
        properties = {}
        
        # Recurrence rate
        properties['recurrence_rate'] = np.sum(recurrence_matrix) / recurrence_matrix.size
        
        # Determinism (fraction of recurrent points forming diagonal lines)
        properties['determinism'] = self._compute_determinism(recurrence_matrix)
        
        # Average diagonal line length
        properties['avg_diagonal_length'] = self._compute_avg_diagonal_length(recurrence_matrix)
        
        # Laminarity (fraction of recurrent points forming vertical lines)
        properties['laminarity'] = self._compute_laminarity(recurrence_matrix)
        
        # Entropy of diagonal line lengths
        properties['entropy'] = self._compute_entropy(recurrence_matrix)
        
        return properties
    
    def _compute_determinism(self, recurrence_matrix: np.ndarray) -> float:
        """Compute determinism measure."""
        # Count diagonal lines of length >= 2
        diagonal_points = 0
        total_recurrent = np.sum(recurrence_matrix)
        
        for i in range(recurrence_matrix.shape[0] - 1):
            for j in range(recurrence_matrix.shape[1] - 1):
                if recurrence_matrix[i, j] and recurrence_matrix[i+1, j+1]:
                    diagonal_points += 1
        
        return diagonal_points / max(total_recurrent, 1)
    
    def _compute_avg_diagonal_length(self, recurrence_matrix: np.ndarray) -> float:
        """Compute average diagonal line length."""
        diagonal_lengths = []
        
        # Find diagonal lines
        for i in range(recurrence_matrix.shape[0]):
            for j in range(recurrence_matrix.shape[1]):
                if recurrence_matrix[i, j]:
                    # Check if this is the start of a diagonal line
                    length = 1
                    k = 1
                    while (i + k < recurrence_matrix.shape[0] and 
                           j + k < recurrence_matrix.shape[1] and 
                           recurrence_matrix[i + k, j + k]):
                        length += 1
                        k += 1
                    
                    if length >= 2:
                        diagonal_lengths.append(length)
        
        return np.mean(diagonal_lengths) if diagonal_lengths else 0
    
    def _compute_laminarity(self, recurrence_matrix: np.ndarray) -> float:
        """Compute laminarity measure."""
        # Count vertical lines of length >= 2
        vertical_points = 0
        total_recurrent = np.sum(recurrence_matrix)
        
        for j in range(recurrence_matrix.shape[1]):
            for i in range(recurrence_matrix.shape[0] - 1):
                if recurrence_matrix[i, j] and recurrence_matrix[i+1, j]:
                    vertical_points += 1
        
        return vertical_points / max(total_recurrent, 1)
    
    def _compute_entropy(self, recurrence_matrix: np.ndarray) -> float:
        """Compute entropy of diagonal line lengths."""
        diagonal_lengths = []
        
        # Find diagonal lines
        for i in range(recurrence_matrix.shape[0]):
            for j in range(recurrence_matrix.shape[1]):
                if recurrence_matrix[i, j]:
                    length = 1
                    k = 1
                    while (i + k < recurrence_matrix.shape[0] and 
                           j + k < recurrence_matrix.shape[1] and 
                           recurrence_matrix[i + k, j + k]):
                        length += 1
                        k += 1
                    
                    if length >= 2:
                        diagonal_lengths.append(length)
        
        if not diagonal_lengths:
            return 0
        
        # Compute probability distribution
        unique_lengths, counts = np.unique(diagonal_lengths, return_counts=True)
        probabilities = counts / len(diagonal_lengths)
        
        # Compute entropy
        entropy = -np.sum(probabilities * np.log(probabilities + 1e-10))
        
        return entropy
        
    def compute_recurrence_matrix(self, trajectory: np.ndarray) -> np.ndarray:
        """
        Compute recurrence matrix for trajectory.
        
        Parameters
        ----------
        trajectory : np.ndarray
            Input trajectory
            
        Returns
        -------
        recurrence_matrix : np.ndarray
            Binary recurrence matrix
        """
        from scipy.spatial.distance import pdist, squareform
        
        n_points = len(trajectory)
        
        # Compute pairwise distances
        distances = squareform(pdist(trajectory))
        
        # Create recurrence matrix
        recurrence_matrix = distances < self.threshold
        
        return recurrence_matrix.astype(int)


class LyapunovAnalysis(BaseAttractorDetector):
    """Detect and characterize attractors using Lyapunov exponents."""
    
    def __init__(
        self,
        threshold: float = 0.1,
        n_neighbors: int = 10,
        evolution_time: int = 10
    ):
        """
        Initialize Lyapunov-based detector.
        
        Parameters
        ----------
        threshold : float
            Detection threshold
        n_neighbors : int
            Number of neighbors for local analysis
        evolution_time : int
            Time steps for evolution
        """
        super().__init__(threshold)
        self.n_neighbors = n_neighbors
        self.evolution_time = evolution_time
        
    def detect(self, trajectory: np.ndarray) -> List[Attractor]:
        """Detect attractors using Lyapunov analysis."""
        # Embed trajectory if needed
        if trajectory.ndim == 1:
            embedded_trajectory = embed_trajectory(trajectory, 3, 1)
        else:
            embedded_trajectory = trajectory
        
        # Compute local Lyapunov exponents
        lyapunov_exponents = self._compute_local_lyapunov_exponents(embedded_trajectory)
        
        # Find regions with negative Lyapunov exponents (attracting regions)
        attracting_regions = lyapunov_exponents < -self.threshold
        
        # Cluster attracting regions
        attractors = self._cluster_attracting_regions(embedded_trajectory, attracting_regions, lyapunov_exponents)
        
        return attractors
    
    def _compute_local_lyapunov_exponents(self, trajectory: np.ndarray) -> np.ndarray:
        """Compute local Lyapunov exponents."""
        from scipy.spatial import cKDTree
        
        n_points = len(trajectory)
        lyapunov_exponents = np.zeros(n_points)
        
        # Build KD-tree for efficient neighbor search
        tree = cKDTree(trajectory)
        
        for i in range(n_points - self.evolution_time):
            # Find nearest neighbors
            distances, indices = tree.query(trajectory[i], k=self.n_neighbors + 1)
            
            # Exclude self
            neighbor_indices = indices[1:]
            
            # Compute divergence rates
            divergence_rates = []
            
            for j in neighbor_indices:
                if j + self.evolution_time < n_points:
                    # Initial separation
                    initial_sep = np.linalg.norm(trajectory[i] - trajectory[j])
                    
                    # Final separation after evolution
                    final_sep = np.linalg.norm(trajectory[i + self.evolution_time] - trajectory[j + self.evolution_time])
                    
                    if initial_sep > 0 and final_sep > 0:
                        divergence_rate = np.log(final_sep / initial_sep) / self.evolution_time
                        divergence_rates.append(divergence_rate)
            
            # Average divergence rate
            if divergence_rates:
                lyapunov_exponents[i] = np.mean(divergence_rates)
        
        return lyapunov_exponents
    
    def _cluster_attracting_regions(self, trajectory: np.ndarray, attracting_mask: np.ndarray, lyapunov_exponents: np.ndarray) -> List[Attractor]:
        """Cluster attracting regions into attractors."""
        from sklearn.cluster import DBSCAN
        
        # Get attracting points
        attracting_indices = np.where(attracting_mask)[0]
        if len(attracting_indices) == 0:
            return []
        
        attracting_points = trajectory[attracting_indices]
        
        # Cluster attracting points
        clustering = DBSCAN(eps=self.threshold * 2, min_samples=5)
        labels = clustering.fit_predict(attracting_points)
        
        attractors = []
        
        for label in np.unique(labels):
            if label == -1:  # Noise
                continue
            
            cluster_mask = labels == label
            cluster_points = attracting_points[cluster_mask]
            cluster_indices = attracting_indices[cluster_mask]
            
            if len(cluster_points) < 5:
                continue
            
            # Compute attractor properties
            center = np.mean(cluster_points, axis=0)
            basin_size = len(cluster_points) / len(trajectory)
            
            # Average Lyapunov exponent for this attractor
            avg_lyapunov = np.mean(lyapunov_exponents[cluster_indices])
            
            # Classify attractor type based on Lyapunov exponent
            if avg_lyapunov < -0.1:
                attractor_type = AttractorType.FIXED_POINT
            elif avg_lyapunov < 0:
                attractor_type = AttractorType.LIMIT_CYCLE
            else:
                attractor_type = AttractorType.STRANGE
            
            attractor = Attractor(
                type=attractor_type,
                center=center,
                basin_size=basin_size,
                lyapunov_exponents=np.array([avg_lyapunov]),
                trajectory_indices=cluster_indices.tolist()
            )
            
            attractors.append(attractor)
        
        return attractors
        
    def characterize(self, attractor: Attractor, trajectory: np.ndarray) -> Dict[str, Any]:
        """Compute basic Lyapunov characterization for an attractor (MVP)."""
        if attractor.trajectory_indices is None or len(attractor.trajectory_indices) == 0:
            return {}
        indices = np.asarray(attractor.trajectory_indices)
        local_exponents = self._compute_local_lyapunov_exponents(trajectory)
        valid = indices[indices < len(local_exponents)]
        if len(valid) == 0:
            return {}
        vals = local_exponents[valid]
        mean_lyap = float(np.mean(vals))
        stability = 'attracting' if mean_lyap < -self.threshold else 'neutral' if mean_lyap < self.threshold else 'repelling'
        return {'mean_lyapunov': mean_lyap, 'stability': stability}
        
    def compute_lyapunov_spectrum(
        self,
        trajectory: np.ndarray,
        dt: float = 1.0
    ) -> np.ndarray:
        """
        Compute Lyapunov exponent spectrum.
        
        Parameters
        ----------
        trajectory : np.ndarray
            Input trajectory
        dt : float
            Time step
            
        Returns
        -------
        spectrum : np.ndarray
            Lyapunov exponents
        """
        # TODO: Implement Lyapunov spectrum computation
        raise NotImplementedError("Lyapunov spectrum to be implemented")


class ClusteringDetector(BaseAttractorDetector):
    """Detect attractors using density-based clustering."""
    
    def __init__(
        self,
        threshold: float = 0.1,
        min_samples: int = 10,
        clustering_method: str = "dbscan"
    ):
        """
        Initialize clustering-based detector.
        
        Parameters
        ----------
        threshold : float
            Detection threshold
        min_samples : int
            Minimum samples per cluster
        clustering_method : str
            Clustering algorithm to use
        """
        super().__init__(threshold)
        self.min_samples = min_samples
        self.clustering_method = clustering_method
        
    def detect(self, trajectory: np.ndarray) -> List[Attractor]:
        """Detect attractors using clustering."""
        from sklearn.cluster import DBSCAN
        
        # Embed trajectory if needed
        if trajectory.ndim == 1:
            embedded_trajectory = embed_trajectory(trajectory, 3, 1)
        else:
            embedded_trajectory = trajectory
        
        # Apply clustering
        if self.clustering_method == "dbscan":
            clustering = DBSCAN(eps=self.threshold, min_samples=self.min_samples)
            labels = clustering.fit_predict(embedded_trajectory)
        else:
            from sklearn.cluster import KMeans
            # Estimate number of clusters
            n_clusters = max(2, min(10, len(embedded_trajectory) // 50))
            clustering = KMeans(n_clusters=n_clusters, random_state=42)
            labels = clustering.fit_predict(embedded_trajectory)
        
        # Convert clusters to attractors
        attractors = []
        
        for label in np.unique(labels):
            if label == -1:  # Noise in DBSCAN
                continue
            
            cluster_mask = labels == label
            cluster_points = embedded_trajectory[cluster_mask]
            cluster_indices = np.where(cluster_mask)[0]
            
            if len(cluster_points) < self.min_samples:
                continue
            
            # Compute attractor properties
            center = np.mean(cluster_points, axis=0)
            basin_size = len(cluster_points) / len(embedded_trajectory)
            
            # Estimate attractor dimension using correlation dimension
            dimension = self._estimate_correlation_dimension(cluster_points)
            
            # Classify attractor type based on point distribution
            attractor_type = self._classify_attractor_by_clustering(cluster_points)
            
            attractor = Attractor(
                type=attractor_type,
                center=center,
                basin_size=basin_size,
                dimension=dimension,
                trajectory_indices=cluster_indices.tolist()
            )
            
            attractors.append(attractor)
        
        return attractors
    
    def _estimate_correlation_dimension(self, points: np.ndarray) -> float:
        """Estimate correlation dimension of point set."""
        from scipy.spatial.distance import pdist
        
        # Compute pairwise distances
        distances = pdist(points)
        
        # Use a range of distance scales
        r_min = np.percentile(distances, 1)
        r_max = np.percentile(distances, 50)
        
        if r_min >= r_max:
            return 0.0
        
        # Compute correlation sum for different scales
        n_scales = 10
        scales = np.logspace(np.log10(r_min), np.log10(r_max), n_scales)
        correlation_sums = []
        
        for scale in scales:
            # Count pairs within scale
            count = np.sum(distances < scale)
            correlation_sum = count / len(distances)
            correlation_sums.append(correlation_sum)
        
        # Estimate dimension from slope
        log_scales = np.log(scales)
        log_sums = np.log(np.array(correlation_sums) + 1e-10)
        
        # Linear regression
        if len(log_scales) > 1:
            slope = np.polyfit(log_scales, log_sums, 1)[0]
            return max(0, slope)
        else:
            return 0.0
    
    def _classify_attractor_by_clustering(self, points: np.ndarray) -> AttractorType:
        """Classify attractor type based on point distribution."""
        # Compute statistics of point distribution
        centroid = np.mean(points, axis=0)
        distances_to_centroid = np.linalg.norm(points - centroid, axis=1)
        
        # Coefficient of variation
        cv = np.std(distances_to_centroid) / (np.mean(distances_to_centroid) + 1e-10)
        
        # Classify based on coefficient of variation
        if cv < 0.1:
            return AttractorType.FIXED_POINT
        elif cv < 0.5:
            return AttractorType.LIMIT_CYCLE
        else:
            return AttractorType.STRANGE
        
    def characterize(self, attractor: Attractor, trajectory: np.ndarray) -> Dict[str, Any]:
        """Characterize attractor geometry using clustering (MVP)."""
        if attractor.trajectory_indices is None or len(attractor.trajectory_indices) == 0:
            return {}
        pts = trajectory[np.asarray(attractor.trajectory_indices)]
        centroid = np.mean(pts, axis=0)
        dists = np.linalg.norm(pts - centroid, axis=1)
        return {
            'radius_mean': float(np.mean(dists)),
            'radius_std': float(np.std(dists)),
            'num_points': int(len(pts)),
        }


class AttractorDetector:
    """Main attractor detection class combining multiple methods."""
    
    def __init__(
        self,
        method: str = "recurrence",
        threshold: float = 0.1,
        **kwargs
    ):
        """
        Initialize attractor detector.
        
        Parameters
        ----------
        method : str
            Detection method ('recurrence', 'lyapunov', 'clustering')
        threshold : float
            Detection threshold
        **kwargs
            Method-specific parameters
        """
        self.method = method
        self.threshold = threshold
        self.method_params = kwargs
        self._detector = self._initialize_detector()
        
    def _initialize_detector(self) -> BaseAttractorDetector:
        """Initialize the appropriate detector backend."""
        if self.method == "recurrence":
            return RecurrenceAnalysis(self.threshold, **self.method_params)
        elif self.method == "lyapunov":
            return LyapunovAnalysis(self.threshold, **self.method_params)
        elif self.method == "clustering":
            return ClusteringDetector(self.threshold, **self.method_params)
        else:
            raise ValueError(f"Unknown detection method: {self.method}")
            
    def detect(self, trajectory: np.ndarray) -> List[Attractor]:
        """Detect attractors in trajectory."""
        return self._detector.detect(trajectory)
        
    def characterize(self, attractor: Attractor, trajectory: np.ndarray) -> Dict[str, Any]:
        """Characterize attractor properties."""
        return self._detector.characterize(attractor, trajectory)
        
    def classify_attractor(self, attractor: Attractor) -> AttractorType:
        """Classify attractor type based on simple heuristics (MVP)."""
        if attractor.dimension is not None:
            if attractor.dimension < 0.2:
                return AttractorType.FIXED_POINT
            if attractor.dimension < 1.2:
                return AttractorType.LIMIT_CYCLE
            return AttractorType.STRANGE
        if attractor.lyapunov_exponents is not None and len(attractor.lyapunov_exponents) > 0:
            lyap = float(np.mean(attractor.lyapunov_exponents))
            if lyap < -0.1:
                return AttractorType.FIXED_POINT
            if lyap < 0.05:
                return AttractorType.LIMIT_CYCLE
            return AttractorType.STRANGE
        if attractor.basin_size < 0.02:
            return AttractorType.FIXED_POINT
        return AttractorType.LIMIT_CYCLE


# Utility functions
def embed_trajectory(
    time_series: TimeSeries,
    embedding_dimension: int,
    time_delay: int
) -> np.ndarray:
    """
    Create delay embedding of time series.
    
    Parameters
    ----------
    time_series : np.ndarray
        Input time series, shape (n_timesteps,) or (n_timesteps, n_features)
    embedding_dimension : int
        Embedding dimension
    time_delay : int
        Time delay
        
    Returns
    -------
    embedded : np.ndarray
        Embedded trajectory
    """
    if time_series.ndim == 1:
        time_series = time_series.reshape(-1, 1)
        
    n_points = len(time_series) - (embedding_dimension - 1) * time_delay
    if n_points <= 0:
        raise ValueError("Time series too short for embedding")
    
    embedded = np.zeros((n_points, embedding_dimension * time_series.shape[1]))
    
    for i in range(embedding_dimension):
        start_idx = i * time_delay
        end_idx = start_idx + n_points
        dim_slice = slice(i * time_series.shape[1], (i + 1) * time_series.shape[1])
        embedded[:, dim_slice] = time_series[start_idx:end_idx]
        
    return embedded


def estimate_embedding_parameters(
    time_series: TimeSeries,
    max_dimension: int = 10,
    max_delay: int = 100
) -> Tuple[int, int]:
    """
    Estimate optimal embedding dimension and time delay.
    
    Parameters
    ----------
    time_series : np.ndarray
        Input time series
    max_dimension : int
        Maximum dimension to test
    max_delay : int
        Maximum delay to test
        
    Returns
    -------
    embedding_dimension : int
        Optimal embedding dimension
    time_delay : int
        Optimal time delay
    """
    if time_series.ndim > 1:
        time_series = time_series.flatten()
    
    # Estimate time delay using first minimum of mutual information
    time_delay = _estimate_time_delay_mutual_info(time_series, max_delay)
    
    # Estimate embedding dimension using false nearest neighbors
    embedding_dimension = _estimate_dimension_fnn(time_series, time_delay, max_dimension)
    
    return embedding_dimension, time_delay


def _estimate_time_delay_mutual_info(time_series: np.ndarray, max_delay: int) -> int:
    """Estimate time delay using mutual information."""
    # Simple approximation using correlation
    delays = range(1, min(max_delay, len(time_series) // 10))
    correlations = []
    
    for delay in delays:
        x = time_series[:-delay]
        y = time_series[delay:]
        
        # Compute correlation
        correlation = np.corrcoef(x, y)[0, 1]
        correlations.append(abs(correlation))
    
    # Find first local minimum
    for i in range(1, len(correlations) - 1):
        if correlations[i] < correlations[i-1] and correlations[i] < correlations[i+1]:
            return delays[i]
    
    # If no minimum found, return 1
    return 1


def _estimate_dimension_fnn(time_series: np.ndarray, time_delay: int, max_dimension: int) -> int:
    """Estimate embedding dimension using false nearest neighbors."""
    from scipy.spatial import cKDTree
    
    fnn_percentages = []
    
    for dim in range(1, max_dimension + 1):
        # Create embedding
        try:
            embedded = embed_trajectory(time_series, dim, time_delay)
        except ValueError:
            break
        
        if len(embedded) < 10:
            break
        
        # Build KD-tree
        tree = cKDTree(embedded)
        
        # Check for false nearest neighbors
        false_neighbors = 0
        total_neighbors = 0
        
        for i in range(len(embedded) - time_delay):
            # Find nearest neighbor
            distances, indices = tree.query(embedded[i], k=2)
            
            if len(indices) < 2:
                continue
            
            neighbor_idx = indices[1]  # Skip self
            
            # Check if neighbor is false in higher dimension
            if neighbor_idx < len(embedded) - time_delay:
                # Distance in current dimension
                current_dist = distances[1]
                
                # Distance in next dimension (approximate)
                next_point_i = time_series[i + dim * time_delay] if i + dim * time_delay < len(time_series) else time_series[i]
                next_point_j = time_series[neighbor_idx + dim * time_delay] if neighbor_idx + dim * time_delay < len(time_series) else time_series[neighbor_idx]
                
                next_dist = abs(next_point_i - next_point_j)
                
                # Check if neighbor becomes false
                if current_dist > 0 and next_dist / current_dist > 2.0:
                    false_neighbors += 1
                
                total_neighbors += 1
        
        fnn_percentage = false_neighbors / max(total_neighbors, 1)
        fnn_percentages.append(fnn_percentage)
        
        # Stop if percentage is low enough
        if fnn_percentage < 0.1:
            return dim
    
    # Return dimension with minimum false neighbors
    if fnn_percentages:
        return np.argmin(fnn_percentages) + 1
    else:
        return 3  # Default


def compute_correlation_dimension(
    trajectory: np.ndarray,
    r_min: float = 0.01,
    r_max: float = 1.0,
    n_points: int = 20
) -> float:
    """
    Compute correlation dimension of attractor.
    
    Parameters
    ----------
    trajectory : np.ndarray
        Phase space trajectory
    r_min, r_max : float
        Range of distances to consider
    n_points : int
        Number of distance values
        
    Returns
    -------
    dimension : float
        Correlation dimension
    """
    from scipy.spatial.distance import pdist
    
    # Compute pairwise distances
    distances = pdist(trajectory)
    
    # Automatically adjust range if needed
    if r_min == 0.01 and r_max == 1.0:
        r_min = np.percentile(distances, 1)
        r_max = np.percentile(distances, 50)
    
    # Create distance scales
    scales = np.logspace(np.log10(r_min), np.log10(r_max), n_points)
    
    # Compute correlation sums
    correlation_sums = []
    
    for scale in scales:
        # Count pairs within scale
        count = np.sum(distances < scale)
        correlation_sum = count / len(distances)
        correlation_sums.append(correlation_sum)
    
    # Estimate dimension from slope
    log_scales = np.log(scales)
    log_sums = np.log(np.array(correlation_sums) + 1e-10)
    
    # Linear regression on the linear part
    if len(log_scales) > 2:
        # Find linear region (middle part)
        start_idx = len(log_scales) // 4
        end_idx = 3 * len(log_scales) // 4
        
        slope = np.polyfit(log_scales[start_idx:end_idx], log_sums[start_idx:end_idx], 1)[0]
        return max(0, slope)
    else:
        return 0.0


def compute_basin_of_attraction(
    attractor: Attractor,
    trajectory: np.ndarray,
    grid_resolution: int = 50
) -> np.ndarray:
    """
    Estimate basin of attraction.
    
    Parameters
    ----------
    attractor : Attractor
        Target attractor
    trajectory : np.ndarray
        Full trajectory data
    grid_resolution : int
        Resolution for basin estimation
        
    Returns
    -------
    basin : np.ndarray
        Basin of attraction indicator
    """
    # TODO: Implement basin computation
    raise NotImplementedError("Basin of attraction to be implemented")