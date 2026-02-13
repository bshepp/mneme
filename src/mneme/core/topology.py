"""Topological Data Analysis methods for field analysis."""

from typing import List, Optional, Dict, Any, Tuple, Union
import numpy as np
from abc import ABC, abstractmethod

from ..types import (
    PersistenceDiagram, FiltrationMethod, TopologyResult,
    Field, FieldData
)

# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------

#: Default maximum number of points returned by field_to_point_cloud.
DEFAULT_MAX_POINTS: int = 2000

#: Entropy stabiliser added inside log to avoid log(0).
_ENTROPY_EPS: float = 1e-12


class BaseTopologyAnalyzer(ABC):
    """Abstract base class for topology analysis methods."""
    
    def __init__(self, max_dimension: int = 2):
        """
        Initialize topology analyzer.
        
        Parameters
        ----------
        max_dimension : int
            Maximum homological dimension to compute
        """
        self.max_dimension = max_dimension
        
    @abstractmethod
    def compute_persistence(self, field: np.ndarray) -> List[PersistenceDiagram]:
        """
        Compute persistence diagrams for field.
        
        Parameters
        ----------
        field : np.ndarray
            Input field data
            
        Returns
        -------
        diagrams : List[PersistenceDiagram]
            Persistence diagrams for each dimension
        """
        raise NotImplementedError
        
    @abstractmethod
    def extract_features(self, diagrams: List[PersistenceDiagram]) -> np.ndarray:
        """
        Extract topological features from persistence diagrams.
        
        Parameters
        ----------
        diagrams : List[PersistenceDiagram]
            Persistence diagrams
            
        Returns
        -------
        features : np.ndarray
            Feature vector
        """
        raise NotImplementedError


class PersistentHomology(BaseTopologyAnalyzer):
    """Compute persistent homology of fields."""
    
    def __init__(
        self,
        max_dimension: int = 2,
        filtration: Union[str, FiltrationMethod] = FiltrationMethod.SUBLEVEL,
        persistence_threshold: float = 0.05,
        compute_cycles: bool = True
    ):
        """
        Initialize persistent homology analyzer.
        
        Parameters
        ----------
        max_dimension : int
            Maximum homological dimension
        filtration : str or FiltrationMethod
            Type of filtration to use
        persistence_threshold : float
            Minimum persistence to consider significant
        compute_cycles : bool
            Whether to compute representative cycles
        """
        super().__init__(max_dimension)
        self.filtration = FiltrationMethod(filtration) if isinstance(filtration, str) else filtration
        self.persistence_threshold = persistence_threshold
        self.compute_cycles = compute_cycles
        self._cycles = None
        
    def compute_persistence(self, field: np.ndarray) -> List[PersistenceDiagram]:
        """Compute persistence diagrams using GUDHI."""
        try:
            import gudhi
        except ImportError:
            # Fallback implementation without GUDHI
            return self._compute_persistence_simple(field)
        
        if field.ndim != 2:
            raise ValueError("Persistence computation only supports 2D fields")
        
        # Create cubical complex
        if self.filtration == FiltrationMethod.SUBLEVEL:
            # For sublevel filtration, negate the field
            filtration_values = -field.flatten()
        else:
            filtration_values = field.flatten()
        
        # Create cubical complex
        cubical_complex = gudhi.CubicalComplex(
            dimensions=field.shape,
            top_dimensional_cells=filtration_values
        )
        
        # Compute persistence
        cubical_complex.compute_persistence()
        
        # Extract diagrams by dimension
        diagrams = []
        for dim in range(self.max_dimension + 1):
            persistence_pairs = cubical_complex.persistence_intervals_in_dimension(dim)
            
            if len(persistence_pairs) > 0:
                # Filter by persistence threshold
                if self.persistence_threshold > 0:
                    persistence_values = persistence_pairs[:, 1] - persistence_pairs[:, 0]
                    mask = persistence_values >= self.persistence_threshold
                    persistence_pairs = persistence_pairs[mask]
                
                diagram = PersistenceDiagram(
                    points=persistence_pairs,
                    dimension=dim,
                    threshold=self.persistence_threshold
                )
                diagrams.append(diagram)
            else:
                # Empty diagram
                diagram = PersistenceDiagram(
                    points=np.empty((0, 2)),
                    dimension=dim,
                    threshold=self.persistence_threshold
                )
                diagrams.append(diagram)
        
        # Store cycles if requested
        if self.compute_cycles:
            self._cycles = self._extract_cycles(cubical_complex, diagrams)
        
        return diagrams
    
    def _compute_persistence_simple(self, field: np.ndarray) -> List[PersistenceDiagram]:
        """Simple persistence computation without GUDHI."""
        # Basic implementation using connected components
        from scipy import ndimage
        
        if field.ndim != 2:
            raise ValueError("Simple persistence only supports 2D fields")
        
        # Match GUDHI behaviour: for sublevel filtration, negate the field
        # so that we track superlevel sets of the original (i.e. detect peaks).
        if self.filtration == FiltrationMethod.SUBLEVEL:
            work_field = -field
        else:
            work_field = field
        
        # Create binary images at different thresholds
        min_val, max_val = work_field.min(), work_field.max()
        n_levels = 50
        thresholds = np.linspace(min_val, max_val, n_levels)
        
        # Track connected components
        components_history = []
        
        for threshold in thresholds:
            binary = work_field <= threshold
            
            # Find connected components
            labeled, num_features = ndimage.label(binary)
            components_history.append((threshold, num_features, labeled))
        
        # Extract 0-dimensional persistence (connected components)
        birth_death_pairs = []
        
        # Simple birth-death tracking
        for i in range(len(components_history) - 1):
            curr_threshold, curr_num, curr_labeled = components_history[i]
            next_threshold, next_num, next_labeled = components_history[i + 1]
            
            if next_num > curr_num:
                # New components born
                for _ in range(next_num - curr_num):
                    birth_death_pairs.append([curr_threshold, np.inf])
            elif next_num < curr_num:
                # Components died
                for _ in range(curr_num - next_num):
                    if birth_death_pairs:
                        # Find the youngest component and kill it
                        for j in range(len(birth_death_pairs)):
                            if birth_death_pairs[j][1] == np.inf:
                                birth_death_pairs[j][1] = curr_threshold
                                break
        
        # Convert to numpy array
        if birth_death_pairs:
            points = np.array(birth_death_pairs)
            # Filter infinite persistence
            finite_mask = np.isfinite(points[:, 1])
            points = points[finite_mask]
        else:
            points = np.empty((0, 2))
        
        # Create diagram
        diagram = PersistenceDiagram(
            points=points,
            dimension=0,
            threshold=self.persistence_threshold
        )
        
        return [diagram]
    
    def _extract_cycles(self, cubical_complex, diagrams):
        """Extract representative cycles."""
        # This is a simplified implementation
        # Full implementation would extract actual cycle representatives
        cycles = []
        for diagram in diagrams:
            if diagram.dimension == 1:  # Only for 1-cycles
                # Placeholder for actual cycle extraction
                cycles.append(np.array([]))
        return cycles
        
    def extract_features(self, diagrams: List[PersistenceDiagram]) -> np.ndarray:
        """
        Extract topological features from persistence diagrams.

        Features per dimension: count, total, max, mean, entropy, std.
        Guards against NaN/Inf and division by zero.
        """
        features: list[float] = []

        for diagram in diagrams:
            if len(diagram.points) == 0:
                dim_features = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            else:
                persistence_values = diagram.persistence
                # Keep only finite values
                finite_vals = persistence_values[np.isfinite(persistence_values)]
                if finite_vals.size == 0:
                    dim_features = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                else:
                    n_features = float(len(finite_vals))
                    total_persistence = float(np.sum(finite_vals))
                    max_persistence = float(np.max(finite_vals))
                    mean_persistence = float(np.mean(finite_vals))
                    with np.errstate(divide='ignore', invalid='ignore'):
                        if total_persistence > 0:
                            p_norm = finite_vals / total_persistence
                            entropy = float(-np.sum(p_norm * np.log(p_norm + _ENTROPY_EPS)))
                        else:
                            entropy = 0.0
                    std_persistence = float(np.std(finite_vals)) if finite_vals.size > 1 else 0.0
                    dim_features = [
                        n_features,
                        total_persistence,
                        max_persistence,
                        mean_persistence,
                        entropy,
                        std_persistence,
                    ]

            features.extend(dim_features)

        return np.asarray(features, dtype=float)
        
    def get_cycles(self) -> Optional[List[np.ndarray]]:
        """
        Get representative cycles for persistent features.
        
        Returns
        -------
        cycles : List[np.ndarray] or None
            Representative cycles if computed
        """
        return self._cycles
        
    def compute_persistence_image(
        self,
        diagram: PersistenceDiagram,
        resolution: Tuple[int, int] = (50, 50),
        sigma: float = 0.1
    ) -> np.ndarray:
        """
        Compute persistence image from diagram.
        
        Parameters
        ----------
        diagram : PersistenceDiagram
            Input persistence diagram
        resolution : Tuple[int, int]
            Image resolution
        sigma : float
            Gaussian kernel width
            
        Returns
        -------
        image : np.ndarray
            Persistence image
        """
        if len(diagram.points) == 0:
            return np.zeros(resolution)
        
        # Transform to birth-persistence coordinates
        birth = diagram.points[:, 0]
        death = diagram.points[:, 1]
        persistence = death - birth
        
        # Create grid
        birth_range = (birth.min(), birth.max()) if len(birth) > 0 else (0, 1)
        pers_range = (0, persistence.max()) if len(persistence) > 0 else (0, 1)
        
        # Add small buffer
        birth_range = (birth_range[0] - 0.1, birth_range[1] + 0.1)
        pers_range = (pers_range[0], pers_range[1] + 0.1)
        
        # Create coordinate grids
        birth_coords = np.linspace(birth_range[0], birth_range[1], resolution[0])
        pers_coords = np.linspace(pers_range[0], pers_range[1], resolution[1])
        
        B, P = np.meshgrid(birth_coords, pers_coords, indexing='ij')
        
        # Initialize image
        image = np.zeros(resolution)
        
        # Add Gaussian for each point
        for i in range(len(birth)):
            b_i = birth[i]
            p_i = persistence[i]
            
            # Weight by persistence
            weight = p_i
            
            # Gaussian kernel
            gaussian = weight * np.exp(-((B - b_i)**2 + (P - p_i)**2) / (2 * sigma**2))
            image += gaussian
        
        return image
        
    def compute_persistence_landscape(
        self,
        diagram: PersistenceDiagram,
        k: int = 5,
        resolution: int = 100
    ) -> np.ndarray:
        """
        Compute persistence landscape.
        
        Parameters
        ----------
        diagram : PersistenceDiagram
            Input persistence diagram
        k : int
            Number of landscape functions
        resolution : int
            Resolution of landscape functions
            
        Returns
        -------
        landscape : np.ndarray
            Persistence landscape functions
        """
        if len(diagram.points) == 0:
            return np.zeros((k, resolution))
        
        # Get birth and death times
        birth = diagram.points[:, 0]
        death = diagram.points[:, 1]
        
        # Create parameter range
        t_min = birth.min()
        t_max = death.max()
        t_range = np.linspace(t_min, t_max, resolution)
        
        # Initialize landscape functions
        landscape = np.zeros((k, resolution))
        
        # Compute landscape functions
        for i, t in enumerate(t_range):
            # Compute landscape values at t
            values = []
            
            for j in range(len(birth)):
                b = birth[j]
                d = death[j]
                
                if b <= t <= d:
                    # Triangle function
                    value = min(t - b, d - t)
                    values.append(value)
            
            # Sort in descending order
            values.sort(reverse=True)
            
            # Assign to landscape functions
            for j in range(min(k, len(values))):
                landscape[j, i] = values[j]
        
        return landscape


class RipsComplex(BaseTopologyAnalyzer):
    """Vietoris-Rips complex for point cloud data."""
    
    def __init__(
        self,
        max_dimension: int = 2,
        max_edge_length: float = np.inf
    ):
        """
        Initialize Rips complex analyzer.
        
        Parameters
        ----------
        max_dimension : int
            Maximum dimension for complex
        max_edge_length : float
            Maximum edge length in complex
        """
        super().__init__(max_dimension)
        self.max_edge_length = max_edge_length
        
    def compute_persistence(self, point_cloud: np.ndarray) -> List[PersistenceDiagram]:
        """Compute persistence diagrams for a 2D/ND point cloud using GUDHI Rips.

        Falls back to a scipy-based distance-matrix approximation when GUDHI
        is not installed.  The fallback only computes 0-dimensional persistence
        (connected components).

        Parameters
        ----------
        point_cloud : np.ndarray
            Array of shape (n_points, n_dims)
        """
        if point_cloud.ndim != 2:
            raise ValueError("point_cloud must be 2D (n_points, n_dims)")

        try:
            import gudhi
        except ImportError:
            return self._compute_persistence_fallback(point_cloud)

        rips = gudhi.RipsComplex(points=point_cloud, max_edge_length=float(self.max_edge_length))
        st = rips.create_simplex_tree(max_dimension=self.max_dimension)
        st.compute_persistence()

        diagrams: List[PersistenceDiagram] = []
        for dim in range(self.max_dimension + 1):
            pairs = st.persistence_intervals_in_dimension(dim)
            if len(pairs) == 0:
                diagrams.append(PersistenceDiagram(points=np.empty((0, 2)), dimension=dim, threshold=None))
            else:
                points = np.asarray(pairs)
                diagrams.append(PersistenceDiagram(points=points, dimension=dim, threshold=None))
        return diagrams

    def _compute_persistence_fallback(self, point_cloud: np.ndarray) -> List[PersistenceDiagram]:
        """Approximate 0-dim Rips persistence using scipy distance matrix.

        Uses single-linkage clustering to track when connected components
        merge as the distance threshold grows.
        """
        import warnings
        from scipy.cluster.hierarchy import single, fcluster
        from scipy.spatial.distance import pdist

        warnings.warn(
            "gudhi not installed — using scipy fallback for Rips persistence "
            "(0-dimensional only). Install gudhi for full functionality: "
            "pip install gudhi",
            UserWarning,
        )

        dists = pdist(point_cloud)
        Z = single(dists)  # single-linkage dendrogram

        n = point_cloud.shape[0]
        # Each point is born at distance 0.  Merges happen at Z[:, 2].
        # The *last* component to merge has infinite death.
        birth_death = []
        merge_dists = Z[:, 2]
        for d in merge_dists:
            birth_death.append([0.0, d])

        # Add one component with infinite death (the last surviving)
        # but we filter infinites out to stay consistent with cubical fallback
        if birth_death:
            pts = np.array(birth_death)
        else:
            pts = np.empty((0, 2))

        diagrams: List[PersistenceDiagram] = [
            PersistenceDiagram(points=pts, dimension=0, threshold=None)
        ]
        # Pad higher dimensions with empties
        for dim in range(1, self.max_dimension + 1):
            diagrams.append(PersistenceDiagram(points=np.empty((0, 2)), dimension=dim, threshold=None))
        return diagrams
        
    def extract_features(self, diagrams: List[PersistenceDiagram]) -> np.ndarray:
        """Extract topological features from Rips persistence (mirrors cubical)."""
        features: list[float] = []
        for diagram in diagrams:
            if len(diagram.points) == 0:
                dim_features = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            else:
                persistence_values = diagram.persistence
                finite_vals = persistence_values[np.isfinite(persistence_values)]
                if finite_vals.size == 0:
                    dim_features = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                else:
                    n_features = float(len(finite_vals))
                    total_persistence = float(np.sum(finite_vals))
                    max_persistence = float(np.max(finite_vals))
                    mean_persistence = float(np.mean(finite_vals))
                    with np.errstate(divide='ignore', invalid='ignore'):
                        if total_persistence > 0:
                            p_norm = finite_vals / total_persistence
                            entropy = float(-np.sum(p_norm * np.log(p_norm + _ENTROPY_EPS)))
                        else:
                            entropy = 0.0
                    std_persistence = float(np.std(finite_vals)) if finite_vals.size > 1 else 0.0
                    dim_features = [n_features, total_persistence, max_persistence, mean_persistence, entropy, std_persistence]
            features.extend(dim_features)
        return np.asarray(features, dtype=float)


class AlphaComplex(BaseTopologyAnalyzer):
    """Alpha complex for point cloud data."""
    
    def __init__(self, max_dimension: int = 2):
        """
        Initialize Alpha complex analyzer.
        
        Parameters
        ----------
        max_dimension : int
            Maximum dimension for complex
        """
        super().__init__(max_dimension)
        
    def compute_persistence(self, point_cloud: np.ndarray) -> List[PersistenceDiagram]:
        """Compute persistence using GUDHI Alpha complex.

        Falls back to a Delaunay-based scipy approximation when GUDHI is not
        installed (0-dimensional persistence only).
        """
        if point_cloud.ndim != 2 or point_cloud.shape[1] < 2:
            raise ValueError("point_cloud must be (n_points, n_dims>=2)")

        try:
            import gudhi
        except ImportError:
            return self._compute_persistence_fallback(point_cloud)

        alpha = gudhi.AlphaComplex(points=point_cloud)
        st = alpha.create_simplex_tree()
        st.compute_persistence()

        diagrams: List[PersistenceDiagram] = []
        for dim in range(self.max_dimension + 1):
            pairs = st.persistence_intervals_in_dimension(dim)
            if len(pairs) == 0:
                diagrams.append(PersistenceDiagram(points=np.empty((0, 2)), dimension=dim, threshold=None))
            else:
                points = np.asarray(pairs)
                diagrams.append(PersistenceDiagram(points=points, dimension=dim, threshold=None))
        return diagrams

    def _compute_persistence_fallback(self, point_cloud: np.ndarray) -> List[PersistenceDiagram]:
        """Approximate 0-dim Alpha persistence using Delaunay + single-linkage.

        This is the same strategy as the Rips fallback (distance-based
        single-linkage) since full Alpha filtration requires GUDHI.
        """
        import warnings
        from scipy.cluster.hierarchy import single
        from scipy.spatial.distance import pdist

        warnings.warn(
            "gudhi not installed — using scipy fallback for Alpha persistence "
            "(0-dimensional only). Install gudhi for full functionality: "
            "pip install gudhi",
            UserWarning,
        )

        dists = pdist(point_cloud)
        Z = single(dists)

        birth_death = []
        for d in Z[:, 2]:
            birth_death.append([0.0, d])

        pts = np.array(birth_death) if birth_death else np.empty((0, 2))

        diagrams: List[PersistenceDiagram] = [
            PersistenceDiagram(points=pts, dimension=0, threshold=None)
        ]
        for dim in range(1, self.max_dimension + 1):
            diagrams.append(PersistenceDiagram(points=np.empty((0, 2)), dimension=dim, threshold=None))
        return diagrams
        
    def extract_features(self, diagrams: List[PersistenceDiagram]) -> np.ndarray:
        """Extract topological features from Alpha persistence (mirrors cubical)."""
        features: list[float] = []
        for diagram in diagrams:
            if len(diagram.points) == 0:
                dim_features = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            else:
                persistence_values = diagram.persistence
                finite_vals = persistence_values[np.isfinite(persistence_values)]
                if finite_vals.size == 0:
                    dim_features = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                else:
                    n_features = float(len(finite_vals))
                    total_persistence = float(np.sum(finite_vals))
                    max_persistence = float(np.max(finite_vals))
                    mean_persistence = float(np.mean(finite_vals))
                    with np.errstate(divide='ignore', invalid='ignore'):
                        if total_persistence > 0:
                            p_norm = finite_vals / total_persistence
                            entropy = float(-np.sum(p_norm * np.log(p_norm + _ENTROPY_EPS)))
                        else:
                            entropy = 0.0
                    std_persistence = float(np.std(finite_vals)) if finite_vals.size > 1 else 0.0
                    dim_features = [n_features, total_persistence, max_persistence, mean_persistence, entropy, std_persistence]
            features.extend(dim_features)
        return np.asarray(features, dtype=float)


# Utility functions
def compute_wasserstein_distance(
    diagram1: PersistenceDiagram,
    diagram2: PersistenceDiagram,
    p: float = 2.0
) -> float:
    """
    Compute Wasserstein distance between persistence diagrams.
    
    Parameters
    ----------
    diagram1, diagram2 : PersistenceDiagram
        Persistence diagrams to compare
    p : float
        Wasserstein parameter (typically 1 or 2)
        
    Returns
    -------
    distance : float
        Wasserstein distance
    """
    try:
        import gudhi
        # Use GUDHI implementation if available
        return gudhi.wasserstein_distance(diagram1.points, diagram2.points, order=p)
    except ImportError:
        # Simple approximation using Hungarian algorithm
        from scipy.optimize import linear_sum_assignment
        
        points1 = diagram1.points
        points2 = diagram2.points
        
        if len(points1) == 0 and len(points2) == 0:
            return 0.0
        
        # Add diagonal points for unmatched points
        diag_points1 = np.array([[(p[0] + p[1]) / 2, (p[0] + p[1]) / 2] for p in points1])
        diag_points2 = np.array([[(p[0] + p[1]) / 2, (p[0] + p[1]) / 2] for p in points2])
        
        # Combine points and diagonal points
        all_points1 = np.vstack([points1, diag_points2]) if len(points2) > 0 else points1
        all_points2 = np.vstack([points2, diag_points1]) if len(points1) > 0 else points2
        
        # Compute cost matrix
        cost_matrix = np.zeros((len(all_points1), len(all_points2)))
        for i, p1 in enumerate(all_points1):
            for j, p2 in enumerate(all_points2):
                cost_matrix[i, j] = np.linalg.norm(p1 - p2, ord=p)
        
        # Solve assignment problem
        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        
        # Return total cost
        return cost_matrix[row_indices, col_indices].sum()


def compute_bottleneck_distance(
    diagram1: PersistenceDiagram,
    diagram2: PersistenceDiagram
) -> float:
    """
    Compute bottleneck distance between persistence diagrams.
    
    Parameters
    ----------
    diagram1, diagram2 : PersistenceDiagram
        Persistence diagrams to compare
        
    Returns
    -------
    distance : float
        Bottleneck distance
    """
    try:
        import gudhi
        # Use GUDHI implementation if available
        return gudhi.bottleneck_distance(diagram1.points, diagram2.points)
    except ImportError:
        # Simple approximation
        from scipy.optimize import linear_sum_assignment
        
        points1 = diagram1.points
        points2 = diagram2.points
        
        if len(points1) == 0 and len(points2) == 0:
            return 0.0
        
        # Add diagonal projections
        diag_points1 = np.array([[(p[0] + p[1]) / 2, (p[0] + p[1]) / 2] for p in points1])
        diag_points2 = np.array([[(p[0] + p[1]) / 2, (p[0] + p[1]) / 2] for p in points2])
        
        # Combine points
        all_points1 = np.vstack([points1, diag_points2]) if len(points2) > 0 else points1
        all_points2 = np.vstack([points2, diag_points1]) if len(points1) > 0 else points2
        
        # Compute cost matrix (L-infinity norm)
        cost_matrix = np.zeros((len(all_points1), len(all_points2)))
        for i, p1 in enumerate(all_points1):
            for j, p2 in enumerate(all_points2):
                cost_matrix[i, j] = np.linalg.norm(p1 - p2, ord=np.inf)
        
        # Solve assignment problem
        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        
        # Return maximum cost (bottleneck)
        return cost_matrix[row_indices, col_indices].max()


def filter_persistence_diagram(
    diagram: PersistenceDiagram,
    threshold: float
) -> PersistenceDiagram:
    """
    Filter persistence diagram by persistence threshold.
    
    Parameters
    ----------
    diagram : PersistenceDiagram
        Input diagram
    threshold : float
        Minimum persistence to keep
        
    Returns
    -------
    filtered : PersistenceDiagram
        Filtered diagram
    """
    persistence = diagram.persistence
    mask = persistence >= threshold
    
    return PersistenceDiagram(
        points=diagram.points[mask],
        dimension=diagram.dimension,
        threshold=threshold
    )


def compute_betti_curve(
    diagram: PersistenceDiagram,
    filtration_values: np.ndarray
) -> np.ndarray:
    """
    Compute Betti curve from persistence diagram.
    
    Parameters
    ----------
    diagram : PersistenceDiagram
        Input persistence diagram
    filtration_values : np.ndarray
        Filtration values at which to compute Betti numbers
        
    Returns
    -------
    betti_curve : np.ndarray
        Betti numbers at each filtration value
    """
    betti_curve = np.zeros_like(filtration_values)
    
    for i, t in enumerate(filtration_values):
        # Count features alive at time t
        alive = (diagram.points[:, 0] <= t) & (diagram.points[:, 1] > t)
        betti_curve[i] = np.sum(alive)
        
    return betti_curve


# Adapters and helpers
def field_to_point_cloud(
    field: np.ndarray,
    method: str = 'peaks',
    percentile: float = 95.0,
    max_points: int = DEFAULT_MAX_POINTS,
    normalize_coords: bool = True,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Convert a 2D field into a 2D point cloud for Rips/Alpha backends.

    Parameters
    ----------
    field : np.ndarray
        2D array (H, W)
    method : str
        'peaks' (local maxima above percentile) or 'threshold' (all pixels above percentile)
    percentile : float
        Value percentile for thresholding
    max_points : int
        Maximum number of points to return (subsampled if exceeded)
    normalize_coords : bool
        If True, scale coordinates to [0,1]^2
    seed : int, optional
        Random seed for reproducible sub-sampling when the number of
        candidate points exceeds *max_points*.

    Returns
    -------
    pc : np.ndarray
        Point cloud of shape (N, 2)
    """
    if field.ndim != 2:
        raise ValueError("field_to_point_cloud expects a 2D array")

    h, w = field.shape
    thr = float(np.percentile(field, percentile))

    if method == 'peaks':
        try:
            from scipy.ndimage import maximum_filter
            size = 3
            neighborhood = maximum_filter(field, size=size)
            mask = (field == neighborhood) & (field >= thr)
        except Exception:
            mask = field >= thr
    else:
        mask = field >= thr

    coords = np.argwhere(mask)  # (y, x)

    if coords.shape[0] == 0:
        # Fallback: take top-k brightest pixels
        flat_idx = np.argsort(field.ravel())[::-1][: max_points]
        ys, xs = np.unravel_index(flat_idx, field.shape)
        coords = np.column_stack([ys, xs])

    # Subsample if too many
    if coords.shape[0] > max_points:
        rng = np.random.RandomState(seed)
        choice = rng.choice(coords.shape[0], size=max_points, replace=False)
        coords = coords[choice]

    # Convert to (x, y) and normalize if requested
    pts = coords[:, ::-1].astype(np.float64)
    if normalize_coords:
        if w > 1:
            pts[:, 0] /= (w - 1)
        if h > 1:
            pts[:, 1] /= (h - 1)
    return pts