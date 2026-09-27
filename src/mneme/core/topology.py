"""Topological Data Analysis methods for field analysis."""

from typing import List, Optional, Dict, Any, Tuple, Union
import warnings
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
        compute_cycles: bool = False
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
            Not implemented. Passing True raises NotImplementedError.
        """
        super().__init__(max_dimension)
        self.filtration = FiltrationMethod(filtration) if isinstance(filtration, str) else filtration
        self.persistence_threshold = persistence_threshold
        if compute_cycles:
            raise NotImplementedError(
                "Representative cycle extraction is not implemented"
            )
        self.compute_cycles = False
        self._cycles = None
        
    def compute_persistence(self, field: np.ndarray) -> List[PersistenceDiagram]:
        """Compute persistence diagrams using GUDHI."""
        field = np.asarray(field, dtype=float)
        if field.ndim != 2:
            raise ValueError("Persistence computation only supports 2D fields")
        if not np.all(np.isfinite(field)):
            raise ValueError(
                "Field contains NaN or infinite values; persistence is undefined"
            )

        try:
            import gudhi
        except ImportError:
            warnings.warn(
                "GUDHI is not installed: computing H0 only with the built-in "
                "union-find fallback. Higher-dimensional diagrams are returned "
                "empty. Install gudhi for H1.",
                RuntimeWarning,
                stacklevel=2,
            )
            return self._compute_persistence_simple(field)

        # GUDHI's cubical complex computes SUBLEVEL persistence of the values
        # it is given. Superlevel persistence of f is sublevel persistence of
        # -f, so superlevel diagrams are expressed in units of -field (which
        # keeps death >= birth).
        values = self._filtration_values(field)

        # GUDHI reads top-dimensional cells with the FIRST axis varying
        # fastest, i.e. Fortran order.
        cubical_complex = gudhi.CubicalComplex(
            dimensions=list(values.shape),
            top_dimensional_cells=values.flatten(order="F"),
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
    
    def _filtration_values(self, field: np.ndarray) -> np.ndarray:
        """Values whose sublevel sets realise the requested filtration."""
        if self.filtration == FiltrationMethod.SUPERLEVEL:
            return -field
        return field

    def _compute_persistence_simple(self, field: np.ndarray) -> List[PersistenceDiagram]:
        """Exact H0 sublevel persistence without GUDHI (union-find, elder rule).

        Pixels are top-dimensional cells, so two pixels that share an edge
        or a corner are connected (8-connectivity). This reproduces the H0
        diagram of GUDHI's cubical complex. H1 and above are NOT computed;
        those diagrams are returned empty.
        """
        values = self._filtration_values(np.asarray(field, dtype=float))
        points = _h0_sublevel_persistence(values)

        finite = np.isfinite(points[:, 1])
        if self.persistence_threshold > 0:
            keep = ~finite | (
                (points[:, 1] - points[:, 0]) >= self.persistence_threshold
            )
            points = points[keep]

        diagrams = [
            PersistenceDiagram(
                points=points, dimension=0, threshold=self.persistence_threshold
            )
        ]
        for dim in range(1, self.max_dimension + 1):
            diagrams.append(
                PersistenceDiagram(
                    points=np.empty((0, 2)),
                    dimension=dim,
                    threshold=self.persistence_threshold,
                )
            )
        return diagrams

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
def _h0_sublevel_persistence(values: np.ndarray) -> np.ndarray:
    """H0 sublevel persistence of a 2-D array of top-cell values.

    Union-find over pixels in increasing value order with 8-connectivity.
    When two components meet, the younger one (larger birth value) dies
    (elder rule). Zero-persistence pairs are dropped. The oldest component
    never dies and is returned with death = inf.

    Returns
    -------
    np.ndarray
        Shape (n, 2) array of (birth, death), sorted by birth.
    """
    rows, cols = values.shape
    flat = values.ravel()
    order = np.argsort(flat, kind="stable")
    parent = np.full(flat.size, -1, dtype=np.int64)
    birth = np.empty(flat.size, dtype=float)

    def find(i: int) -> int:
        root = i
        while parent[root] != root:
            root = parent[root]
        while parent[i] != root:
            parent[i], i = root, parent[i]
        return root

    pairs = []
    offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    for idx in order:
        idx = int(idx)
        value = flat[idx]
        parent[idx] = idx
        birth[idx] = value
        r, c = divmod(idx, cols)
        for dr, dc in offsets:
            rr, cc = r + dr, c + dc
            if not (0 <= rr < rows and 0 <= cc < cols):
                continue
            nb = rr * cols + cc
            if parent[nb] < 0:
                continue
            root_a, root_b = find(idx), find(nb)
            if root_a == root_b:
                continue
            # Elder rule: the component born later dies now.
            if birth[root_a] < birth[root_b]:
                root_a, root_b = root_b, root_a
            if value > birth[root_a]:
                pairs.append((birth[root_a], value))
            parent[root_a] = root_b

    pairs.append((float(flat[order[0]]), np.inf))
    out = np.asarray(pairs, dtype=float)
    return out[np.argsort(out[:, 0], kind="stable")]


def _finite_points(diagram: PersistenceDiagram) -> np.ndarray:
    """Finite (birth, death) pairs of a diagram; essential classes are dropped."""
    points = np.asarray(diagram.points, dtype=float).reshape(-1, 2)
    return points[np.all(np.isfinite(points), axis=1)]


def _matching_costs(points1: np.ndarray, points2: np.ndarray) -> np.ndarray:
    """Cost matrix for diagram matching under the L-infinity ground metric.

    Rows are the points of diagram 1 followed by one diagonal slot per point
    of diagram 2; columns mirror that. A point may be matched to its own
    diagonal projection at cost (death - birth) / 2. Diagonal slots match
    each other at zero cost. Disallowed pairings are inf.
    """
    n1, n2 = len(points1), len(points2)
    cost = np.zeros((n1 + n2, n1 + n2))
    if n1 and n2:
        cost[:n1, :n2] = np.max(
            np.abs(points1[:, None, :] - points2[None, :, :]), axis=2
        )
    if n1:
        block = np.full((n1, n1), np.inf)
        np.fill_diagonal(block, (points1[:, 1] - points1[:, 0]) / 2.0)
        cost[:n1, n2:] = block
    if n2:
        block = np.full((n2, n2), np.inf)
        np.fill_diagonal(block, (points2[:, 1] - points2[:, 0]) / 2.0)
        cost[n1:, :n2] = block
    return cost


def compute_wasserstein_distance(
    diagram1: PersistenceDiagram,
    diagram2: PersistenceDiagram,
    p: float = 2.0
) -> float:
    """
    Compute the p-Wasserstein distance between persistence diagrams.

    Uses the L-infinity ground metric. Essential classes (infinite death)
    are ignored. GUDHI with POT is used when available; otherwise an exact
    assignment-based computation is used and a RuntimeWarning is emitted.

    Parameters
    ----------
    diagram1, diagram2 : PersistenceDiagram
        Persistence diagrams to compare
    p : float
        Wasserstein order (typically 1 or 2)

    Returns
    -------
    distance : float
        Wasserstein distance
    """
    points1 = _finite_points(diagram1)
    points2 = _finite_points(diagram2)
    if len(points1) == 0 and len(points2) == 0:
        return 0.0

    try:
        from gudhi.wasserstein import wasserstein_distance as _gudhi_wasserstein
    except (ImportError, ModuleNotFoundError):
        warnings.warn(
            "gudhi.wasserstein is unavailable (needs gudhi and POT): using the "
            "built-in assignment-based Wasserstein distance.",
            RuntimeWarning,
            stacklevel=2,
        )
    else:
        return float(
            _gudhi_wasserstein(points1, points2, order=p, internal_p=np.inf)
        )

    from scipy.optimize import linear_sum_assignment

    cost = _matching_costs(points1, points2) ** p
    allowed = np.isfinite(cost)
    big = (cost[allowed].max() + 1.0) * cost.shape[0] * 10.0
    rows, cols = linear_sum_assignment(np.where(allowed, cost, big))
    return float(cost[rows, cols].sum() ** (1.0 / p))


def compute_bottleneck_distance(
    diagram1: PersistenceDiagram,
    diagram2: PersistenceDiagram
) -> float:
    """
    Compute bottleneck distance between persistence diagrams.

    Essential classes (infinite death) are ignored. GUDHI is used when
    available; otherwise an exact threshold search over bipartite matchings
    is used and a RuntimeWarning is emitted.

    Parameters
    ----------
    diagram1, diagram2 : PersistenceDiagram
        Persistence diagrams to compare

    Returns
    -------
    distance : float
        Bottleneck distance
    """
    points1 = _finite_points(diagram1)
    points2 = _finite_points(diagram2)
    if len(points1) == 0 and len(points2) == 0:
        return 0.0

    try:
        import gudhi
    except (ImportError, ModuleNotFoundError):
        warnings.warn(
            "GUDHI is not installed: using the built-in bottleneck distance.",
            RuntimeWarning,
            stacklevel=2,
        )
    else:
        return float(gudhi.bottleneck_distance(points1, points2))

    from scipy.optimize import linear_sum_assignment

    cost = _matching_costs(points1, points2)
    candidates = np.unique(cost[np.isfinite(cost)])
    lo, hi = 0, len(candidates) - 1
    # Smallest threshold at which a perfect matching uses only allowed edges.
    while lo < hi:
        mid = (lo + hi) // 2
        blocked = (~(cost <= candidates[mid])).astype(float)
        rows, cols = linear_sum_assignment(blocked)
        if blocked[rows, cols].sum() == 0:
            hi = mid
        else:
            lo = mid + 1
    return float(candidates[lo])


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