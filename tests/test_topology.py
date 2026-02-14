"""Tests for mneme.core.topology — topological data analysis."""

import numpy as np
import pytest

from mneme.core.topology import (
    PersistentHomology,
    compute_betti_curve,
    compute_wasserstein_distance,
    compute_bottleneck_distance,
    field_to_point_cloud,
    filter_persistence_diagram,
)
from mneme.types import FiltrationMethod, PersistenceDiagram


# ---------------------------------------------------------------------------
# PersistentHomology
# ---------------------------------------------------------------------------

class TestPersistentHomology:
    """Tests for the PersistentHomology analyzer."""

    def test_compute_persistence_returns_diagrams(self, gaussian_blob_field):
        ph = PersistentHomology(max_dimension=1, persistence_threshold=0.0)
        diagrams = ph.compute_persistence(gaussian_blob_field)

        assert isinstance(diagrams, list)
        assert len(diagrams) >= 1
        for d in diagrams:
            assert isinstance(d, PersistenceDiagram)

    def test_two_peaks_detected(self, two_peak_field):
        """A field with two peaks should produce at least 2 zero-dim features."""
        ph = PersistentHomology(max_dimension=0, persistence_threshold=0.0)
        diagrams = ph.compute_persistence(two_peak_field)

        dim0 = diagrams[0]
        # The exact count depends on the implementation / fallback, but we
        # should see at least 2 features in dimension 0.
        assert len(dim0.points) >= 2

    def test_3d_field_raises(self):
        ph = PersistentHomology()
        with pytest.raises(ValueError, match="2D"):
            ph.compute_persistence(np.zeros((4, 4, 4)))

    def test_extract_features_finite(self, gaussian_blob_field):
        ph = PersistentHomology(max_dimension=1, persistence_threshold=0.0)
        diagrams = ph.compute_persistence(gaussian_blob_field)
        features = ph.extract_features(diagrams)

        assert isinstance(features, np.ndarray)
        assert np.all(np.isfinite(features))

    def test_extract_features_empty_diagram(self):
        empty_diag = PersistenceDiagram(points=np.empty((0, 2)), dimension=0)
        ph = PersistentHomology()
        features = ph.extract_features([empty_diag])

        assert np.allclose(features, 0.0)

    def test_persistence_image_shape(self, gaussian_blob_field):
        ph = PersistentHomology(max_dimension=0, persistence_threshold=0.0)
        diagrams = ph.compute_persistence(gaussian_blob_field)
        if len(diagrams[0].points) > 0:
            img = ph.compute_persistence_image(diagrams[0], resolution=(20, 20))
            assert img.shape == (20, 20)

    def test_persistence_landscape_shape(self, gaussian_blob_field):
        ph = PersistentHomology(max_dimension=0, persistence_threshold=0.0)
        diagrams = ph.compute_persistence(gaussian_blob_field)
        if len(diagrams[0].points) > 0:
            landscape = ph.compute_persistence_landscape(diagrams[0], k=3, resolution=50)
            assert landscape.shape == (3, 50)

    def test_superlevel_filtration(self, gaussian_blob_field):
        ph = PersistentHomology(
            max_dimension=0,
            filtration=FiltrationMethod.SUPERLEVEL,
            persistence_threshold=0.0,
        )
        diagrams = ph.compute_persistence(gaussian_blob_field)
        assert len(diagrams) >= 1


# ---------------------------------------------------------------------------
# filter_persistence_diagram
# ---------------------------------------------------------------------------

class TestFilterPersistenceDiagram:
    """Tests for the filter_persistence_diagram utility."""

    def test_filters_low_persistence(self):
        pts = np.array([[0.0, 0.5], [0.0, 0.01], [0.1, 0.9]])
        diag = PersistenceDiagram(points=pts, dimension=0)
        filtered = filter_persistence_diagram(diag, threshold=0.1)

        assert len(filtered.points) == 2  # (0,0.5) and (0.1,0.9) survive
        assert filtered.threshold == 0.1

    def test_all_removed(self):
        pts = np.array([[0.0, 0.01], [0.5, 0.51]])
        diag = PersistenceDiagram(points=pts, dimension=0)
        filtered = filter_persistence_diagram(diag, threshold=1.0)

        assert len(filtered.points) == 0


# ---------------------------------------------------------------------------
# compute_betti_curve
# ---------------------------------------------------------------------------

class TestComputeBettiCurve:
    """Tests for the Betti curve computation."""

    def test_betti_curve_shape(self):
        pts = np.array([[0.0, 1.0], [0.5, 2.0]])
        diag = PersistenceDiagram(points=pts, dimension=0)
        filt_vals = np.linspace(0.0, 2.5, 50)
        curve = compute_betti_curve(diag, filt_vals)

        assert curve.shape == (50,)
        assert np.all(curve >= 0)

    def test_empty_diagram_gives_zero_curve(self):
        diag = PersistenceDiagram(points=np.empty((0, 2)), dimension=0)
        filt_vals = np.linspace(0, 1, 20)
        curve = compute_betti_curve(diag, filt_vals)

        assert np.allclose(curve, 0.0)


# ---------------------------------------------------------------------------
# field_to_point_cloud
# ---------------------------------------------------------------------------

class TestFieldToPointCloud:
    """Tests for converting a 2D field to a point cloud."""

    def test_output_shape(self, gaussian_blob_field):
        pc = field_to_point_cloud(gaussian_blob_field, max_points=100)
        assert pc.ndim == 2
        assert pc.shape[1] == 2
        assert pc.shape[0] <= 100

    def test_normalized_coords(self, gaussian_blob_field):
        pc = field_to_point_cloud(gaussian_blob_field, normalize_coords=True)
        assert pc[:, 0].min() >= 0.0
        assert pc[:, 0].max() <= 1.0
        assert pc[:, 1].min() >= 0.0
        assert pc[:, 1].max() <= 1.0

    def test_threshold_method(self, gaussian_blob_field):
        pc = field_to_point_cloud(gaussian_blob_field, method="threshold")
        assert pc.ndim == 2
        assert pc.shape[1] == 2

    def test_3d_raises(self):
        with pytest.raises(ValueError, match="2D"):
            field_to_point_cloud(np.zeros((4, 4, 4)))


# ---------------------------------------------------------------------------
# Wasserstein / Bottleneck distance
# ---------------------------------------------------------------------------

class TestWassersteinDistance:
    """Tests for Wasserstein distance between persistence diagrams."""

    def test_identical_diagrams_zero(self):
        pts = np.array([[0.0, 1.0], [0.5, 2.0], [1.0, 3.0]])
        d1 = PersistenceDiagram(points=pts, dimension=0)
        d2 = PersistenceDiagram(points=pts.copy(), dimension=0)
        dist = compute_wasserstein_distance(d1, d2, p=2.0)
        assert dist == pytest.approx(0.0, abs=1e-6)

    def test_different_diagrams_positive(self):
        pts1 = np.array([[0.0, 1.0], [0.5, 2.0]])
        pts2 = np.array([[0.0, 1.5], [0.5, 3.0]])
        d1 = PersistenceDiagram(points=pts1, dimension=0)
        d2 = PersistenceDiagram(points=pts2, dimension=0)
        dist = compute_wasserstein_distance(d1, d2, p=2.0)
        assert dist > 0.0

    def test_empty_diagrams_zero(self):
        d1 = PersistenceDiagram(points=np.empty((0, 2)), dimension=0)
        d2 = PersistenceDiagram(points=np.empty((0, 2)), dimension=0)
        dist = compute_wasserstein_distance(d1, d2)
        assert dist == 0.0

    def test_symmetry(self):
        pts1 = np.array([[0.0, 1.0], [1.0, 2.5]])
        pts2 = np.array([[0.0, 2.0]])
        d1 = PersistenceDiagram(points=pts1, dimension=0)
        d2 = PersistenceDiagram(points=pts2, dimension=0)
        assert compute_wasserstein_distance(d1, d2) == pytest.approx(
            compute_wasserstein_distance(d2, d1), abs=1e-6
        )

    def test_from_computed_persistence(self, gaussian_blob_field, two_peak_field):
        """Wasserstein between persistence of two different fields should be > 0."""
        ph = PersistentHomology(max_dimension=0, persistence_threshold=0.0)
        diags1 = ph.compute_persistence(gaussian_blob_field)
        diags2 = ph.compute_persistence(two_peak_field)
        dist = compute_wasserstein_distance(diags1[0], diags2[0])
        assert dist > 0.0


class TestBottleneckDistance:
    """Tests for bottleneck distance between persistence diagrams."""

    def test_identical_diagrams_zero(self):
        pts = np.array([[0.0, 1.0], [0.5, 2.0]])
        d1 = PersistenceDiagram(points=pts, dimension=0)
        d2 = PersistenceDiagram(points=pts.copy(), dimension=0)
        dist = compute_bottleneck_distance(d1, d2)
        assert dist == pytest.approx(0.0, abs=1e-6)

    def test_different_diagrams_positive(self):
        pts1 = np.array([[0.0, 1.0]])
        pts2 = np.array([[0.0, 5.0]])
        d1 = PersistenceDiagram(points=pts1, dimension=0)
        d2 = PersistenceDiagram(points=pts2, dimension=0)
        dist = compute_bottleneck_distance(d1, d2)
        assert dist > 0.0

    def test_empty_diagrams_zero(self):
        d1 = PersistenceDiagram(points=np.empty((0, 2)), dimension=0)
        d2 = PersistenceDiagram(points=np.empty((0, 2)), dimension=0)
        dist = compute_bottleneck_distance(d1, d2)
        assert dist == 0.0

    def test_bottleneck_leq_wasserstein(self):
        """Bottleneck distance is always <= Wasserstein-infinity."""
        pts1 = np.array([[0.0, 1.0], [0.5, 2.0], [1.0, 3.0]])
        pts2 = np.array([[0.0, 1.5], [0.5, 3.0], [1.0, 4.0]])
        d1 = PersistenceDiagram(points=pts1, dimension=0)
        d2 = PersistenceDiagram(points=pts2, dimension=0)
        b = compute_bottleneck_distance(d1, d2)
        w = compute_wasserstein_distance(d1, d2, p=2.0)
        # Bottleneck is max matching cost; Wasserstein is sum — bottleneck <= wasserstein
        assert b <= w + 1e-6
