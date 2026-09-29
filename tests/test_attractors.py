"""Tests for mneme.core.attractors — attractor detection and characterisation."""

import numpy as np
import pytest

from mneme.types import AttractorType
from mneme.core.attractors import (
    AttractorDetector,
    ClusteringDetector,
    LyapunovAnalysis,
    RecurrenceAnalysis,
    compute_correlation_dimension,
)


# ---------------------------------------------------------------------------
# RecurrenceAnalysis
# ---------------------------------------------------------------------------

class TestRecurrenceAnalysis:
    """Tests for recurrence-based attractor detection."""

    def test_compute_recurrence_matrix_symmetric(self, sine_trajectory):
        ra = RecurrenceAnalysis(threshold=0.5)
        rm = ra.compute_recurrence_matrix(sine_trajectory)
        assert rm.shape == (len(sine_trajectory), len(sine_trajectory))
        np.testing.assert_array_equal(rm, rm.T)

    def test_recurrence_matrix_binary(self, sine_trajectory):
        ra = RecurrenceAnalysis(threshold=0.5)
        rm = ra.compute_recurrence_matrix(sine_trajectory)
        assert set(np.unique(rm)).issubset({0, 1})

    def test_detect_returns_list(self, sine_trajectory):
        ra = RecurrenceAnalysis(threshold=0.3, min_persistence=0.01)
        attractors = ra.detect(sine_trajectory)
        assert isinstance(attractors, list)


# ---------------------------------------------------------------------------
# AttractorDetector (facade)
# ---------------------------------------------------------------------------

class TestAttractorDetector:
    """Tests for the AttractorDetector facade."""

    def test_recurrence_method(self):
        det = AttractorDetector(method="recurrence")
        assert isinstance(det._detector, RecurrenceAnalysis)

    def test_lyapunov_method(self):
        det = AttractorDetector(method="lyapunov")
        assert isinstance(det._detector, LyapunovAnalysis)

    def test_clustering_method(self):
        det = AttractorDetector(method="clustering")
        assert isinstance(det._detector, ClusteringDetector)

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError, match="Unknown"):
            AttractorDetector(method="nonexistent")

    def test_detect_returns_list(self, sine_trajectory):
        det = AttractorDetector(method="clustering", threshold=0.5, min_samples=5)
        attractors = det.detect(sine_trajectory)
        assert isinstance(attractors, list)


# ---------------------------------------------------------------------------
# compute_correlation_dimension
# ---------------------------------------------------------------------------

class TestComputeCorrelationDimension:
    """Tests for correlation dimension estimation."""

    def test_returns_non_negative(self, sine_trajectory):
        dim = compute_correlation_dimension(sine_trajectory)
        assert dim >= 0.0

    def test_finite_result(self, sine_trajectory):
        dim = compute_correlation_dimension(sine_trajectory)
        assert np.isfinite(dim)


# ---------------------------------------------------------------------------
# Detectors must not assign attractor types
# ---------------------------------------------------------------------------

class TestDetectorsDoNotLabel:
    """Regression: a sine wave and white noise were both labelled 'strange'."""

    @staticmethod
    def _signals():
        t = np.linspace(0, 40 * np.pi, 600)
        circle = np.column_stack([np.sin(t), np.cos(t)])
        noise = np.random.RandomState(0).standard_normal((600, 2))
        return {"circle": circle, "noise": noise}

    @pytest.mark.parametrize("method", ["recurrence", "clustering", "lyapunov"])
    @pytest.mark.parametrize("signal", ["circle", "noise"])
    def test_type_is_undetermined(self, method, signal):
        det = AttractorDetector(method=method, threshold=0.3)
        for attractor in det.detect(self._signals()[signal]):
            assert attractor.type == AttractorType.UNDETERMINED
            assert det.classify_attractor(attractor) == AttractorType.UNDETERMINED

    def test_recurrence_finds_something_on_a_circle(self):
        det = AttractorDetector(method="recurrence", threshold=0.3)
        assert len(det.detect(self._signals()["circle"])) > 0

    def test_basin_size_is_a_fraction(self):
        det = AttractorDetector(method="recurrence", threshold=0.3)
        for attractor in det.detect(self._signals()["circle"]):
            assert 0.0 < attractor.basin_size <= 1.0
            assert len(set(attractor.trajectory_indices)) == len(attractor.trajectory_indices)
