"""Tests for mneme.core.attractors — attractor detection and characterisation."""

import numpy as np
import pytest

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
