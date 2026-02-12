"""Tests for mneme.core.attractors — attractor detection and characterisation."""

import numpy as np
import pytest

from mneme.core.attractors import (
    AttractorDetector,
    ClusteringDetector,
    LyapunovAnalysis,
    RecurrenceAnalysis,
    classify_attractor_by_lyapunov,
    compute_correlation_dimension,
    compute_lyapunov_spectrum,
    embed_trajectory,
    kaplan_yorke_dimension,
)
from mneme.types import AttractorType


# ---------------------------------------------------------------------------
# embed_trajectory
# ---------------------------------------------------------------------------

class TestEmbedTrajectory:
    """Tests for delay embedding."""

    def test_1d_embedding_shape(self, periodic_1d_signal):
        embedded = embed_trajectory(periodic_1d_signal, embedding_dimension=3, time_delay=1)
        # n_points = len(signal) - (dim-1)*delay = 1000 - 2 = 998
        assert embedded.shape == (998, 3)

    def test_2d_input_shape(self, sine_trajectory):
        embedded = embed_trajectory(sine_trajectory, embedding_dimension=2, time_delay=5)
        expected_rows = len(sine_trajectory) - (2 - 1) * 5
        assert embedded.shape == (expected_rows, 4)  # 2 dims * 2 embedding

    def test_short_series_raises(self):
        short = np.array([1.0, 2.0])
        with pytest.raises(ValueError, match="too short"):
            embed_trajectory(short, embedding_dimension=5, time_delay=2)

    def test_delay_one(self, periodic_1d_signal):
        emb = embed_trajectory(periodic_1d_signal, embedding_dimension=2, time_delay=1)
        # First column should equal signal[0:N], second column signal[1:N+1]
        np.testing.assert_array_equal(emb[:, 0], periodic_1d_signal[:len(emb)])
        np.testing.assert_array_equal(emb[:, 1], periodic_1d_signal[1:len(emb) + 1])


# ---------------------------------------------------------------------------
# classify_attractor_by_lyapunov
# ---------------------------------------------------------------------------

class TestClassifyAttractorByLyapunov:
    """Tests for Lyapunov-spectrum-based classification."""

    def test_fixed_point(self):
        spectrum = np.array([-0.5, -1.0, -2.0])
        assert classify_attractor_by_lyapunov(spectrum) == AttractorType.FIXED_POINT

    def test_limit_cycle(self):
        spectrum = np.array([0.0, -0.5, -1.0])
        assert classify_attractor_by_lyapunov(spectrum) == AttractorType.LIMIT_CYCLE

    def test_quasi_periodic(self):
        spectrum = np.array([0.0, 0.0, -1.0])
        assert classify_attractor_by_lyapunov(spectrum) == AttractorType.QUASI_PERIODIC

    def test_strange_attractor(self):
        spectrum = np.array([0.9, 0.0, -14.0])
        assert classify_attractor_by_lyapunov(spectrum) == AttractorType.STRANGE


# ---------------------------------------------------------------------------
# kaplan_yorke_dimension
# ---------------------------------------------------------------------------

class TestKaplanYorkeDimension:
    """Tests for the Kaplan-Yorke dimension estimate."""

    def test_all_negative_returns_zero(self):
        spectrum = np.array([-1.0, -2.0, -3.0])
        dim = kaplan_yorke_dimension(spectrum)
        assert dim == 0.0

    def test_lorenz_like_spectrum(self):
        # Lorenz-like: [+0.9, 0, -14.6]
        spectrum = np.array([0.9, 0.0, -14.6])
        dim = kaplan_yorke_dimension(spectrum)
        # D_KY = 2 + (0.9 + 0.0) / 14.6 ≈ 2.06
        assert 2.0 < dim < 3.0

    def test_descending_order_enforced(self):
        # Pass in wrong order — function should sort
        spectrum = np.array([-14.6, 0.0, 0.9])
        dim = kaplan_yorke_dimension(spectrum)
        assert dim > 2.0


# ---------------------------------------------------------------------------
# compute_lyapunov_spectrum
# ---------------------------------------------------------------------------

class TestComputeLyapunovSpectrum:
    """Tests for the Wolf-algorithm Lyapunov spectrum."""

    def test_sine_trajectory_non_chaotic(self, sine_trajectory):
        spectrum = compute_lyapunov_spectrum(sine_trajectory, dt=0.01)
        # Non-chaotic: largest exponent should be near zero or negative
        assert spectrum[0] < 0.5  # generous upper bound

    @pytest.mark.slow
    def test_lorenz_has_positive_exponent(self, lorenz_trajectory):
        spectrum = compute_lyapunov_spectrum(lorenz_trajectory, dt=0.01)
        # Lorenz is chaotic: largest exponent should be positive
        assert spectrum[0] > 0.0

    def test_short_trajectory_raises(self):
        short = np.column_stack([np.arange(10), np.arange(10)])
        with pytest.raises(ValueError, match="too short"):
            compute_lyapunov_spectrum(short, dt=0.01)

    def test_1d_input_embeds_automatically(self, periodic_1d_signal):
        spectrum = compute_lyapunov_spectrum(periodic_1d_signal, dt=0.01)
        assert len(spectrum) == 3  # default embedding dim


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
