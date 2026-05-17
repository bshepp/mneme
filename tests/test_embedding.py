"""Tests for mneme.core.embedding."""

import numpy as np
import pytest

from mneme.core.embedding import (
    cao_embedding_dimension,
    embed_trajectory,
    estimate_embedding_parameters,
    mutual_information_delay,
    theiler_window,
)


class TestEmbedTrajectory:
    def test_1d_embedding_shape(self):
        sig = np.sin(np.linspace(0, 50, 1000))
        emb = embed_trajectory(sig, embedding_dimension=3, time_delay=1)
        assert emb.shape == (998, 3)

    def test_delay_one_columns(self):
        sig = np.arange(100.0)
        emb = embed_trajectory(sig, embedding_dimension=2, time_delay=1)
        np.testing.assert_array_equal(emb[:, 0], sig[: len(emb)])
        np.testing.assert_array_equal(emb[:, 1], sig[1 : len(emb) + 1])

    def test_short_series_raises(self):
        with pytest.raises(ValueError, match="too short"):
            embed_trajectory(np.array([1.0, 2.0]), embedding_dimension=5, time_delay=2)


class TestTheilerWindow:
    def test_positive_int(self):
        sig = np.sin(np.linspace(0, 80 * np.pi, 4000))
        w = theiler_window(sig)
        assert isinstance(w, int) and w >= 1

    def test_sine_window_near_quarter_period(self):
        t = np.arange(4000)
        sig = np.sin(2 * np.pi * t / 100.0)
        w = theiler_window(sig)
        assert 15 <= w <= 40


class TestMutualInformationDelay:
    def test_recovers_quarter_period_on_sine(self):
        t = np.arange(5000)
        sig = np.sin(2 * np.pi * t / 40.0)
        d = mutual_information_delay(sig, max_delay=60)
        assert 6 <= d <= 14

    def test_returns_positive_int(self):
        rng = np.random.RandomState(0)
        d = mutual_information_delay(rng.randn(2000), max_delay=50)
        assert isinstance(d, int) and d >= 1


class TestCaoEmbeddingDimension:
    def test_lorenz_dimension(self, lorenz_rk4):
        traj, _ = lorenz_rk4
        d = cao_embedding_dimension(traj[:, 0], delay=8, max_dim=8)
        assert 2 <= d <= 5

    def test_returns_positive_int(self):
        sig = np.sin(np.linspace(0, 100, 3000))
        d = cao_embedding_dimension(sig, delay=10, max_dim=8)
        assert isinstance(d, int) and d >= 1


class TestEstimateEmbeddingParameters:
    def test_returns_two_positive_ints(self, lorenz_rk4):
        traj, _ = lorenz_rk4
        dim, delay = estimate_embedding_parameters(traj[:, 0])
        assert isinstance(dim, int) and isinstance(delay, int)
        assert dim >= 1 and delay >= 1
