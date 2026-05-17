"""Validation tests for mneme.core.lyapunov against known systems."""

import numpy as np
import pytest

from mneme.core.lyapunov import LyapunovResult, largest_lyapunov, lyapunov_spectrum


def test_fixture_shapes(lorenz_rk4, rossler_rk4):
    assert lorenz_rk4[0].shape[1] == 3
    assert rossler_rk4[0].shape[1] == 3


class TestLargestLyapunov:
    def test_lorenz_lambda1(self, lorenz_rk4):
        traj, dt = lorenz_rk4
        res = largest_lyapunov(traj[:, 0], dt=dt)
        assert isinstance(res, LyapunovResult)
        assert 0.85 <= res.lambda1 <= 0.97

    def test_lorenz_scale_invariant(self, lorenz_rk4):
        traj, dt = lorenz_rk4
        a = largest_lyapunov(traj[:, 0], dt=dt).lambda1
        b = largest_lyapunov(1000.0 * traj[:, 0], dt=dt).lambda1
        assert abs(a - b) <= 0.01 * abs(a)

    def test_rossler_lambda1(self, rossler_rk4):
        traj, dt = rossler_rk4
        res = largest_lyapunov(traj[:, 0], dt=dt)
        assert 0.04 <= res.lambda1 <= 0.11

    def test_sine_near_zero(self):
        t = np.arange(6000)
        sig = np.sin(2 * np.pi * t / 50.0)
        res = largest_lyapunov(sig, dt=1.0)
        assert abs(res.lambda1) < 0.02

    def test_result_fields_populated(self, lorenz_rk4):
        traj, dt = lorenz_rk4
        res = largest_lyapunov(traj[:, 0], dt=dt)
        assert res.divergence_curve.ndim == 1
        assert res.fit_region[0] < res.fit_region[1]
        assert res.emb_dim >= 2 and res.delay >= 1 and res.theiler >= 1


class TestLyapunovSpectrum:
    def test_lorenz_spectrum_sum_and_lambda1(self, lorenz_rk4):
        traj, dt = lorenz_rk4
        spec = lyapunov_spectrum(traj, dt=dt)
        assert len(spec) == 3
        assert 0.80 <= spec[0] <= 1.05
        assert -16.0 <= float(np.sum(spec)) <= -11.0

    def test_short_trajectory_raises(self):
        with pytest.raises(ValueError, match="too short"):
            lyapunov_spectrum(np.column_stack([np.arange(20.0)] * 3), dt=0.01)


def test_lorenz_lambda1_runs_in_default_ci(lorenz_rk4):
    """Downsized, NOT marked slow — guards the headline claim in default CI."""
    traj, dt = lorenz_rk4
    res = largest_lyapunov(traj[:3000, 0], dt=dt)
    assert res.lambda1 > 0.3  # clearly positive, fast
