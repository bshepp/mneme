"""Tests for mneme.core.lyapunov."""

import numpy as np


def test_lorenz_rk4_fixture_shape(lorenz_rk4):
    traj, dt = lorenz_rk4
    assert traj.shape[1] == 3
    assert traj.shape[0] >= 6000
    assert dt == 0.01
    assert np.all(np.isfinite(traj))


def test_rossler_rk4_fixture_shape(rossler_rk4):
    traj, dt = rossler_rk4
    assert traj.shape[1] == 3
    assert np.all(np.isfinite(traj))
