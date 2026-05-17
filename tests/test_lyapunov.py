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


# ---------------------------------------------------------------------------
# Anti-overfit generalisation suite
#
# The old fixed-fraction probe in ``_linear_region`` was overfit to the
# canonical Lorenz RK4 fixture: λ₁ vs probe fraction was a smooth monotonic
# ramp with NO plateau, and only a ~3-point probe window passed the Lorenz
# gate. The same Lorenz system failed badly under modest changes (other
# initial conditions / observables / sample rates), and maps were grossly
# under-estimated. These deterministic gates (no RNG anywhere) prove the
# data-driven R²-selected scaling-region detector generalises: λ₁ stays
# in-band across initial conditions, observables and system families.
# ---------------------------------------------------------------------------


def _rk4(deriv, state0, dt, n_steps):
    """Deterministic RK4 integrator (no RNG)."""
    states = np.empty((n_steps, len(state0)))
    s = np.asarray(state0, dtype=float)
    for i in range(n_steps):
        states[i] = s
        k1 = deriv(s)
        k2 = deriv(s + 0.5 * dt * k1)
        k3 = deriv(s + 0.5 * dt * k2)
        k4 = deriv(s + dt * k3)
        s = s + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return states


def _lorenz(ic, dt=0.01, n_steps=6500, transient=100):
    sigma, rho, beta = 10.0, 28.0, 8.0 / 3.0

    def deriv(s):
        x, y, z = s
        return np.array([sigma * (y - x), x * (rho - z) - y, x * y - beta * z])

    return _rk4(deriv, ic, dt, n_steps)[transient:], dt


def _rossler(ic, dt=0.05, n_steps=8000, transient=500):
    a, b, c = 0.2, 0.2, 5.7

    def deriv(s):
        x, y, z = s
        return np.array([-y - z, x + a * y, b + z * (x - c)])

    return _rk4(deriv, ic, dt, n_steps)[transient:], dt


def _van_der_pol(mu=1.0, dt=0.05, n_steps=6500, transient=500):
    def deriv(s):
        x, v = s
        return np.array([v, mu * (1.0 - x * x) * v - x])

    return _rk4(deriv, [2.0, 0.0], dt, n_steps)[transient:], dt


def _logistic(x0=0.123, n=6000, discard=100):
    out = np.empty(n)
    x = x0
    for i in range(n):
        out[i] = x
        x = 4.0 * x * (1.0 - x)
    return out[discard:]


def _henon(n=6000, discard=100):
    a, b = 1.4, 0.3
    x, y = 0.1, 0.1
    out = np.empty(n)
    for i in range(n):
        out[i] = x
        x, y = 1.0 - a * x * x + y, b * x
    return out[discard:]


class TestGeneralisation:
    """λ₁ must stay in-band across ICs, observables and system families."""

    @pytest.mark.parametrize("ic", [[1, 1, 1], [5, -3, 10], [10, 10, 10]])
    def test_lorenz_x_across_initial_conditions(self, ic):
        traj, dt = _lorenz(ic)
        lam = largest_lyapunov(traj[:, 0], dt=dt).lambda1
        assert 0.80 <= lam <= 1.00, f"IC={ic} -> {lam}"

    def test_lorenz_y_observable(self):
        traj, dt = _lorenz([1, 1, 1])
        lam = largest_lyapunov(traj[:, 1], dt=dt).lambda1
        assert 0.80 <= lam <= 1.00, lam

    @pytest.mark.parametrize("ic", [[1, 1, 1], [0.1, 0.1, 0.1]])
    def test_rossler_x_across_initial_conditions(self, ic):
        traj, dt = _rossler(ic)
        lam = largest_lyapunov(traj[:, 0], dt=dt).lambda1
        assert 0.04 <= lam <= 0.12, f"IC={ic} -> {lam}"

    def test_logistic_map(self):
        series = _logistic()
        lam = largest_lyapunov(series, dt=1.0).lambda1
        assert 0.55 <= lam <= 0.80, lam  # true = ln 2 ≈ 0.693

    def test_henon_map(self):
        series = _henon()
        lam = largest_lyapunov(series, dt=1.0).lambda1
        assert 0.33 <= lam <= 0.50, lam  # true ≈ 0.419

    def test_van_der_pol_limit_cycle(self):
        traj, dt = _van_der_pol()
        lam = largest_lyapunov(traj[:, 0], dt=dt).lambda1
        assert abs(lam) < 0.05, lam  # non-degenerate limit cycle ⇒ ~0


def test_exact_sine_degenerate_sentinel():
    """The exact-period sine is a documented degenerate case.

    Make the degeneracy explicit rather than silently relied upon: with
    no resolvable divergence the curve is empty, ``fit_region == (0, 0)``
    and ``lambda1`` is exactly 0.0.
    """
    t = np.arange(6000)
    sig = np.sin(2 * np.pi * t / 50.0)
    res = largest_lyapunov(sig, dt=1.0)
    assert res.fit_region == (0, 0)
    assert res.divergence_curve.size <= 1
    assert res.lambda1 == 0.0
    assert res.fit_r2 == 0.0


def test_fit_r2_populated(lorenz_rk4):
    """fit_r2 is a valid R² and the scaling region is genuinely linear."""
    traj, dt = lorenz_rk4
    res = largest_lyapunov(traj[:, 0], dt=dt)
    assert 0.0 <= res.fit_r2 <= 1.0
    assert res.fit_r2 > 0.95
