"""Tests for mneme.core.surrogates (IAAFT + surrogate significance test)."""

import numpy as np
import pytest

from mneme.core.surrogates import (
    SurrogateResult,
    iaaft_surrogates,
    surrogate_test,
)


class TestIAAFT:
    def test_shape_and_amplitude_preserved(self):
        rng = np.random.RandomState(1)
        x = np.cumsum(rng.randn(512))
        sur = iaaft_surrogates(x, n=5, seed=0)
        assert sur.shape == (5, 512)
        np.testing.assert_allclose(np.sort(sur[0]), np.sort(x), rtol=0, atol=1e-6)

    def test_power_spectrum_approx_preserved(self):
        rng = np.random.RandomState(2)
        x = np.sin(np.linspace(0, 60, 1024)) + 0.1 * rng.randn(1024)
        sur = iaaft_surrogates(x, n=3, seed=1)
        px = np.abs(np.fft.rfft(x - x.mean()))
        ps = np.abs(np.fft.rfft(sur[0] - sur[0].mean()))
        r = np.corrcoef(px, ps)[0, 1]
        assert r > 0.95

    def test_reproducible_with_seed(self):
        rng = np.random.RandomState(3)
        x = rng.randn(256)
        a = iaaft_surrogates(x, n=2, seed=42)
        b = iaaft_surrogates(x, n=2, seed=42)
        np.testing.assert_array_equal(a, b)


def _ar1(seed: int, length: int = 1500, phi: float = 0.7) -> np.ndarray:
    rng = np.random.RandomState(seed)
    x = np.zeros(length)
    for i in range(1, length):
        x[i] = phi * x[i - 1] + rng.randn()
    return x


class TestSurrogateTest:
    @pytest.mark.parametrize("seed", [4, 11, 23])
    def test_white_noise_not_significant(self, seed):
        rng = np.random.RandomState(seed)
        res = surrogate_test(
            rng.randn(1500), statistic="lambda1", n=30, seed=0
        )
        assert isinstance(res, SurrogateResult)
        assert res.significant is False

    @pytest.mark.parametrize("seed", [5, 17])
    def test_ar1_noise_not_significant(self, seed):
        x = _ar1(seed, length=1500, phi=0.7)
        res = surrogate_test(x, statistic="lambda1", n=30, seed=0)
        assert res.significant is False

    @pytest.mark.slow
    def test_lorenz_is_significant(self, lorenz_rk4):
        # L=4000, n=40 is the shortest/cheapest config that clears the
        # two-sided rank + effect-size gate (~163 s wall -> @slow).
        # n>=40 is required because the two-sided p-value floor is
        # 2/(n+1); n=40 gives 2/41 = 0.0488 <= 0.05. The IAAFT
        # surrogates score a spuriously *higher* lambda1 than the
        # deterministic Lorenz series, so the deviation is on the LOW
        # side (effect_size ~ -9.7): a one-sided upper-tail test would
        # wrongly miss it, which is exactly why the test is two-sided.
        traj, dt = lorenz_rk4
        res = surrogate_test(
            traj[:4000, 0], statistic="lambda1", n=40, seed=0, dt=dt
        )
        assert res.significant is True
        assert res.p_value <= 0.05
        assert abs(res.effect_size) > 3.0

    def test_result_has_effect_size_fields(self):
        rng = np.random.RandomState(4)
        res = surrogate_test(rng.randn(1500), statistic="lambda1", n=30, seed=0)
        assert isinstance(res.effect_size, float)
        assert isinstance(res.min_sigma, float)
        assert res.min_sigma == 2.5
        assert hasattr(res, "embedding")
        assert isinstance(res.embedding, dict)
        assert isinstance(res.p_value, float)
        assert 0.0 <= res.p_value <= 1.0
