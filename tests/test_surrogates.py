"""Tests for mneme.core.surrogates (IAAFT + surrogate significance test)."""

import numpy as np
import pytest

from mneme.core.classify import classify_attractor
from mneme.core.surrogates import (
    SurrogateResult,
    iaaft_surrogates,
    min_surrogates_for,
    surrogate_test,
)
from mneme.types import AttractorType

# Smallest surrogate count at which the two-sided test can reach alpha=0.05.
# Anything lower makes "not significant" true of every input.
N_SUR = 40


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
        x = rng.randn(1500)
        with pytest.warns(RuntimeWarning, match="uninformative"):
            res = surrogate_test(x, statistic="lambda1", n=N_SUR, seed=0)
        assert isinstance(res, SurrogateResult)
        assert res.significant is False
        # End to end: noise must never be given an attractor type.
        label = classify_attractor(res.statistic_value, surrogate=res)
        assert label == AttractorType.UNDETERMINED

    @pytest.mark.parametrize("seed", [5, 17])
    def test_ar1_noise_not_significant(self, seed):
        x = _ar1(seed, length=1500, phi=0.7)
        with pytest.warns(RuntimeWarning, match="uninformative"):
            res = surrogate_test(x, statistic="lambda1", n=N_SUR, seed=0)
        assert res.significant is False
        label = classify_attractor(res.statistic_value, surrogate=res)
        assert label == AttractorType.UNDETERMINED

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
        with pytest.warns(RuntimeWarning):
            res = surrogate_test(
                rng.randn(1500), statistic="lambda1", n=N_SUR, seed=0
            )
        assert isinstance(res.effect_size, float)
        assert isinstance(res.min_sigma, float)
        assert res.min_sigma == 2.5
        assert hasattr(res, "embedding")
        assert isinstance(res.embedding, dict)
        assert isinstance(res.p_value, float)
        assert 0.0 <= res.p_value <= 1.0


class TestSurrogateGuards:
    def test_min_surrogates_for_alpha(self):
        assert min_surrogates_for(0.05) == 39
        assert min_surrogates_for(0.01) == 199

    def test_too_few_surrogates_raises(self):
        # 2/(30+1) = 0.065 > 0.05: significance would be unreachable.
        with pytest.raises(ValueError, match="cannot reach significance"):
            surrogate_test(np.random.RandomState(0).randn(500), n=30)

    def test_multidimensional_input_warns(self):
        x = np.random.RandomState(0).randn(300, 3)
        with pytest.warns(RuntimeWarning, match="first column"):
            surrogate_test(x, n=39, seed=0, emb_dim=3, delay=1, theiler=5)
