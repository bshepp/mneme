"""Tests for data.preprocessors."""

import numpy as np
import pytest

from mneme.data.preprocessors import (
    Denoiser,
    Interpolator,
    Normalizer,
    Registrator,
)


def test_denoiser_gaussian():
    rng = np.random.RandomState(0)
    data = rng.randn(16, 16)
    d = Denoiser(method="gaussian", sigma=1.0)
    result = d.fit_transform(data)
    assert result.shape == data.shape
    # smoothing reduces variance
    assert result.std() < data.std()


def test_denoiser_median():
    data = np.random.RandomState(1).randn(16, 16)
    d = Denoiser(method="median")
    result = d.fit_transform(data)
    assert result.shape == data.shape


def test_denoiser_unfitted_raises():
    d = Denoiser(method="gaussian")
    with pytest.raises(RuntimeError):
        d.transform(np.ones((4, 4)))


def test_denoiser_unknown_method():
    d = Denoiser(method="bogus")
    d.is_fitted = True
    with pytest.raises(ValueError):
        d.transform(np.ones((4, 4)))


def test_normalizer_zscore():
    data = np.random.RandomState(2).randn(8, 8) * 5 + 3
    n = Normalizer(method="z_score")
    result = n.fit_transform(data)
    assert abs(result.mean()) < 1e-6
    assert abs(result.std() - 1.0) < 1e-6


def test_normalizer_minmax():
    data = np.array([[0.0, 5.0], [10.0, 15.0]])
    n = Normalizer(method="min_max")
    result = n.fit_transform(data)
    assert result.min() == pytest.approx(0.0)
    assert result.max() == pytest.approx(1.0)


def test_normalizer_robust():
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    n = Normalizer(method="robust")
    result = n.fit_transform(data)
    assert result.shape == data.shape


def test_normalizer_per_frame_3d():
    rng = np.random.RandomState(3)
    data = rng.randn(4, 8, 8) * np.array([1, 2, 3, 4])[:, None, None]
    n = Normalizer(method="z_score", per_frame=True)
    result = n.fit_transform(data)
    for i in range(4):
        assert abs(result[i].mean()) < 1e-6


def test_normalizer_clip():
    data = np.array([[-100.0, 0.0, 100.0], [-50.0, 50.0, 0.0]])
    n = Normalizer(method="z_score", clip_percentile=10.0)
    result = n.fit_transform(data)
    assert result.shape == data.shape


def test_normalizer_unknown_method():
    with pytest.raises(ValueError):
        Normalizer(method="bogus").fit(np.ones((4, 4)))


def test_normalizer_unfitted():
    with pytest.raises(RuntimeError):
        Normalizer().transform(np.ones((4, 4)))


def test_registrator_first():
    rng = np.random.RandomState(0)
    data = rng.randn(3, 16, 16)
    r = Registrator(reference="first", max_shift=2)
    out = r.fit_transform(data)
    assert out.shape == data.shape


def test_registrator_mean_and_median():
    data = np.random.RandomState(1).randn(3, 8, 8)
    for ref in ("mean", "median"):
        r = Registrator(reference=ref, max_shift=1)
        assert r.fit_transform(data).shape == data.shape


def test_registrator_requires_3d():
    r = Registrator()
    with pytest.raises(ValueError):
        r.fit(np.ones((4, 4)))


def test_registrator_unknown_reference():
    r = Registrator(reference="bogus")
    with pytest.raises(ValueError):
        r.fit(np.ones((2, 4, 4)))


def test_registrator_unfitted():
    with pytest.raises(RuntimeError):
        Registrator().transform(np.ones((2, 4, 4)))


def test_interpolator_2d():
    data = np.arange(16, dtype=float).reshape(4, 4)
    interp = Interpolator(target_shape=(8, 8))
    out = interp.fit_transform(data)
    assert out.shape == (8, 8)


def test_interpolator_3d():
    data = np.random.RandomState(4).randn(3, 4, 4)
    interp = Interpolator(target_shape=(8, 8))
    out = interp.fit_transform(data)
    assert out.shape == (3, 8, 8)


def test_interpolator_unfitted():
    with pytest.raises(RuntimeError):
        Interpolator(target_shape=(2, 2)).transform(np.ones((4, 4)))


def test_interpolator_invalid_dim():
    interp = Interpolator(target_shape=(2, 2)).fit(np.ones((4, 4)))
    with pytest.raises(ValueError):
        interp.transform(np.ones((2, 2, 2, 2)))
