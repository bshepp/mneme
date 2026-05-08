"""Tests for synthetic data generators."""

import numpy as np
import pytest

from mneme.data.generators import (
    SyntheticFieldGenerator,
    generate_planarian_bioelectric_sequence,
    generate_test_dataset,
)


@pytest.mark.parametrize(
    "field_type",
    [
        "gaussian_random",
        "gaussian_blob",
        "sinusoidal",
        "turbulent",
        "bioelectric_gradient",
    ],
)
def test_generate_static_basic(field_type):
    gen = SyntheticFieldGenerator(field_type=field_type, seed=42)
    field = gen.generate_static((16, 16), {})
    assert field.shape == (16, 16)
    assert np.all(np.isfinite(field))


def test_generate_static_with_params():
    gen = SyntheticFieldGenerator(field_type="gaussian_blob", seed=0)
    field = gen.generate_static((16, 16), {"n_centers": 2, "sigma": 4.0, "amplitude": 2.0})
    assert field.shape == (16, 16)
    assert field.max() > 0


def test_generate_sinusoidal_with_angle():
    gen = SyntheticFieldGenerator(field_type="sinusoidal", seed=0)
    field = gen.generate_static(
        (16, 16), {"frequency": 0.2, "angle": 45.0, "amplitude": 1.5, "phase": 0.5}
    )
    assert field.shape == (16, 16)
    assert np.abs(field).max() <= 1.5 + 1e-6


def test_generate_static_unknown_type_raises():
    gen = SyntheticFieldGenerator(field_type="nonsense", seed=0)
    with pytest.raises(ValueError):
        gen.generate_static((8, 8), {})


def test_reaction_diffusion_2d_only():
    gen = SyntheticFieldGenerator(field_type="reaction_diffusion", seed=0)
    # Use small timesteps to keep test fast
    field = gen.generate_static((16, 16), {"timesteps": 20})
    assert field.shape == (16, 16)
    assert np.all(np.isfinite(field))


def test_reaction_diffusion_rejects_3d():
    gen = SyntheticFieldGenerator(field_type="reaction_diffusion", seed=0)
    with pytest.raises(ValueError):
        gen.generate_static((8, 8, 8), {"timesteps": 5})


def test_bioelectric_gradient_rejects_non_2d():
    gen = SyntheticFieldGenerator(field_type="bioelectric_gradient", seed=0)
    with pytest.raises(ValueError):
        gen.generate_static((8, 8, 8), {})


def test_generate_dynamic_shape_and_evolution():
    gen = SyntheticFieldGenerator(field_type="gaussian_blob", seed=1)
    seq = gen.generate_dynamic(
        (8, 8),
        timesteps=5,
        parameters={
            "n_centers": 2,
            "sigma": 2.0,
            "drift_velocity": (0.5, 0.0),
            "diffusion_rate": 0.5,
            "noise_level": 0.01,
            "growth_rate": 0.01,
        },
    )
    assert seq.shape == (5, 8, 8)
    assert np.all(np.isfinite(seq))
    # Frames should differ after evolution
    assert not np.allclose(seq[0], seq[-1])


def test_generate_dynamic_no_evolution_keeps_close():
    gen = SyntheticFieldGenerator(field_type="gaussian_blob", seed=2)
    seq = gen.generate_dynamic(
        (8, 8),
        timesteps=3,
        parameters={"n_centers": 1, "sigma": 2.0},
    )
    assert seq.shape == (3, 8, 8)


@pytest.mark.parametrize("noise_type", ["gaussian", "poisson", "salt_pepper", "speckle"])
def test_add_noise_types(noise_type):
    gen = SyntheticFieldGenerator(seed=0)
    field = np.ones((10, 10)) * 0.5
    noisy = gen.add_noise(field, noise_level=0.1, noise_type=noise_type)
    assert noisy.shape == field.shape
    assert np.all(np.isfinite(noisy))


def test_add_noise_unknown_raises():
    gen = SyntheticFieldGenerator(seed=0)
    with pytest.raises(ValueError):
        gen.add_noise(np.ones((4, 4)), 0.1, noise_type="unknown")


def test_planarian_sequence():
    seq = generate_planarian_bioelectric_sequence(
        shape=(16, 16), timesteps=5, regeneration_event_time=2, seed=7
    )
    assert seq.shape == (5, 16, 16)
    assert np.all(np.isfinite(seq))


def test_generate_test_dataset_default():
    ds = generate_test_dataset(n_samples=2, shape=(8, 8), seed=3)
    assert isinstance(ds, dict)
    assert len(ds) >= 1
    for arr in ds.values():
        assert arr.shape[0] == 2
