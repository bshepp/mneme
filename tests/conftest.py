"""Shared test fixtures for the Mneme test suite."""

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Synthetic 2D fields
# ---------------------------------------------------------------------------

@pytest.fixture
def gaussian_blob_field():
    """32x32 field with a single Gaussian blob at the centre."""
    size = 32
    y, x = np.mgrid[:size, :size]
    cx, cy = size / 2, size / 2
    sigma = size / 6
    field = np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma ** 2))
    return field


@pytest.fixture
def two_peak_field():
    """32x32 field with two well-separated Gaussian peaks.

    Useful for testing topology — should yield 2 zero-dimensional features.
    """
    size = 32
    y, x = np.mgrid[:size, :size]
    sigma = size / 10
    peak1 = np.exp(-((x - 8) ** 2 + (y - 16) ** 2) / (2 * sigma ** 2))
    peak2 = np.exp(-((x - 24) ** 2 + (y - 16) ** 2) / (2 * sigma ** 2))
    return peak1 + peak2


@pytest.fixture
def sinusoidal_field():
    """32x32 sinusoidal field."""
    size = 32
    y, x = np.mgrid[:size, :size]
    return np.sin(2 * np.pi * x / size) * np.cos(2 * np.pi * y / size)


# ---------------------------------------------------------------------------
# Temporal field sequences
# ---------------------------------------------------------------------------

@pytest.fixture
def temporal_field_sequence():
    """10 frames of 32x32 evolving Gaussian blob (shifts rightward)."""
    frames = []
    size = 32
    y, x = np.mgrid[:size, :size]
    sigma = size / 6
    for t in range(10):
        cx = size / 2 + t * 0.5
        cy = size / 2
        frame = np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma ** 2))
        frames.append(frame)
    return np.array(frames)


# ---------------------------------------------------------------------------
# Trajectories for attractor tests
# ---------------------------------------------------------------------------

@pytest.fixture
def sine_trajectory():
    """Simple 2D circular trajectory (non-chaotic limit cycle)."""
    t = np.linspace(0, 20 * np.pi, 2000)
    return np.column_stack([np.sin(t), np.cos(t)])


@pytest.fixture
def fixed_point_trajectory():
    """Trajectory converging to a fixed point with noise."""
    rng = np.random.RandomState(42)
    n = 500
    decay = np.exp(-np.linspace(0, 5, n))
    x = decay * rng.randn(n) * 0.1
    y = decay * rng.randn(n) * 0.1
    return np.column_stack([x, y])


@pytest.fixture
def lorenz_trajectory():
    """Short Lorenz attractor trajectory (chaotic).

    Integrated with simple Euler steps — not publication-grade but
    sufficient for testing that Lyapunov exponent code detects chaos.
    """
    dt = 0.01
    n_steps = 5000
    sigma, rho, beta = 10.0, 28.0, 8.0 / 3.0
    xyz = np.zeros((n_steps, 3))
    xyz[0] = [1.0, 1.0, 1.0]
    for i in range(n_steps - 1):
        x, y, z = xyz[i]
        xyz[i + 1] = xyz[i] + dt * np.array([
            sigma * (y - x),
            x * (rho - z) - y,
            x * y - beta * z,
        ])
    return xyz


@pytest.fixture
def periodic_1d_signal():
    """1D periodic signal for recurrence / embedding tests."""
    t = np.linspace(0, 10 * np.pi, 1000)
    return np.sin(t)


# ---------------------------------------------------------------------------
# Sparse observation fixtures for reconstruction
# ---------------------------------------------------------------------------

@pytest.fixture
def sparse_observations():
    """100 random observations on the unit square with sinusoidal ground truth."""
    rng = np.random.RandomState(42)
    positions = rng.rand(100, 2)
    values = np.sin(2 * np.pi * positions[:, 0]) * np.cos(2 * np.pi * positions[:, 1])
    values += rng.randn(100) * 0.05  # small noise
    return values, positions


# ---------------------------------------------------------------------------
# Pipeline configuration
# ---------------------------------------------------------------------------

@pytest.fixture
def minimal_pipeline_config():
    """Minimal pipeline configuration for integration tests."""
    return {
        "reconstruction": {
            "method": "ift",
            "resolution": [16, 16],
        },
        "topology": {
            "max_dimension": 1,
            "persistence_threshold": 0.01,
        },
    }
