"""Tests for analysis.metrics."""

import numpy as np
import pytest

from mneme.analysis.metrics import (
    compute_attractor_metrics,
    compute_field_statistics,
    compute_pipeline_metrics,
    compute_quality_score,
    compute_reconstruction_metrics,
    compute_spatial_correlation,
    compute_ssim,
    compute_topological_metrics,
    estimate_correlation_length,
)
from mneme.types import (
    Attractor,
    AttractorType,
    Field,
    PersistenceDiagram,
)


def test_reconstruction_metrics_perfect():
    a = np.random.RandomState(0).randn(16, 16)
    m = compute_reconstruction_metrics(a, a)
    assert m["mse"] == pytest.approx(0.0)
    assert m["correlation"] == pytest.approx(1.0)
    assert m["nmse"] == pytest.approx(0.0)
    assert "ssim" in m
    assert "snr" in m


def test_reconstruction_metrics_noisy():
    rng = np.random.RandomState(1)
    a = rng.randn(16, 16)
    b = a + 0.01 * rng.randn(16, 16)
    m = compute_reconstruction_metrics(a, b)
    assert m["correlation"] > 0.99
    assert m["mse"] > 0


def test_reconstruction_metrics_with_uncertainty():
    rng = np.random.RandomState(2)
    a = rng.randn(8, 8)
    b = a + 0.05 * rng.randn(8, 8)
    u = np.ones_like(a) * 0.1
    m = compute_reconstruction_metrics(a, b, uncertainty=u)
    assert "coverage_95" in m
    assert "normalized_residual" in m
    assert 0.0 <= m["coverage_95"] <= 1.0


def test_reconstruction_shape_mismatch():
    with pytest.raises(ValueError):
        compute_reconstruction_metrics(np.zeros((4, 4)), np.zeros((5, 5)))


def test_ssim_identical_one():
    a = np.random.RandomState(3).randn(10, 10)
    assert compute_ssim(a, a) == pytest.approx(1.0, abs=1e-3)


def test_topological_metrics_basic():
    diag = PersistenceDiagram(
        points=np.array([[0.0, 1.0], [0.5, 2.0], [1.0, 3.0]]),
        dimension=0,
    )
    m = compute_topological_metrics([diag])
    assert m["dim_0"]["n_features"] == 3
    assert m["dim_0"]["max_persistence"] == pytest.approx(2.0)
    assert m["dim_0"]["entropy"] >= 0
    assert m["total_features"] == 3


def test_topological_metrics_empty():
    diag = PersistenceDiagram(points=np.zeros((0, 2)), dimension=0)
    m = compute_topological_metrics([diag])
    assert m["dim_0"]["n_features"] == 0
    assert m["dim_0"]["total_persistence"] == 0


def test_attractor_metrics_empty():
    m = compute_attractor_metrics([])
    assert m["n_attractors"] == 0


def test_attractor_metrics_multiple():
    a1 = Attractor(
        type=AttractorType.FIXED_POINT,
        center=np.array([0.0, 0.0]),
        basin_size=1.0,
        dimension=0.0,
        lyapunov_exponents=np.array([-0.1]),
        trajectory_indices=[0, 1, 2],
    )
    a2 = Attractor(
        type=AttractorType.LIMIT_CYCLE,
        center=np.array([1.0, 1.0]),
        basin_size=2.0,
        dimension=1.0,
        lyapunov_exponents=np.array([0.0]),
        trajectory_indices=[3, 4, 5],
    )
    traj = np.zeros((10, 2))
    m = compute_attractor_metrics([a1, a2], trajectory=traj)
    assert m["n_attractors"] == 2
    assert "basin_sizes" in m
    assert "separation" in m
    assert "trajectory_coverage" in m
    assert m["trajectory_coverage"] == pytest.approx(0.6)


def test_field_statistics_2d():
    rng = np.random.RandomState(4)
    data = rng.randn(20, 20)
    s = compute_field_statistics(data)
    assert s["has_data"]
    assert s["shape"] == (20, 20)
    assert "mean" in s and "std" in s and "median" in s
    assert "p95" in s
    assert "skewness" in s and "kurtosis" in s
    assert "gradient_mean" in s
    assert "spatial_correlation" in s


def test_field_statistics_3d():
    rng = np.random.RandomState(5)
    data = rng.randn(5, 8, 8)
    s = compute_field_statistics(data)
    assert s["has_data"]
    assert "temporal_variance_mean" in s
    assert "temporal_correlation_mean" in s


def test_field_statistics_field_object():
    f = Field(data=np.ones((6, 6)))
    s = compute_field_statistics(f)
    assert s["has_data"]
    assert s["mean"] == pytest.approx(1.0)


def test_field_statistics_all_nan():
    s = compute_field_statistics(np.full((4, 4), np.nan))
    assert s["has_data"] is False


def test_spatial_correlation():
    rng = np.random.RandomState(6)
    field = rng.randn(20, 20)
    result = compute_spatial_correlation(field, max_distance=5)
    assert "distances" in result
    assert "correlations" in result
    assert "correlation_length" in result


def test_spatial_correlation_rejects_non_2d():
    with pytest.raises(ValueError):
        compute_spatial_correlation(np.ones((3, 3, 3)))


def test_estimate_correlation_length():
    distances = np.arange(10)
    correlations = np.exp(-distances / 3.0)
    length = estimate_correlation_length(distances, correlations)
    assert length > 0


def test_pipeline_metrics():
    pr = {
        "stage_a": {"computation_time": 0.5, "score": 0.9},
        "stage_b": {"computation_time": 1.5, "value": np.array([3.0])},
    }
    m = compute_pipeline_metrics(pr)
    assert m["execution_times"]["total"] == pytest.approx(2.0)
    assert "stage_a_metrics" in m


def test_quality_score_basic():
    q = {"metrics": {"snr": 20}}
    score = compute_quality_score(q)
    assert score >= 0
