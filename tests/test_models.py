"""Tests for mneme.models — VAE and symbolic regression."""

import numpy as np
import pytest

from mneme.models import (
    FieldAutoencoder,
    SymbolicRegressor,
    TrainingResult,
    VAEOutput,
    create_field_vae,
    discover_field_dynamics,
)


# ---------------------------------------------------------------------------
# FieldAutoencoder / create_field_vae
# ---------------------------------------------------------------------------

class TestFieldAutoencoder:
    """Tests for the convolutional VAE."""

    def test_create_field_vae(self):
        vae = create_field_vae(input_shape=(32, 32), latent_dim=8)
        assert isinstance(vae, FieldAutoencoder)

    def test_forward_pass_output_shape(self):
        import torch

        vae = create_field_vae(input_shape=(32, 32), latent_dim=8)
        vae.eval()
        x = torch.randn(2, 1, 32, 32)
        with torch.no_grad():
            output = vae(x)

        assert isinstance(output, VAEOutput)
        assert output.reconstruction.shape == (2, 1, 32, 32)
        assert output.mu.shape[1] == 8
        assert output.log_var.shape[1] == 8
        assert output.z.shape[1] == 8

    def test_encode_fields(self):
        vae = create_field_vae(input_shape=(32, 32), latent_dim=8)
        fields = np.random.randn(4, 32, 32).astype(np.float32)
        latent = vae.encode_fields(fields)

        assert latent.shape == (4, 8)
        assert np.all(np.isfinite(latent))

    def test_interpolate(self):
        vae = create_field_vae(input_shape=(32, 32), latent_dim=8)
        a = np.random.randn(32, 32).astype(np.float32)
        b = np.random.randn(32, 32).astype(np.float32)
        interp = vae.interpolate(a, b, n_steps=5)

        assert interp.shape[0] == 5
        assert interp.shape[-2:] == (32, 32)


# ---------------------------------------------------------------------------
# SymbolicRegressor
# ---------------------------------------------------------------------------

class TestSymbolicRegressor:
    """Tests for symbolic regression (uses linear fallback without PySR)."""

    def test_fit_predict_cycle(self):
        rng = np.random.RandomState(0)
        X = rng.rand(100, 2)
        y = 2.0 * X[:, 0] + 3.0 * X[:, 1] + rng.randn(100) * 0.1
        sr = SymbolicRegressor(niterations=5)
        sr.fit(X, y, variable_names=["x0", "x1"])
        pred = sr.predict(X)

        assert pred.shape == (100,)
        assert np.all(np.isfinite(pred))

    def test_predict_before_fit_raises(self):
        sr = SymbolicRegressor()
        with pytest.raises(RuntimeError):
            sr.predict(np.zeros((5, 2)))

    def test_get_equations_returns_list(self):
        rng = np.random.RandomState(0)
        X = rng.rand(50, 2)
        y = X[:, 0] + X[:, 1]
        sr = SymbolicRegressor(niterations=5)
        sr.fit(X, y)
        eqs = sr.get_equations()

        assert isinstance(eqs, list)
        assert len(eqs) > 0

    def test_score(self):
        rng = np.random.RandomState(0)
        X = rng.rand(80, 2)
        y = X[:, 0] + X[:, 1]
        sr = SymbolicRegressor(niterations=5)
        sr.fit(X, y)
        r2 = sr.score(X, y)

        assert isinstance(r2, float)
        # Linear data should be fit well even by fallback
        assert r2 > 0.8


# ---------------------------------------------------------------------------
# discover_field_dynamics
# ---------------------------------------------------------------------------

class TestDiscoverFieldDynamics:
    """Tests for the convenience function."""

    def test_returns_expected_keys(self):
        rng = np.random.RandomState(42)
        # 10 frames of 8x8
        field_seq = rng.rand(10, 8, 8)
        result = discover_field_dynamics(field_seq, dt=1.0, niterations=5)

        assert isinstance(result, dict)
        assert "equations" in result
        assert "best_equation" in result
        assert "r2_score" in result

    def test_small_field_sequence(self):
        """5 frames of 8x8 — minimum viable input."""
        rng = np.random.RandomState(0)
        field_seq = rng.rand(5, 8, 8)
        result = discover_field_dynamics(field_seq, dt=0.5, niterations=2)

        assert isinstance(result["best_equation"], str)
        assert len(result["best_equation"]) > 0
        assert isinstance(result["r2_score"], float)

    def test_spatial_features_included(self):
        rng = np.random.RandomState(42)
        field_seq = rng.rand(8, 8, 8)
        result = discover_field_dynamics(
            field_seq, dt=1.0, spatial_features=True, niterations=2
        )

        assert "laplacian_u" in result["features_used"]
        assert "du_dx" in result["features_used"]
        assert "du_dy" in result["features_used"]

    def test_no_spatial_features(self):
        rng = np.random.RandomState(42)
        field_seq = rng.rand(8, 8, 8)
        result = discover_field_dynamics(
            field_seq, dt=1.0, spatial_features=False, niterations=2
        )

        assert result["features_used"] == ["u"]

    def test_2d_input_raises(self):
        with pytest.raises(ValueError, match="3D"):
            discover_field_dynamics(np.zeros((10, 10)), dt=1.0)

    def test_regressor_is_returned(self):
        rng = np.random.RandomState(42)
        result = discover_field_dynamics(rng.rand(5, 8, 8), niterations=2)
        assert result["regressor"] is not None


class TestSymbolicRegressorEdgeCases:
    """Additional edge-case tests for SymbolicRegressor."""

    def test_1d_input(self):
        """Single-feature input should be reshaped automatically."""
        x = np.linspace(0, 10, 50)
        y = 2.0 * x + 1.0
        sr = SymbolicRegressor(niterations=2)
        sr.fit(x, y, variable_names=["t"])
        pred = sr.predict(x)
        assert pred.shape == (50,)

    def test_get_best_equation_before_fit(self):
        sr = SymbolicRegressor()
        assert sr.get_best_equation() == "<no_equation_found>"

    def test_latex_output(self):
        rng = np.random.RandomState(0)
        X = rng.rand(50, 2)
        y = X[:, 0] + X[:, 1]
        sr = SymbolicRegressor(niterations=2)
        sr.fit(X, y)
        latex_str = sr.latex()
        assert isinstance(latex_str, str)
        assert len(latex_str) > 0

    def test_is_available_property(self):
        sr = SymbolicRegressor()
        assert isinstance(sr.is_available, bool)
