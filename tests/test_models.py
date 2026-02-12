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
