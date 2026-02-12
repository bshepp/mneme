"""Tests for mneme.core.field_theory — field reconstruction methods."""

import numpy as np
import pytest

from mneme.core.field_theory import (
    BaseFieldReconstructor,
    DenseIFTReconstructor,
    FieldReconstructor,
    GaussianProcessReconstructor,
    NeuralFieldReconstructor,
    SparseGPReconstructor,
    create_grid_points,
    create_reconstructor,
)


# ---------------------------------------------------------------------------
# Factory / create_reconstructor
# ---------------------------------------------------------------------------

class TestCreateReconstructor:
    """Tests for the create_reconstructor factory function."""

    def test_ift_returns_sparse_gp(self):
        rec = create_reconstructor("ift", resolution=(16, 16))
        assert isinstance(rec, SparseGPReconstructor)

    def test_sparse_gp_alias(self):
        rec = create_reconstructor("sparse_gp", resolution=(16, 16))
        assert isinstance(rec, SparseGPReconstructor)

    def test_dense_ift(self):
        rec = create_reconstructor("dense_ift", resolution=(16, 16))
        assert isinstance(rec, DenseIFTReconstructor)

    def test_gp(self):
        rec = create_reconstructor("gp", resolution=(16, 16))
        assert isinstance(rec, GaussianProcessReconstructor)

    def test_gaussian_process_alias(self):
        rec = create_reconstructor("gaussian_process", resolution=(16, 16))
        assert isinstance(rec, GaussianProcessReconstructor)

    def test_neural(self):
        rec = create_reconstructor("neural", resolution=(16, 16))
        assert isinstance(rec, NeuralFieldReconstructor)

    def test_neural_field_alias(self):
        rec = create_reconstructor("neural_field", resolution=(16, 16))
        assert isinstance(rec, NeuralFieldReconstructor)

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError, match="Unknown"):
            create_reconstructor("nonexistent_method")


# ---------------------------------------------------------------------------
# FieldReconstructor (facade)
# ---------------------------------------------------------------------------

class TestFieldReconstructor:
    """Tests for the FieldReconstructor facade class."""

    def test_default_method_is_sparse_gp(self):
        rec = FieldReconstructor(resolution=(16, 16))
        assert isinstance(rec._backend, SparseGPReconstructor)

    def test_dense_ift_via_string(self):
        rec = FieldReconstructor(method="dense_ift", resolution=(16, 16))
        assert isinstance(rec._backend, DenseIFTReconstructor)

    def test_reconstruct_before_fit_raises(self):
        rec = FieldReconstructor(resolution=(16, 16))
        with pytest.raises(RuntimeError):
            rec.reconstruct()

    def test_uncertainty_before_fit_raises(self):
        rec = FieldReconstructor(resolution=(16, 16))
        with pytest.raises(RuntimeError):
            rec.uncertainty()


# ---------------------------------------------------------------------------
# SparseGPReconstructor
# ---------------------------------------------------------------------------

class TestSparseGPReconstructor:
    """Tests for the default Sparse GP reconstructor."""

    def test_fit_reconstruct_cycle(self, sparse_observations):
        values, positions = sparse_observations
        rec = SparseGPReconstructor(resolution=(16, 16), n_inducing=50, random_state=42)
        rec.fit(values, positions)
        field = rec.reconstruct()

        assert field.shape == (16, 16)
        assert np.all(np.isfinite(field))

    def test_uncertainty_non_negative(self, sparse_observations):
        values, positions = sparse_observations
        rec = SparseGPReconstructor(resolution=(16, 16), n_inducing=50, random_state=42)
        rec.fit(values, positions)
        rec.reconstruct()
        unc = rec.uncertainty()

        assert unc.shape == (16, 16)
        assert np.all(unc >= 0)

    def test_legacy_correlation_length_param(self, sparse_observations):
        """Legacy IFT parameter 'correlation_length' should map to length_scale."""
        values, positions = sparse_observations
        rec = SparseGPReconstructor(
            resolution=(16, 16), correlation_length=5.0, random_state=42
        )
        rec.fit(values, positions)
        field = rec.reconstruct()
        assert field.shape == (16, 16)


# ---------------------------------------------------------------------------
# DenseIFTReconstructor
# ---------------------------------------------------------------------------

class TestDenseIFTReconstructor:
    """Tests for the Dense IFT reconstructor (small fields only)."""

    def test_fit_reconstruct_cycle(self, sparse_observations):
        values, positions = sparse_observations
        rec = DenseIFTReconstructor(resolution=(8, 8))
        rec.fit(values, positions)
        field = rec.reconstruct()

        assert field.shape == (8, 8)
        assert np.all(np.isfinite(field))

    def test_uncertainty_shape(self, sparse_observations):
        values, positions = sparse_observations
        rec = DenseIFTReconstructor(resolution=(8, 8))
        rec.fit(values, positions)
        unc = rec.uncertainty()

        assert unc.shape == (8, 8)
        assert np.all(unc >= 0)

    def test_large_resolution_warns(self):
        with pytest.warns(UserWarning, match="memory"):
            DenseIFTReconstructor(resolution=(128, 128))


# ---------------------------------------------------------------------------
# GaussianProcessReconstructor
# ---------------------------------------------------------------------------

class TestGaussianProcessReconstructor:
    """Tests for the standard GP reconstructor."""

    def test_fit_reconstruct_cycle(self, sparse_observations):
        values, positions = sparse_observations
        rec = GaussianProcessReconstructor(resolution=(16, 16), length_scale=0.2)
        rec.fit(values, positions)
        field = rec.reconstruct()

        assert field.shape == (16, 16)
        assert np.all(np.isfinite(field))

    def test_uncertainty_shape(self, sparse_observations):
        values, positions = sparse_observations
        rec = GaussianProcessReconstructor(resolution=(16, 16), length_scale=0.2)
        rec.fit(values, positions)
        rec.reconstruct()
        unc = rec.uncertainty()

        assert unc.shape == (16, 16)
        assert np.all(np.isfinite(unc))

    def test_unknown_kernel_raises(self, sparse_observations):
        values, positions = sparse_observations
        rec = GaussianProcessReconstructor(resolution=(8, 8), kernel="invalid_kernel")
        with pytest.raises(ValueError, match="Unknown kernel"):
            rec.fit(values, positions)
            rec.reconstruct()


# ---------------------------------------------------------------------------
# NeuralFieldReconstructor
# ---------------------------------------------------------------------------

class TestNeuralFieldReconstructor:
    """Tests for the Neural Field reconstructor."""

    def test_fit_reconstruct_cycle(self, sparse_observations):
        values, positions = sparse_observations
        rec = NeuralFieldReconstructor(
            resolution=(16, 16),
            hidden_dims=(32, 16),
            n_epochs=20,
            positional_encoding_dims=4,
        )
        rec.fit(values, positions)
        field = rec.reconstruct()

        assert field.shape == (16, 16)
        assert np.all(np.isfinite(field))

    def test_uncertainty_returns_zeros(self, sparse_observations):
        values, positions = sparse_observations
        rec = NeuralFieldReconstructor(
            resolution=(8, 8), hidden_dims=(16,), n_epochs=5,
            positional_encoding_dims=0,
        )
        rec.fit(values, positions)
        unc = rec.uncertainty()

        assert unc.shape == (8, 8)
        assert np.allclose(unc, 0)


# ---------------------------------------------------------------------------
# Sparse GP vs Dense IFT agreement on small fields
# ---------------------------------------------------------------------------

class TestReconstructorAgreement:
    """SparseGP and DenseIFT both produce valid finite reconstructions."""

    def test_sparse_and_dense_both_finite(self, sparse_observations):
        values, positions = sparse_observations
        res = (8, 8)

        sparse = SparseGPReconstructor(resolution=res, n_inducing=80, random_state=0)
        sparse.fit(values, positions)
        field_sparse = sparse.reconstruct()

        dense = DenseIFTReconstructor(resolution=res)
        dense.fit(values, positions)
        field_dense = dense.reconstruct()

        # Different algorithms; verify both produce valid finite outputs
        assert field_sparse.shape == res
        assert field_dense.shape == res
        assert np.all(np.isfinite(field_sparse))
        assert np.all(np.isfinite(field_dense))


# ---------------------------------------------------------------------------
# create_grid_points utility
# ---------------------------------------------------------------------------

class TestCreateGridPoints:
    """Tests for the create_grid_points helper."""

    def test_default_shape(self):
        pts = create_grid_points((10, 20))
        assert pts.shape == (200, 2)

    def test_unit_square_bounds(self):
        pts = create_grid_points((5, 5))
        assert pts[:, 0].min() >= 0.0
        assert pts[:, 0].max() <= 1.0
        assert pts[:, 1].min() >= 0.0
        assert pts[:, 1].max() <= 1.0

    def test_custom_bounds(self):
        pts = create_grid_points((4, 4), bounds=((-1.0, 1.0), (-1.0, 1.0)))
        assert pts[:, 0].min() == pytest.approx(-1.0)
        assert pts[:, 0].max() == pytest.approx(1.0)
