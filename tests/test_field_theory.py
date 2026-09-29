"""Tests for mneme.core.field_theory — field reconstruction methods."""

import warnings

import numpy as np
import pytest

from mneme.core import field_theory
from mneme.core.field_theory import (
    BaseFieldReconstructor,
    FieldReconstructor,
    GaussianProcessReconstructor,
    NeuralFieldReconstructor,
    SubsetGPReconstructor,
    WienerFilterReconstructor,
    create_grid_points,
    create_reconstructor,
)
from mneme.types import ReconstructionMethod


def _truth(points: np.ndarray) -> np.ndarray:
    """Smooth known field on the unit square."""
    return np.sin(2 * np.pi * points[:, 0]) * np.cos(2 * np.pi * points[:, 1])


@pytest.fixture
def known_field():
    """300 noisy samples of a known field, plus the truth on a 24x24 grid."""
    rng = np.random.RandomState(0)
    positions = rng.uniform(0.0, 1.0, (300, 2))
    values = _truth(positions) + 0.02 * rng.standard_normal(300)
    resolution = (24, 24)
    truth = _truth(create_grid_points(resolution)).reshape(resolution)
    return values, positions, resolution, truth


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


# ---------------------------------------------------------------------------
# Factory / create_reconstructor
# ---------------------------------------------------------------------------

class TestCreateReconstructor:
    """Tests for the create_reconstructor factory function."""

    def test_default_is_subset_gp(self):
        assert isinstance(create_reconstructor(resolution=(16, 16)), SubsetGPReconstructor)

    def test_gp_subset(self):
        rec = create_reconstructor("gp_subset", resolution=(16, 16))
        assert isinstance(rec, SubsetGPReconstructor)

    @pytest.mark.parametrize("name", ["wiener_filter", "wiener"])
    def test_wiener_filter(self, name):
        rec = create_reconstructor(name, resolution=(16, 16))
        assert isinstance(rec, WienerFilterReconstructor)

    @pytest.mark.parametrize("name", ["gp", "gaussian_process"])
    def test_gp(self, name):
        rec = create_reconstructor(name, resolution=(16, 16))
        assert isinstance(rec, GaussianProcessReconstructor)

    @pytest.mark.parametrize("name", ["neural", "neural_field"])
    def test_neural(self, name):
        rec = create_reconstructor(name, resolution=(16, 16))
        assert isinstance(rec, NeuralFieldReconstructor)

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError, match="Unknown"):
            create_reconstructor("nonexistent_method")


class TestDeprecatedNames:
    """Old names keep working and say what replaced them."""

    @pytest.mark.parametrize(
        "name, cls",
        [
            ("ift", SubsetGPReconstructor),
            ("sparse_gp", SubsetGPReconstructor),
            ("sparse", SubsetGPReconstructor),
            ("dense_ift", WienerFilterReconstructor),
        ],
    )
    def test_method_names(self, name, cls):
        with pytest.warns(DeprecationWarning, match="deprecated"):
            rec = create_reconstructor(name, resolution=(16, 16))
        assert isinstance(rec, cls)
        with pytest.warns(DeprecationWarning, match="deprecated"):
            facade = FieldReconstructor(method=name, resolution=(16, 16))
        assert isinstance(facade._backend, cls)

    def test_enum_ift_selects_subset_gp(self):
        with pytest.warns(DeprecationWarning):
            rec = FieldReconstructor(method=ReconstructionMethod.IFT, resolution=(8, 8))
        assert rec.method is ReconstructionMethod.GP_SUBSET

    @pytest.mark.parametrize(
        "old, new",
        [
            ("SparseGPReconstructor", SubsetGPReconstructor),
            ("IFTReconstructor", SubsetGPReconstructor),
            ("DenseIFTReconstructor", WienerFilterReconstructor),
        ],
    )
    def test_class_names_are_the_same_class(self, old, new):
        import mneme.core

        with pytest.warns(DeprecationWarning, match=new.__name__):
            assert getattr(field_theory, old) is new
        with pytest.warns(DeprecationWarning, match=new.__name__):
            assert getattr(mneme.core, old) is new

    def test_n_inducing_maps_to_n_subset(self):
        with pytest.warns(DeprecationWarning, match="n_subset"):
            rec = SubsetGPReconstructor(resolution=(8, 8), n_inducing=40)
        assert rec.n_subset == 40
        assert rec.n_inducing == 40


# ---------------------------------------------------------------------------
# FieldReconstructor (facade)
# ---------------------------------------------------------------------------

class TestFieldReconstructor:
    """Tests for the FieldReconstructor facade class."""

    def test_default_method_is_subset_gp(self):
        rec = FieldReconstructor(resolution=(16, 16))
        assert isinstance(rec._backend, SubsetGPReconstructor)
        assert rec.method is ReconstructionMethod.GP_SUBSET

    def test_wiener_via_string(self):
        rec = FieldReconstructor(method="wiener_filter", resolution=(16, 16))
        assert isinstance(rec._backend, WienerFilterReconstructor)

    def test_reconstruct_before_fit_raises(self):
        rec = FieldReconstructor(resolution=(16, 16))
        with pytest.raises(RuntimeError):
            rec.reconstruct()

    def test_uncertainty_before_fit_raises(self):
        rec = FieldReconstructor(resolution=(16, 16))
        with pytest.raises(RuntimeError):
            rec.uncertainty()

    def test_fit_reconstruct_without_uncertainty_reports_none(self, sparse_observations):
        values, positions = sparse_observations
        rec = FieldReconstructor(
            method="neural_field", resolution=(8, 8), hidden_dims=(16,),
            n_epochs=5, positional_encoding_dims=0,
        )
        result = rec.fit_reconstruct(values, positions)
        assert result.uncertainty is None
        assert result.field.data.shape == (8, 8)


# ---------------------------------------------------------------------------
# Accuracy against a known field
# ---------------------------------------------------------------------------

class TestReconstructionAccuracy:
    """Reconstructions must recover a known field, not merely be finite.

    The truth has standard deviation 0.5, so a constant prediction scores
    an RMSE of about 0.5. Observation noise is 0.02.
    """

    def test_subset_gp_recovers_field(self, known_field):
        values, positions, resolution, truth = known_field
        rec = SubsetGPReconstructor(resolution=resolution, random_state=0)
        field = rec.fit(values, positions).reconstruct()
        assert rec.n_discarded_ == 0
        assert _rmse(field, truth) < 0.05

    def test_standard_gp_recovers_field(self, known_field):
        values, positions, resolution, truth = known_field
        rec = GaussianProcessReconstructor(resolution=resolution, length_scale=0.2)
        field = rec.fit(values, positions).reconstruct()
        assert _rmse(field, truth) < 0.05

    def test_wiener_filter_beats_a_constant(self, known_field):
        values, positions, _, _ = known_field
        resolution = (16, 16)
        truth = _truth(create_grid_points(resolution)).reshape(resolution)
        rec = WienerFilterReconstructor(
            resolution=resolution, correlation_length=2.0, noise_var=0.01
        )
        field = rec.fit(values, positions).reconstruct()
        assert _rmse(field, truth) < 0.5 * float(np.std(truth))
        # Not collapsed to a flat field.
        assert float(np.std(field)) > 0.5 * float(np.std(truth))

    def test_subset_gp_uncertainty_is_calibrated(self, known_field):
        values, positions, resolution, truth = known_field
        rec = SubsetGPReconstructor(resolution=resolution, random_state=0)
        field = rec.fit(values, positions).reconstruct()
        std = rec.uncertainty()
        inside = np.abs(field - truth) <= 1.96 * std
        # Nominal 95%; allow for a smooth field being easier than the prior.
        assert inside.mean() > 0.85

    def test_subset_discards_and_reports(self, known_field):
        values, positions, resolution, truth = known_field
        rec = SubsetGPReconstructor(resolution=resolution, n_subset=50, random_state=0)
        rec.fit(values, positions)
        assert rec.n_used_ == 50
        assert rec.n_discarded_ == 250

    def test_fixed_hyperparameters_are_not_optimised(self, known_field):
        values, positions, resolution, _ = known_field
        rec = SubsetGPReconstructor(
            resolution=resolution, length_scale=0.5,
            optimize_hyperparameters=False, random_state=0,
        )
        rec.fit(values, positions)
        fitted = rec._gp.kernel_.get_params()
        assert fitted["k1__k2__length_scale"] == pytest.approx(0.5)
        assert fitted["k1__k1__constant_value"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Per-backend mechanics
# ---------------------------------------------------------------------------

class TestSubsetGPReconstructor:
    def test_uncertainty_non_negative(self, sparse_observations):
        values, positions = sparse_observations
        rec = SubsetGPReconstructor(resolution=(16, 16), n_subset=50, random_state=42)
        rec.fit(values, positions)
        rec.reconstruct()
        unc = rec.uncertainty()
        assert unc.shape == (16, 16)
        assert np.all(unc >= 0)

    def test_legacy_correlation_length_param(self, sparse_observations):
        """'correlation_length' in pixels maps to a length scale."""
        values, positions = sparse_observations
        rec = SubsetGPReconstructor(
            resolution=(16, 16), correlation_length=5.0, random_state=42
        )
        assert rec.length_scale == pytest.approx(5.0 / 16)
        rec.fit(values, positions)
        assert rec.reconstruct().shape == (16, 16)


class TestWienerFilterReconstructor:
    def test_correlation_length_is_converted_from_pixels(self):
        rec = WienerFilterReconstructor(resolution=(8, 16), correlation_length=4.0)
        assert rec.correlation_length == pytest.approx(4.0 / 16)
        assert rec.correlation_length_pixels == 4.0

    def test_uncertainty_shape(self, sparse_observations):
        values, positions = sparse_observations
        rec = WienerFilterReconstructor(resolution=(8, 8))
        rec.fit(values, positions)
        unc = rec.uncertainty()
        assert unc.shape == (8, 8)
        assert np.all(unc >= 0)

    def test_large_resolution_warns(self):
        with pytest.warns(UserWarning, match="memory"):
            WienerFilterReconstructor(resolution=(128, 128))


class TestGaussianProcessReconstructor:
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


class TestNeuralFieldReconstructor:
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

    def test_uncertainty_is_not_implemented(self, sparse_observations):
        values, positions = sparse_observations
        rec = NeuralFieldReconstructor(
            resolution=(8, 8), hidden_dims=(16,), n_epochs=5,
            positional_encoding_dims=0,
        )
        rec.fit(values, positions)
        with pytest.raises(NotImplementedError):
            rec.uncertainty()


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
