"""Tests for mneme.analysis.pipeline — end-to-end pipeline."""

import numpy as np
import pytest

from mneme.analysis.pipeline import (
    MnemePipeline,
    PipelineResult,
    create_bioelectric_pipeline,
    create_standard_pipeline,
)


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------

class TestPipelineFactories:
    """Tests that factory functions construct pipelines without error."""

    def test_create_standard_pipeline(self):
        pipe = create_standard_pipeline()
        assert isinstance(pipe, MnemePipeline)

    def test_create_bioelectric_pipeline(self):
        pipe = create_bioelectric_pipeline()
        assert isinstance(pipe, MnemePipeline)

    def test_create_with_custom_config(self, minimal_pipeline_config):
        pipe = MnemePipeline(minimal_pipeline_config)
        assert isinstance(pipe, MnemePipeline)


# ---------------------------------------------------------------------------
# End-to-end run with synthetic data
# ---------------------------------------------------------------------------

class TestPipelineEndToEnd:
    """Integration tests running the pipeline on small synthetic fields."""

    @pytest.mark.integration
    def test_standard_pipeline_on_synthetic(self):
        """Standard pipeline should succeed on a small random field."""
        rng = np.random.RandomState(42)
        data = rng.rand(32, 32)

        config = {
            "preprocessing": {
                "denoise": {"enabled": True, "method": "gaussian", "sigma": 1.0},
                "normalize": {"enabled": True, "method": "z_score"},
                "register": {"enabled": False},
                "interpolate": {"enabled": False},
            },
            "reconstruction": {
                "method": "gaussian_process",
                "resolution": [16, 16],
                "parameters": {"kernel": "rbf", "length_scale": 10.0},
            },
            "topology": {
                "max_dimension": 1,
                "persistence_threshold": 0.01,
            },
        }

        pipe = MnemePipeline(config)
        result = pipe.run(data)

        assert isinstance(result, PipelineResult)
        assert result.success is True
        assert result.execution_time > 0

    @pytest.mark.integration
    def test_pipeline_topology_disabled(self):
        """Pipeline should work when topology analysis is skipped."""
        rng = np.random.RandomState(42)
        data = rng.rand(32, 32)

        config = {
            "preprocessing": {
                "denoise": {"enabled": False},
                "normalize": {"enabled": True, "method": "min_max"},
                "register": {"enabled": False},
                "interpolate": {"enabled": False},
            },
            "reconstruction": {
                "method": "gaussian_process",
                "resolution": [16, 16],
                "parameters": {"kernel": "rbf", "length_scale": 10.0},
            },
            # No topology section — should be skipped
        }

        pipe = MnemePipeline(config)
        result = pipe.run(data)

        assert isinstance(result, PipelineResult)
        assert result.success is True


# ---------------------------------------------------------------------------
# Defaults, failures and skipped stages must be visible
# ---------------------------------------------------------------------------

from mneme.analysis.pipeline import default_config, merge_config


class TestDefaults:
    def test_empty_config_uses_defaults(self):
        """Regression: the CLI passed {} and every component was disabled."""
        for factory in (create_standard_pipeline, create_bioelectric_pipeline):
            pipe = factory({})
            assert pipe.topology_analyzer is not None
            assert pipe.preprocessor is not None
            assert pipe.reconstructor is not None

    def test_default_config_is_a_fresh_copy(self):
        first = default_config("standard")
        first["topology"]["backend"] = "rips"
        assert default_config("standard")["topology"]["backend"] == "cubical"

    def test_unknown_pipeline_raises(self):
        with pytest.raises(ValueError, match="Unknown pipeline"):
            default_config("nonexistent")

    def test_merge_config_overlays_recursively(self):
        merged = merge_config(
            default_config("standard"), {"topology": {"max_dimension": 1}}
        )
        assert merged["topology"]["max_dimension"] == 1
        assert merged["topology"]["backend"] == "cubical"
        assert "preprocessing" in merged


class TestStageOutcomesAreReported:
    CONFIG = {"topology": {"max_dimension": 1, "persistence_threshold": 0.01}}

    def test_topology_runs_and_produces_features(self):
        data = np.random.RandomState(0).rand(24, 24)
        result = MnemePipeline(self.CONFIG).run(data)
        assert result.success is True
        assert result.failed_stages == []
        assert result.analysis_result.topology is not None
        assert result.stage_results["topology"]["total_features"] > 0

    def test_failed_stage_is_not_reported_as_success(self):
        """Regression: a NaN field returned success with '0 features'."""
        data = np.random.RandomState(0).rand(24, 24)
        data[3, 3] = np.nan
        result = MnemePipeline(self.CONFIG).run(data)
        assert result.success is False
        assert result.failed_stages == ["topology"]
        assert any("NaN" in message for message in result.errors)
        assert result.stage_results["topology"]["status"] == "failed"
        # The rest of the result is still available.
        assert result.analysis_result is not None
        assert result.analysis_result.topology is None

    def test_injected_stage_failure_is_reported(self, monkeypatch):
        pipe = MnemePipeline(self.CONFIG)

        def boom(_field):
            raise RuntimeError("injected failure")

        monkeypatch.setattr(pipe.topology_analyzer, "compute_persistence", boom)
        result = pipe.run(np.random.RandomState(0).rand(16, 16))
        assert result.success is False
        assert "injected failure" in result.errors[0]

    def test_reconstruction_is_skipped_without_observations(self):
        """Regression: the input was returned as a 'reconstruction'."""
        config = {"reconstruction": {"method": "gp_subset", "resolution": [16, 16]}}
        result = MnemePipeline(config).run(np.random.RandomState(0).rand(16, 16))
        assert result.success is True
        assert result.analysis_result.reconstruction is None
        assert result.stage_results["reconstruction"]["status"] == "skipped"

    def test_reconstruction_runs_with_observations(self):
        rng = np.random.RandomState(0)
        positions = rng.rand(60, 2)
        config = {"reconstruction": {"method": "gp_subset", "resolution": [12, 12]}}
        result = MnemePipeline(config).run({
            "field": rng.rand(12, 12),
            "observations": np.sin(4 * positions[:, 0]),
            "positions": positions,
        })
        assert result.success is True
        assert result.stage_results["reconstruction"]["status"] == "completed"
        assert result.analysis_result.reconstruction.field.data.shape == (12, 12)
