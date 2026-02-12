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
