"""Experimental components must announce themselves; core ones must not."""

import warnings

import numpy as np
import pytest

from mneme import ExperimentalWarning
from mneme.analysis.pipeline import MnemePipeline, default_config
from mneme.core.attractors import AttractorDetector
from mneme.core.field_theory import (
    NeuralFieldReconstructor,
    SubsetGPReconstructor,
    WienerFilterReconstructor,
)
from mneme.core.topology import PersistentHomology
from mneme.models import SymbolicRegressor, create_field_vae


class TestExperimentalComponentsWarn:
    def test_attractor_detector(self):
        with pytest.warns(ExperimentalWarning, match="AttractorDetector"):
            AttractorDetector(method="clustering")

    def test_neural_field(self):
        with pytest.warns(ExperimentalWarning, match="NeuralFieldReconstructor"):
            NeuralFieldReconstructor(resolution=(8, 8))

    def test_symbolic_regressor(self):
        with pytest.warns(ExperimentalWarning, match="SymbolicRegressor"):
            SymbolicRegressor(niterations=1)

    def test_vae(self):
        pytest.importorskip("torch")
        with pytest.warns(ExperimentalWarning, match="FieldAutoencoder"):
            create_field_vae((16, 16), latent_dim=4)


class TestCoreComponentsDoNotWarn:
    def test_core_constructors_are_silent(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error", ExperimentalWarning)
            SubsetGPReconstructor(resolution=(8, 8))
            WienerFilterReconstructor(resolution=(8, 8))
            PersistentHomology()

    @pytest.mark.parametrize("kind", ["standard", "bioelectric"])
    def test_default_pipelines_use_core_stages_only(self, kind):
        assert "attractors" not in default_config(kind)
        with warnings.catch_warnings():
            warnings.simplefilter("error", ExperimentalWarning)
            pipe = MnemePipeline(default_config(kind))
        assert pipe.attractor_detector is None
        result = pipe.run(np.random.RandomState(0).rand(3, 32, 32))
        assert result.analysis_result.attractors is None
