"""Import-and-run smoke test for the migrated analysis scripts.

Only checks the scripts import and their Lyapunov code path runs against
synthetic data with the new API. Numeric reconciliation is Tier 1.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_analyze_physionet_imports():
    pytest.importorskip("wfdb")
    _load("analyze_physionet")


@pytest.mark.slow
def test_deep_analysis_lyapunov_path_runs():
    mod = _load("deep_analysis")
    rng = np.random.RandomState(0)
    pca = rng.randn(900, 3)
    res = mod.lyapunov_from_pca(pca, label="smoke")
    assert "error" in res or "lambda1" in res


def test_analyze_betse_imports():
    _load("analyze_betse")
