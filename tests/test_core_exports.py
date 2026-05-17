"""Public API surface of mneme.core after the Tier 0 clean break."""

import pytest


def test_new_names_exported():
    import mneme.core as c

    assert hasattr(c, "largest_lyapunov")
    assert hasattr(c, "lyapunov_spectrum")
    assert hasattr(c, "surrogate_test")
    assert hasattr(c, "classify_attractor")
    assert hasattr(c, "kaplan_yorke_dimension")
    assert hasattr(c, "embed_trajectory")


def test_old_names_removed():
    import mneme.core as c

    assert not hasattr(c, "compute_lyapunov_spectrum")
    assert not hasattr(c, "classify_attractor_by_lyapunov")
