"""Tests for mneme.core.classify — gated classification + Kaplan-Yorke."""

import numpy as np

from mneme.core.classify import classify_attractor, kaplan_yorke_dimension
from mneme.core.surrogates import SurrogateResult
from mneme.types import AttractorType


def _sig(significant: bool) -> SurrogateResult:
    # Construct with ALL required fields of the CURRENT SurrogateResult dataclass.
    # effect_size/p_value consistent with `significant`.
    return SurrogateResult(
        statistic_name="lambda1",
        statistic_value=0.9,
        null_distribution=np.zeros(10),
        p_value=0.001 if significant else 0.5,
        n_surrogates=10,
        alpha=0.05,
        significant=significant,
        effect_size=10.0 if significant else 0.5,
        min_sigma=2.5,
        embedding={},
    )


def test_undetermined_member_exists():
    assert AttractorType.UNDETERMINED.value == "undetermined"


class TestClassifyAttractor:
    def test_positive_lambda_no_surrogate_is_undetermined(self):
        assert classify_attractor(0.9) == AttractorType.UNDETERMINED

    def test_positive_lambda_insignificant_surrogate_is_undetermined(self):
        assert classify_attractor(0.9, surrogate=_sig(False)) == AttractorType.UNDETERMINED

    def test_positive_lambda_significant_surrogate_is_strange(self):
        assert classify_attractor(0.9, surrogate=_sig(True)) == AttractorType.STRANGE

    def test_near_zero_is_limit_cycle(self):
        assert classify_attractor(0.001, oscillatory=True) == AttractorType.LIMIT_CYCLE

    def test_negative_is_fixed_point(self):
        assert classify_attractor(-0.5) == AttractorType.FIXED_POINT


class TestKaplanYorke:
    def test_all_negative_returns_zero(self):
        assert kaplan_yorke_dimension(np.array([-1.0, -2.0, -3.0])) == 0.0

    def test_lorenz_like_spectrum(self):
        dim = kaplan_yorke_dimension(np.array([0.9, 0.0, -14.6]))
        assert 2.0 < dim < 3.0

    def test_descending_order_enforced(self):
        assert kaplan_yorke_dimension(np.array([-14.6, 0.0, 0.9])) > 2.0
