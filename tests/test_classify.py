"""Tests for mneme.core.classify — gated attractor classification."""

import numpy as np
import pytest

from mneme.types import AttractorType


def test_undetermined_member_exists():
    assert AttractorType.UNDETERMINED == "undetermined"
    assert AttractorType.UNDETERMINED.value == "undetermined"
