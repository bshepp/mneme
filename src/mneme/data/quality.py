"""Quality assessment utilities (alignment with docs).

Thin wrapper around mneme.data.validation.QualityChecker for simpler imports.
"""
from __future__ import annotations

from typing import Any, Dict, Union
import numpy as np

from .validation import QualityChecker as _QC
from ..types import Field


class QualityChecker(_QC):
    """Alias of the main QualityChecker."""
    pass


def check_field(field: Union[Field, np.ndarray]) -> Dict[str, Any]:
    """Convenience function to assess field quality.

    Parameters
    ----------
    field : Field or np.ndarray
        Input field

    Returns
    -------
    Dict[str, Any]
        Quality report
    """
    return QualityChecker().check_field(field)
