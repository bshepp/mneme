"""Surrogate-gated attractor classification and Kaplan-Yorke dimension.

`classify_attractor` REFUSES to return STRANGE without passed surrogate
evidence — positive λ₁ alone yields UNDETERMINED. This is the central
credibility fix.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..types import AttractorType
from .surrogates import SurrogateResult


def classify_attractor(
    lambda1: float,
    *,
    surrogate: Optional[SurrogateResult] = None,
    oscillatory: bool = False,
    zero_tol: Optional[float] = None,
) -> AttractorType:
    """Classify an attractor from λ₁, gating chaos on surrogate evidence.

    Parameters
    ----------
    lambda1 : float
        Largest Lyapunov exponent (e.g. from `largest_lyapunov`).
    surrogate : SurrogateResult, optional
        Result of `surrogate_test`. STRANGE is only returned when this is
        provided AND `surrogate.significant` is True.
    oscillatory : bool
        Hint that near-zero λ₁ corresponds to a limit cycle vs fixed point.
    zero_tol : float, optional
        Half-width of the "λ₁ ≈ 0" band. Defaults to the surrogate null
        spread (std) when available, else 0.01.

    Returns
    -------
    AttractorType
        STRANGE only with significant surrogate evidence; otherwise
        UNDETERMINED (positive λ₁), LIMIT_CYCLE / FIXED_POINT (≈0), or
        FIXED_POINT (negative).
    """
    if zero_tol is None:
        if surrogate is not None and surrogate.null_distribution.size > 1:
            zero_tol = max(0.01, float(np.std(surrogate.null_distribution)))
        else:
            zero_tol = 0.01

    if abs(lambda1) <= zero_tol:
        return AttractorType.LIMIT_CYCLE if oscillatory else AttractorType.FIXED_POINT

    if lambda1 < 0:
        return AttractorType.FIXED_POINT

    # lambda1 clearly positive — chaos claim requires surrogate evidence.
    if surrogate is not None and surrogate.significant:
        return AttractorType.STRANGE
    return AttractorType.UNDETERMINED


def kaplan_yorke_dimension(spectrum: np.ndarray) -> float:
    """Kaplan-Yorke (Lyapunov) dimension from a Lyapunov spectrum.

    D_KY = j + (λ_1 + ... + λ_j) / |λ_{j+1}|, where j is the largest index
    whose partial sum is non-negative. Formula unchanged from prior code.
    """
    spectrum = np.sort(np.asarray(spectrum, dtype=float))[::-1]
    cumsum = np.cumsum(spectrum)
    j_indices = np.where(cumsum >= 0)[0]
    if len(j_indices) == 0:
        return 0.0
    j = j_indices[-1]
    if j >= len(spectrum) - 1:
        return float(len(spectrum))
    if abs(spectrum[j + 1]) < 1e-10:
        return float(j + 1)
    return max(0.0, (j + 1) + cumsum[j] / abs(spectrum[j + 1]))
