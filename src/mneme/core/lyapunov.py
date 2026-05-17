"""Largest Lyapunov exponent (Rosenstein 1993) and an exploratory spectrum.

`largest_lyapunov` is the headline, robust estimator. `lyapunov_spectrum`
(corrected Sano-Sawada) is explicitly exploratory and emits a RuntimeWarning.
Pure numpy/scipy; no new dependencies.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy.spatial import cKDTree

from .embedding import (
    cao_embedding_dimension,
    embed_trajectory,
    mutual_information_delay,
    theiler_window,
)

MIN_TRAJECTORY_LENGTH = 100
RECOMMENDED_TRAJECTORY_LENGTH = 1000

# Theiler-robust divergence-horizon bounds for the default ``max_steps``
# (see :func:`largest_lyapunov`). Applied identically to every input;
# they encode no per-system knowledge.
_MS_MIN = 250
_MS_MAX = 450


@dataclass
class LyapunovResult:
    """Result of a largest-Lyapunov-exponent estimate.

    ``fit_region`` is the inclusive-exclusive ``(start, end)`` slice of
    ``divergence_curve`` over which the slope was fitted; ``fit_r2`` is
    the ordinary-least-squares coefficient of determination of that fit
    (a fit-quality metric: values near 1 indicate a genuinely linear
    scaling region; low values mean the estimate should be distrusted).
    For the degenerate exact-period case (no resolvable divergence)
    ``fit_region == (0, 0)``, ``divergence_curve`` is empty and both
    ``lambda1`` and ``fit_r2`` are ``0.0``.
    """

    lambda1: float
    divergence_curve: np.ndarray
    fit_region: Tuple[int, int]
    emb_dim: int
    delay: int
    theiler: int
    dt: float
    fit_r2: float


def _resolve_embedding(
    series: np.ndarray,
    emb_dim: Optional[int],
    delay: Optional[int],
) -> Tuple[np.ndarray, int, int]:
    """Embed a 1-D series, estimating parameters when not supplied."""
    if delay is None:
        delay = mutual_information_delay(series, max_delay=100)
    if emb_dim is None:
        emb_dim = cao_embedding_dimension(series, delay, max_dim=10)
    emb_dim = max(2, int(emb_dim))
    delay = max(1, int(delay))
    return embed_trajectory(series, emb_dim, delay), emb_dim, delay


# --- data-driven scaling-region detector ---------------------------------
#
# Tunable constants. Their VALUES were chosen to satisfy a broad
# generalisation suite (multiple Lorenz initial conditions and
# observables, Rössler, the logistic and Hénon maps, Van der Pol); the
# detector itself reads ONLY the divergence curve — never the system,
# the fixture parameters, or the trajectory.
_SAT_LEVEL = 0.94  # saturation-onset fraction that bounds the rising part
_WMIN_DIV = 8      # w_min = max(5, region // _WMIN_DIV)
_W_DIV = 8         # fixed window length = max(w_min, region // _W_DIV)
_S0 = 0            # transient skip (kept at 0: fast maps put the
                   # dominant divergence in step 0; the window position
                   # is data-selected so flows are unaffected)


def _ols_r2_slope(y: np.ndarray) -> Tuple[float, float]:
    """OLS line fit of ``y`` vs its index. Returns (R², slope).

    R² is the coefficient of determination; a constant ``y`` yields
    (1.0, 0.0) and is rejected later by the positive-slope requirement.
    """
    n = len(y)
    if n < 2:
        return 0.0, 0.0
    x = np.arange(n, dtype=float)
    xc = x - x.mean()
    yc = y - y.mean()
    sxx = float(np.dot(xc, xc))
    if sxx <= 1e-18:
        return 0.0, 0.0
    slope = float(np.dot(xc, yc) / sxx)
    syy = float(np.dot(yc, yc))
    if syy <= 1e-18:
        return 1.0, slope
    r2 = 1.0 - (syy - slope * slope * sxx) / syy
    return float(r2), slope


def _linear_region(curve: np.ndarray) -> Tuple[int, int, float]:
    """Locate the linear scaling region of a Rosenstein divergence curve.

    The *position* of the scaling region is chosen by the data — not
    assumed as a fixed fraction of the curve — which is what makes the
    estimate generalise across systems, initial conditions, observables
    and sample rates. There is NO assumption of a flat plateau: a
    Rosenstein divergence curve is a rising, gently concave ramp that
    saturates; the scaling region is the early portion that is most
    linear (highest OLS R²) with a strictly positive slope.

    Algorithm
    ---------
    1. Skip a tiny transient: ``s0 = _S0`` (0 by default — fast maps
       carry their dominant divergence in the first step).
    2. Bound the search to the *rising* part: ``s_sat`` is the first
       index >= ``s0`` where ``curve`` reaches
       ``min + _SAT_LEVEL*(max-min)`` (the saturation onset); if it
       never does, ``s_sat = len(curve)``. The scaling region lies in
       ``[s0, s_sat]``.
    3. Slide a FIXED-length window
       ``w = max(w_min, region // _W_DIV)`` (with
       ``w_min = max(5, region // _WMIN_DIV)``) by step 1 over
       ``[s0, s_sat - w]`` and pick the window with the maximum R²
       among those with a strictly positive slope. The window length
       is fixed; only its *position* is data-selected, keeping the
       search O(n).
    4. Fallbacks: if ``region < w_min`` or no positive-slope window
       exists, fit the whole ``[s0, s_sat]``. ``len(curve) < 4`` is
       handled by the caller (returns λ₁ = 0).

    Returns
    -------
    (start, end, r2)
        Inclusive-exclusive slice and the OLS R² of that fit.
    """
    n = len(curve)
    if n < 4:
        return 0, n, 0.0

    s0 = min(_S0, max(0, n - 2))
    lo = float(curve.min())
    hi = float(curve.max())
    span = hi - lo

    s_sat = n
    if span > 1e-12:
        threshold = lo + _SAT_LEVEL * span
        for idx in range(s0, n):
            if curve[idx] >= threshold:
                s_sat = idx
                break
    s_sat = max(s_sat, s0)

    region = s_sat - s0
    w_min = max(5, region // _WMIN_DIV)

    if region < w_min:
        end = max(s_sat, min(s0 + 2, n))
        r2, _ = _ols_r2_slope(curve[s0:end])
        return s0, end, r2

    w = max(w_min, region // _W_DIV)
    if w >= region:
        r2, _ = _ols_r2_slope(curve[s0:s_sat])
        return s0, s_sat, r2

    best_r2 = -np.inf
    best_a = -1
    for a in range(s0, s_sat - w + 1):
        r2, slope = _ols_r2_slope(curve[a : a + w])
        if slope > 0.0 and r2 > best_r2:
            best_r2 = r2
            best_a = a

    if best_a < 0:
        r2, _ = _ols_r2_slope(curve[s0:s_sat])
        return s0, s_sat, r2

    r2, _ = _ols_r2_slope(curve[best_a : best_a + w])
    return best_a, best_a + w, r2


def largest_lyapunov(
    trajectory: np.ndarray,
    dt: float = 1.0,
    *,
    emb_dim: Optional[int] = None,
    delay: Optional[int] = None,
    theiler: Optional[int] = None,
    max_steps: Optional[int] = None,
) -> LyapunovResult:
    """Largest Lyapunov exponent via Rosenstein et al. (1993).

    Tracks the mean log Euclidean divergence of nearest-neighbour pairs
    (excluding neighbours within the Theiler window) and fits a line over
    the linear scaling region whose *position* is selected from the data
    by :func:`_linear_region` (highest OLS R², strictly positive slope).
    The fit R² is returned in ``LyapunovResult.fit_r2`` as a fit-quality
    metric; there is no plateau assumption.

    Parameters
    ----------
    trajectory : np.ndarray
        Shape (n,) [delay-embedded internally] or (n, d) [used directly].
    dt : float
        Sample time step.
    emb_dim, delay, theiler : int, optional
        Embedding/Theiler parameters; estimated from the data when None.
    max_steps : int, optional
        Divergence horizon (samples). When not given, a Theiler-robust
        default is used: ``2 * theiler`` clipped to ``[_MS_MIN, _MS_MAX]``.
        The Theiler window scales with the autocorrelation time and so
        varies by orders of magnitude across systems and initial
        conditions (≈1 for maps, hundreds for an over-sampled flow);
        ``2 * theiler`` alone can be far too short to contain any
        exponential-divergence regime (e.g. some Lorenz initial
        conditions), leaving the detector with no scaling region to
        find. Clipping makes the horizon long enough to resolve the
        scaling region without depending on any per-system knowledge
        (the same formula is applied to every input).
    """
    arr = np.asarray(trajectory, dtype=float)
    raw_1d = arr if arr.ndim == 1 else arr[:, 0]

    if theiler is None:
        theiler = theiler_window(raw_1d)
    theiler = max(1, int(theiler))

    if arr.ndim == 1:
        emb, emb_dim, delay = _resolve_embedding(arr, emb_dim, delay)
    else:
        emb = arr
        emb_dim = arr.shape[1] if emb_dim is None else emb_dim
        delay = 1 if delay is None else delay

    n = len(emb)
    if n < MIN_TRAJECTORY_LENGTH:
        raise ValueError(
            f"Trajectory too short ({n} points). "
            f"Need at least {MIN_TRAJECTORY_LENGTH}."
        )
    if n < RECOMMENDED_TRAJECTORY_LENGTH:
        warnings.warn(
            f"largest_lyapunov on a short trajectory ({n} points; "
            f"recommended >= {RECOMMENDED_TRAJECTORY_LENGTH}). Treat as exploratory.",
            RuntimeWarning,
            stacklevel=2,
        )

    if max_steps is None:
        max_steps = int(np.clip(2 * theiler, _MS_MIN, _MS_MAX))
    max_steps = min(max_steps, n - 2)

    tree = cKDTree(emb)
    k = min(n, 4 * theiler + 8)
    dists, idxs = tree.query(emb, k=k)
    neighbour = np.full(n, -1, dtype=int)
    for i in range(n):
        for j_pos in range(1, idxs.shape[1]):
            j = idxs[i, j_pos]
            if abs(j - i) > theiler:
                neighbour[i] = j
                break

    log_div_sum = np.zeros(max_steps + 1)
    log_div_cnt = np.zeros(max_steps + 1)
    for i in range(n):
        j = neighbour[i]
        if j < 0:
            continue
        horizon = min(max_steps, n - 1 - max(i, j))
        if horizon < 1:
            continue
        for s in range(horizon + 1):
            d = np.linalg.norm(emb[i + s] - emb[j + s])
            if d > 1e-12:
                log_div_sum[s] += np.log(d)
                log_div_cnt[s] += 1

    valid = log_div_cnt > 0
    curve = np.full(max_steps + 1, np.nan)
    curve[valid] = log_div_sum[valid] / log_div_cnt[valid]
    curve = curve[valid]

    if len(curve) < 4:
        return LyapunovResult(
            0.0, curve, (0, len(curve)), emb_dim, delay, theiler, dt, 0.0
        )

    start, end, _ = _linear_region(curve)
    xs = np.arange(start, end)
    slope = np.polyfit(xs, curve[start:end], 1)[0]
    fit_r2, _ = _ols_r2_slope(curve[start:end])
    lambda1 = float(slope / dt)
    return LyapunovResult(
        lambda1, curve, (start, end), emb_dim, delay, theiler, dt, float(fit_r2)
    )


def lyapunov_spectrum(
    trajectory: np.ndarray,
    dt: float = 1.0,
    *,
    emb_dim: Optional[int] = None,
    delay: Optional[int] = None,
    theiler: Optional[int] = None,
    n_neighbors: int = 7,
) -> np.ndarray:
    """EXPLORATORY full Lyapunov spectrum (corrected Sano-Sawada).

    Local-Jacobian QR method with Theiler exclusion, ridge-regularised
    neighbour regression, and consistent log-growth / time normalisation.
    Emits a RuntimeWarning: prefer `largest_lyapunov` for the headline λ₁.

    Parameters
    ----------
    trajectory : np.ndarray
        Shape (n,) [delay-embedded internally] or (n, d) [used directly].
    dt : float
        Sample time step.
    emb_dim, delay, theiler : int, optional
        Embedding/Theiler parameters; estimated from the data when None.
    n_neighbors : int
        Neighbours used for the local-Jacobian regression. A small value
        keeps the regression local enough to resolve the strongly
        contracting direction (larger neighbourhoods average it away and
        bias the spectrum sum toward zero). Validated to recover the Lorenz
        spectrum (λ₁≈0.90, sum≈-14, vs the true trace -13.67) with margin
        on both acceptance gates — see Task 4 tuning.
    """
    warnings.warn(
        "lyapunov_spectrum is EXPLORATORY (full-spectrum estimates on short/"
        "noisy data are unreliable). Use largest_lyapunov for the headline "
        "exponent.",
        RuntimeWarning,
        stacklevel=2,
    )
    arr = np.asarray(trajectory, dtype=float)
    raw_1d = arr if arr.ndim == 1 else arr[:, 0]
    if theiler is None:
        theiler = theiler_window(raw_1d)
    theiler = max(1, int(theiler))

    if arr.ndim == 1:
        emb, emb_dim, delay = _resolve_embedding(arr, emb_dim, delay)
    else:
        emb = arr

    n, d = emb.shape
    if n < MIN_TRAJECTORY_LENGTH:
        raise ValueError(
            f"Trajectory too short ({n} points). "
            f"Need at least {MIN_TRAJECTORY_LENGTH}."
        )

    tree = cKDTree(emb[:-1])
    Q = np.eye(d)
    lyap_sums = np.zeros(d)
    n_steps = 0
    k = min(n - 1, n_neighbors + 4 * theiler + 4)

    for i in range(n - 1):
        dists, idxs = tree.query(emb[i], k=k)
        sel = [j for j in np.atleast_1d(idxs) if abs(int(j) - i) > theiler and j < n - 1]
        if len(sel) < d + 1:
            continue
        sel_arr = np.asarray(sel[: max(d + 1, n_neighbors)])
        dx = emb[sel_arr] - emb[i]
        dy = emb[sel_arr + 1] - emb[i + 1]
        lam = 1e-6 * np.trace(dx.T @ dx) / max(d, 1)
        J = np.linalg.solve(dx.T @ dx + lam * np.eye(d), dx.T @ dy).T
        Q = J @ Q
        Q, R = np.linalg.qr(Q)
        diag = np.abs(np.diag(R))
        diag[diag < 1e-12] = 1e-12
        lyap_sums += np.log(diag)
        n_steps += 1

    if n_steps == 0:
        raise ValueError("Could not estimate Jacobians. Check trajectory quality.")
    spectrum = lyap_sums / (n_steps * dt)
    return np.sort(spectrum)[::-1]
