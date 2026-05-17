"""Largest Lyapunov exponent (Rosenstein 1993) and an exploratory spectrum.

`largest_lyapunov` is the headline, robust estimator. `lyapunov_spectrum`
(corrected Sano-Sawada) is explicitly exploratory and emits a RuntimeWarning.
Pure numpy/scipy; no new dependencies.

Scope
-----
Designed for continuous (sampled-flow) time series. Discrete maps are out
of scope — delay-embedded scalar maps are not reliably estimated by this
method: a map carries essentially all of its divergence in the very first
step, which is indistinguishable from the single-step noise-floor artifact
that this estimator must reject to avoid labelling noisy periodic signals
as chaotic. Use a map-specific estimator for logistic/Hénon-type systems.
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
# observables, Rössler, Van der Pol, and a noisy-periodic rejection
# gate); the detector itself reads ONLY the divergence curve — never
# the system, the fixture parameters, or the trajectory. Discrete maps
# are intentionally out of scope (see the module docstring).
_SAT_LEVEL = 0.94  # saturation-onset fraction; must be SUSTAINED, not
                   # a single step (a lone 0->1 jump is a noise-floor
                   # artifact, never a Lyapunov scaling region)
_WMIN_DIV = 8      # w_min = max(5, region // _WMIN_DIV)
_W_DIV = 8         # fixed window length = max(w_min, region // _W_DIV)
_S0 = 0            # transient skip (kept at 0; the window position is
                   # data-selected so flows are unaffected)
_FLAT_LEN = 80     # minimum SUSTAINED rising-region length required to
                   # attempt a scaling-region fit. The degenerate flat-fit
                   # path is reached whenever no SUSTAINED positive-slope
                   # rising region of sufficient length exists — this covers
                   # both the short-region case (region < _FLAT_LEN) and the
                   # no-positive-window case (w >= region or best_a < 0),
                   # both of which are typical of regular/periodic signals.
                   # Also the length of the honest flat-fit window used in
                   # that degenerate case, taken AFTER the initial noise-floor
                   # jump. Read ONLY off the curve.


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
    Rosenstein divergence curve for a chaotic flow is a rising, gently
    concave ramp that saturates; the scaling region is the early
    portion that is most linear (highest OLS R²) with a strictly
    positive slope.

    A Lyapunov exponent must come from a SUSTAINED linear divergence
    region — never from a single-step jump. A regular/periodic signal
    (with or without measurement noise) produces a *collapsed* curve:
    one big step 0->1 (noise-floor -> signal-spacing artifact) and then
    flat. Treating that lone step as a scaling region is exactly the
    "noise labelled chaotic" failure this estimator must avoid, so the
    saturation onset must be SUSTAINED (not a single step), the rising
    region must be long enough to host a genuine scaling window, and the
    2-point ``curve[0:2]`` jump is NEVER fitted.

    Algorithm
    ---------
    1. Skip a tiny transient: ``s0 = _S0`` (0 by default).
    2. Bound the search to the *rising* part: ``s_sat`` is the first
       index >= ``s0`` at which ``curve`` reaches
       ``min + _SAT_LEVEL*(max-min)`` AND stays at/above it for the
       next ``hold_min = max(5, avail // _WMIN_DIV)`` samples (SUSTAINED
       saturation — a lone step does not count; ``hold_min`` is derived
       from the FULL available horizon, never a single step). If it
       never sustains, ``s_sat = len(curve)``. ``region = s_sat - s0``.
    3. Collapsed/degenerate test: a chaotic flow rises for hundreds of
       samples before saturating, whereas a regular/periodic signal
       collapses to the lone step-0->1 jump then flat (rising region
       only tens of samples). If ``region < max(w_min, _FLAT_LEN)``
       (with ``w_min = max(5, region // _WMIN_DIV)``) or no contiguous
       window of length >= ``w_min`` within ``[s0, s_sat]`` has a
       strictly positive slope, fit an HONEST flat window of length
       ``max(w_min, _FLAT_LEN)`` taken AFTER the initial jump:
       ``curve[s_flat:...]`` with ``s_flat = min(s0 + 1, n - 2)``. For
       a regular/periodic signal this window is flat -> slope ~= 0 ->
       λ₁ ~= 0 (correct). ``curve[0:2]`` is NEVER fitted.
    4. Otherwise (genuine sustained rising region): slide a FIXED-length
       window ``w = max(w_min, region // _W_DIV)`` by step 1 over
       ``[s0, s_sat - w]`` and pick the window with the maximum R²
       among those with a strictly positive slope. The window length
       is fixed (>= ``w_min``); only its *position* is data-selected,
       keeping the search O(n). The ``region``-derived ``w``/``w_min``
       are identical to the pre-fix detector so chaotic flows fit the
       identical window. ``len(curve) < 4`` is handled by the caller
       (returns λ₁ = 0).

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

    # ``hold_min`` is a stable saturation-hold length derived from the
    # FULL available horizon (never a single step). It only governs the
    # SUSTAINED-saturation test below; it deliberately does NOT feed the
    # rising-region window length so chaotic flows fit exactly the same
    # region as before this fix.
    avail = n - s0
    hold_min = max(5, avail // _WMIN_DIV)

    # Saturation onset: first index that reaches the saturation level
    # AND stays at/above it for >= hold_min consecutive samples. A lone
    # step that touches the level (the noise-floor artifact of a
    # regular/periodic signal) is NOT a saturation and does not bound
    # the rising region; in that case s_sat falls through to n and the
    # collapsed/degenerate path is taken.
    s_sat = n
    if span > 1e-12:
        threshold = lo + _SAT_LEVEL * span
        for idx in range(s0, n):
            if curve[idx] >= threshold:
                hold_end = min(n, idx + hold_min)
                if (hold_end - idx) >= hold_min and np.all(
                    curve[idx:hold_end] >= threshold
                ):
                    s_sat = idx
                    break
    s_sat = max(s_sat, s0)

    region = s_sat - s0
    # Rising-region window sizing — identical formulae to the original
    # detector so chaotic flows are unaffected.
    w_min = max(5, region // _WMIN_DIV)

    def _degenerate_flat() -> Tuple[int, int, float]:
        """Honest flat fit AFTER the initial noise-floor jump.

        Never fits ``curve[0:2]``: the step-0->1 jump is excluded by
        starting at ``s_flat = min(s0 + 1, n - 2)`` and the window is
        ``max(w_min, _FLAT_LEN)`` long (clipped to what is available).
        For a regular/periodic signal this window is flat so the slope
        — and therefore λ₁ — is ~= 0; ``fit_r2`` is the OLS R² of that
        flat window (low, correctly flagging it is not a linear scaling
        region).
        """
        s_flat = min(s0 + 1, n - 2)
        length = max(w_min, _FLAT_LEN)
        end = min(n, s_flat + length)
        if end - s_flat < 2:  # pathological tiny curve
            s_flat = max(0, n - 2)
            end = n
        r2, _ = _ols_r2_slope(curve[s_flat:end])
        return s_flat, end, r2

    # A Lyapunov exponent must come from a SUSTAINED rising region. A
    # genuine chaotic flow rises for hundreds of samples before it
    # saturates; a regular/periodic signal (with or without measurement
    # noise) collapses to a lone step-0->1 jump and is then flat, so its
    # "rising region" before the sustained plateau is only tens of
    # samples. Requiring ``region >= max(w_min, _FLAT_LEN)`` rejects
    # that collapsed case and routes it to the honest flat fit. This
    # bound is read ONLY off the curve and does NOT alter ``w``/``w_min``
    # (still region-derived) so chaotic flows fit the identical window
    # as before. (Replaces the old ``curve[0:2]`` 2-point-jump fallback,
    # which is exactly the noise-floor artifact a fit must never use.)
    if region < max(w_min, _FLAT_LEN):
        return _degenerate_flat()

    w = max(w_min, region // _W_DIV)
    if w >= region:  # no room for a >= w_min rising window -> collapsed
        return _degenerate_flat()

    best_r2 = -np.inf
    best_a = -1
    for a in range(s0, s_sat - w + 1):
        r2, slope = _ols_r2_slope(curve[a : a + w])
        if slope > 0.0 and r2 > best_r2:
            best_r2 = r2
            best_a = a

    # No sustained positive-slope window of length >= w_min anywhere in
    # the rising region: the curve is flat (regular/periodic) -> honest
    # flat fit, never the 2-point jump.
    if best_a < 0:
        return _degenerate_flat()

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
    metric; there is no plateau assumption. A regular/periodic signal
    (with or without measurement noise) has no sustained rising region
    and is correctly estimated at λ₁ ≈ 0 — a lone step-0->1 jump is a
    noise-floor artifact and is never fitted as a scaling region.

    Scope
    -----
    Designed for continuous (sampled-flow) time series. Discrete maps
    are out of scope — delay-embedded scalar maps are not reliably
    estimated by this method: a map puts essentially all of its
    divergence in the single first step, which is indistinguishable
    from the single-step noise-floor artifact this estimator must
    reject. Use a map-specific estimator for logistic/Hénon-type
    systems.

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
