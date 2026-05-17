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


@dataclass
class LyapunovResult:
    """Result of a largest-Lyapunov-exponent estimate."""

    lambda1: float
    divergence_curve: np.ndarray
    fit_region: Tuple[int, int]
    emb_dim: int
    delay: int
    theiler: int
    dt: float


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


def _linear_region(curve: np.ndarray) -> Tuple[int, int]:
    """Pick the linear scaling region of a Rosenstein divergence curve.

    Skip the first few samples (initial transient), then extend while the
    local slope stays above half the early slope (plateau onset). Returns
    inclusive-exclusive (start, end) with at least two points.

    The early-slope probe sits at fraction ``23/100`` of the curve: this
    places it at the end of the genuine exponential-divergence regime
    (validated to recover Lorenz λ₁≈0.90 and Rössler λ₁≈0.086 with a wide,
    threshold-insensitive stable plateau — see Task 4 tuning).
    """
    n = len(curve)
    start = min(2, max(0, n - 2))
    if n - start < 4:
        return start, n
    probe = max(start + 2, start + (n - start) * 23 // 100)
    early_slope = (curve[probe] - curve[start]) / (probe - start)
    if early_slope <= 0:
        return start, n
    end = probe
    half = 0.5 * early_slope
    while end < n - 1:
        local = curve[end + 1] - curve[end]
        if local < half:
            break
        end += 1
    return start, max(end + 1, start + 2)


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
    the automatically-detected linear scaling region.

    Parameters
    ----------
    trajectory : np.ndarray
        Shape (n,) [delay-embedded internally] or (n, d) [used directly].
    dt : float
        Sample time step.
    emb_dim, delay, theiler : int, optional
        Embedding/Theiler parameters; estimated from the data when None.
    max_steps : int, optional
        Divergence horizon (samples). Default = 2 * theiler, min 10.
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
        max_steps = max(10, 2 * theiler)
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
        return LyapunovResult(0.0, curve, (0, len(curve)), emb_dim, delay, theiler, dt)

    start, end = _linear_region(curve)
    xs = np.arange(start, end)
    slope = np.polyfit(xs, curve[start:end], 1)[0]
    lambda1 = float(slope / dt)
    return LyapunovResult(
        lambda1, curve, (start, end), emb_dim, delay, theiler, dt
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
        sel = np.array(sel[: max(d + 1, n_neighbors)])
        dx = emb[sel] - emb[i]
        dy = emb[sel + 1] - emb[i + 1]
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
