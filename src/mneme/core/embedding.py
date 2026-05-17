"""Phase-space embedding and parameter selection.

Pure-numpy implementations of delay embedding, the Theiler window,
Fraser–Swinney mutual-information delay selection, and Cao's (1997)
minimum embedding dimension. These feed the Lyapunov estimators.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.spatial import cKDTree


def embed_trajectory(
    time_series: np.ndarray,
    embedding_dimension: int,
    time_delay: int,
) -> np.ndarray:
    """Create a delay embedding of a time series.

    Parameters
    ----------
    time_series : np.ndarray
        Shape (n,) or (n, n_features).
    embedding_dimension : int
        Number of delayed copies.
    time_delay : int
        Delay (in samples) between copies.

    Returns
    -------
    np.ndarray
        Embedded trajectory, shape
        (n - (embedding_dimension-1)*time_delay, embedding_dimension*n_features).
    """
    ts = np.asarray(time_series, dtype=float)
    if ts.ndim == 1:
        ts = ts.reshape(-1, 1)

    n_points = len(ts) - (embedding_dimension - 1) * time_delay
    if n_points <= 0:
        raise ValueError("Time series too short for embedding")

    n_feat = ts.shape[1]
    embedded = np.zeros((n_points, embedding_dimension * n_feat))
    for i in range(embedding_dimension):
        start = i * time_delay
        embedded[:, i * n_feat : (i + 1) * n_feat] = ts[start : start + n_points]
    return embedded


def theiler_window(time_series: np.ndarray) -> int:
    """Theiler window = first zero crossing of the autocorrelation function.

    Used to exclude temporally-correlated neighbours from divergence /
    Jacobian estimates. Falls back to the 1/e decay time, then to 1.
    """
    x = np.asarray(time_series, dtype=float)
    if x.ndim > 1:
        x = x[:, 0]
    x = x - x.mean()
    n = len(x)
    if n < 4:
        return 1
    ac = np.correlate(x, x, mode="full")[n - 1 :]
    if ac[0] == 0:
        return 1
    ac = ac / ac[0]
    for lag in range(1, len(ac)):
        if ac[lag] <= 0.0:
            return max(1, lag)
    for lag in range(1, len(ac)):
        if ac[lag] <= np.exp(-1.0):
            return max(1, lag)
    return 1


def _mutual_information(x: np.ndarray, y: np.ndarray, n_bins: int) -> float:
    """Histogram-based mutual information of two equal-length series (nats)."""
    c_xy, _, _ = np.histogram2d(x, y, bins=n_bins)
    p_xy = c_xy / c_xy.sum()
    p_x = p_xy.sum(axis=1)
    p_y = p_xy.sum(axis=0)
    nz = p_xy > 0
    outer = p_x[:, None] * p_y[None, :]
    return float(np.sum(p_xy[nz] * np.log(p_xy[nz] / outer[nz])))


def mutual_information_delay(time_series: np.ndarray, max_delay: int = 100) -> int:
    """Time delay = first local minimum of time-delayed mutual information.

    Fraser–Swinney method. Bin count via a Freedman–Diaconis-like rule.
    If no local minimum is found, returns the delay of the global minimum
    within [1, max_delay]; if MI is monotone/degenerate, returns 1.
    """
    x = np.asarray(time_series, dtype=float)
    if x.ndim > 1:
        x = x[:, 0]
    n = len(x)
    upper = max(2, min(max_delay, n // 5))
    n_bins = max(8, int(np.sqrt(n / 5.0)))

    mis = []
    for d in range(1, upper):
        mis.append(_mutual_information(x[:-d], x[d:], n_bins))
    mis = np.asarray(mis)
    if len(mis) < 3:
        return 1
    for i in range(1, len(mis) - 1):
        if mis[i] < mis[i - 1] and mis[i] <= mis[i + 1]:
            return i + 1
    return int(np.argmin(mis)) + 1


def cao_embedding_dimension(
    time_series: np.ndarray,
    delay: int,
    max_dim: int = 10,
) -> int:
    """Minimum embedding dimension via Cao's (1997) E1 statistic.

    E1(d) saturates near 1 once the attractor is unfolded. Returns the
    smallest d where the relative change in E1 drops below 5% (or max_dim).
    """
    x = np.asarray(time_series, dtype=float)
    if x.ndim > 1:
        x = x[:, 0]

    e_values = []
    for d in range(1, max_dim + 2):
        try:
            emb = embed_trajectory(x, d + 1, delay)
        except ValueError:
            break
        m = len(emb)
        if m < 10:
            break
        emb_d = emb[:, :d]
        tree = cKDTree(emb_d)
        dist, idx = tree.query(emb_d, k=2)
        nn = idx[:, 1]
        denom = dist[:, 1]
        full = np.linalg.norm(emb - emb[nn], axis=1)
        good = denom > 1e-12
        if not np.any(good):
            break
        e_values.append(float(np.mean(full[good] / denom[good])))

    if len(e_values) < 2:
        return min(3, max_dim)
    e = np.asarray(e_values)
    e1 = e[1:] / e[:-1]
    for d in range(1, len(e1)):
        if abs(e1[d] - e1[d - 1]) < 0.05:
            return d + 1
    return min(len(e1) + 1, max_dim)


def estimate_embedding_parameters(
    time_series: np.ndarray,
    max_dimension: int = 10,
    max_delay: int = 100,
) -> Tuple[int, int]:
    """Estimate (embedding_dimension, time_delay) via MI delay + Cao dimension."""
    x = np.asarray(time_series, dtype=float)
    if x.ndim > 1:
        x = x.flatten()
    delay = mutual_information_delay(x, max_delay)
    dim = cao_embedding_dimension(x, delay, max_dimension)
    return int(dim), int(delay)
