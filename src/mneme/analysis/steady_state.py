"""Steady-state and multistability analysis of relaxing systems.

Tools for the question "does this system have more than one stable state?"
given several runs that each relax toward rest:

1. :func:`convergence_rate` and :func:`assess_steady_state` decide whether a
   run has settled, and estimate how far it still has to go.
2. :func:`count_distinct_states` groups the end states of several runs and
   counts how many are distinct, against a stated threshold.

All functions take a trajectory of shape ``(n_times, n_cells)``: one value
per cell per sample, with no spatial interpolation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

__all__ = [
    "SteadyStateReport",
    "DistinctStates",
    "convergence_rate",
    "assess_steady_state",
    "pairwise_max_difference",
    "count_distinct_states",
]


@dataclass
class SteadyStateReport:
    """Whether a run settled, and how far it was from settling.

    Attributes
    ----------
    settled : bool
        True when the rate stayed below ``rate_tolerance`` for the final
        ``sustain`` samples AND the estimated remaining drift is below
        ``drift_tolerance``.
    final_rate : float
        Largest per-cell rate of change at the final sample, in value
        units per unit time.
    time_constant : float
        Relaxation time fitted to the decay of the rate over the final
        part of the run. ``inf`` when the rate is not decaying, ``nan``
        when it could not be fitted.
    remaining_drift : float
        Estimated distance still to travel, ``final_rate * time_constant``.
        This is what an exponential relaxation has left to go.
    rate : np.ndarray
        Largest per-cell rate of change at each interval, shape
        ``(n_times - 1,)``.
    """

    settled: bool
    final_rate: float
    time_constant: float
    remaining_drift: float
    rate: np.ndarray
    rate_tolerance: float
    drift_tolerance: float
    sustain: int


@dataclass
class DistinctStates:
    """Grouping of end states.

    Attributes
    ----------
    n_states : int
        Number of distinct states.
    labels : np.ndarray
        State index of each run, shape ``(n_runs,)``. States are numbered
        in order of first appearance.
    distances : np.ndarray
        Largest per-cell difference between each pair of end states,
        shape ``(n_runs, n_runs)``.
    threshold : float
        Two end states closer than this are the same state.
    largest_within : float
        Largest distance between two runs placed in the same state.
        0 when every state has one run.
    smallest_between : float
        Smallest distance between two runs placed in different states.
        ``inf`` when there is one state.
    """

    n_states: int
    labels: np.ndarray
    distances: np.ndarray
    threshold: float
    largest_within: float
    smallest_between: float


def _as_trajectory(values: np.ndarray, times: Optional[np.ndarray]) -> tuple:
    values = np.asarray(values, dtype=float)
    if values.ndim != 2:
        raise ValueError(
            f"trajectory must have shape (n_times, n_cells); got {values.shape}"
        )
    if values.shape[0] < 3:
        raise ValueError("trajectory needs at least 3 samples")
    if not np.all(np.isfinite(values)):
        raise ValueError("trajectory contains NaN or infinite values")
    if times is None:
        times = np.arange(values.shape[0], dtype=float)
    times = np.asarray(times, dtype=float)
    if times.shape != (values.shape[0],):
        raise ValueError("times must have one entry per sample")
    if np.any(np.diff(times) <= 0):
        raise ValueError("times must be strictly increasing")
    return values, times


def convergence_rate(
    values: np.ndarray, times: Optional[np.ndarray] = None
) -> np.ndarray:
    """Largest per-cell rate of change over each sampling interval.

    Parameters
    ----------
    values : np.ndarray
        Shape ``(n_times, n_cells)``.
    times : np.ndarray, optional
        Sample times, shape ``(n_times,)``. Defaults to 0, 1, 2, ...

    Returns
    -------
    np.ndarray
        Shape ``(n_times - 1,)``, in value units per unit time.
    """
    values, times = _as_trajectory(values, times)
    step = np.max(np.abs(np.diff(values, axis=0)), axis=1)
    return step / np.diff(times)


def assess_steady_state(
    values: np.ndarray,
    times: Optional[np.ndarray] = None,
    *,
    rate_tolerance: float,
    drift_tolerance: float,
    sustain: int = 20,
    fit_fraction: float = 0.5,
) -> SteadyStateReport:
    """Decide whether a relaxing run has settled.

    A low rate of change is not enough: a slow relaxation can have a small
    rate and still be far from its end state. The remaining distance is
    estimated by fitting an exponential decay to the rate over the final
    ``fit_fraction`` of the run and taking ``final_rate * time_constant``.

    Parameters
    ----------
    values, times
        As for :func:`convergence_rate`.
    rate_tolerance : float
        The rate must stay below this for the final ``sustain`` samples.
    drift_tolerance : float
        The estimated remaining drift must be below this.
    sustain : int
        Number of final intervals over which the rate is checked.
    fit_fraction : float
        Fraction of the run, at its end, used to fit the decay.
    """
    values, times = _as_trajectory(values, times)
    if not 0.0 < fit_fraction <= 1.0:
        raise ValueError("fit_fraction must be in (0, 1]")
    rate = convergence_rate(values, times)
    sustain = int(min(max(1, sustain), len(rate)))
    final_rate = float(rate[-1])

    # Fit log(rate) = a - t / tau over the tail, where the rate is resolved.
    mid = 0.5 * (times[1:] + times[:-1])
    start = int(np.floor(len(rate) * (1.0 - fit_fraction)))
    tail_t, tail_r = mid[start:], rate[start:]
    resolved = tail_r > 0
    if resolved.sum() >= 3:
        slope = float(np.polyfit(tail_t[resolved], np.log(tail_r[resolved]), 1)[0])
        time_constant = -1.0 / slope if slope < 0 else float("inf")
    elif np.all(tail_r == 0):
        time_constant = 0.0  # no movement at all over the tail
    else:
        time_constant = float("nan")

    if final_rate == 0.0:
        remaining = 0.0
    else:
        remaining = final_rate * time_constant  # inf or nan propagate

    settled = bool(
        np.all(rate[-sustain:] < rate_tolerance)
        and np.isfinite(remaining)
        and remaining < drift_tolerance
    )
    return SteadyStateReport(
        settled=settled,
        final_rate=final_rate,
        time_constant=float(time_constant),
        remaining_drift=float(remaining),
        rate=rate,
        rate_tolerance=float(rate_tolerance),
        drift_tolerance=float(drift_tolerance),
        sustain=sustain,
    )


def pairwise_max_difference(states: Sequence[np.ndarray]) -> np.ndarray:
    """Largest per-cell difference between each pair of states.

    Parameters
    ----------
    states : sequence of np.ndarray
        Each of shape ``(n_cells,)``, all over the same cells in the same
        order.

    Returns
    -------
    np.ndarray
        Symmetric matrix of shape ``(n_states, n_states)``.
    """
    stacked = np.asarray(states, dtype=float)
    if stacked.ndim != 2:
        raise ValueError("states must all have the same length")
    if not np.all(np.isfinite(stacked)):
        raise ValueError("states contain NaN or infinite values")
    return np.max(np.abs(stacked[:, None, :] - stacked[None, :, :]), axis=2)


def count_distinct_states(
    states: Sequence[np.ndarray], threshold: float
) -> DistinctStates:
    """Count the distinct states among several end states.

    Two states are linked when their largest per-cell difference is below
    ``threshold``; a distinct state is a group of linked states
    (single-linkage). ``largest_within`` and ``smallest_between`` show how
    clean the grouping is: a trustworthy count has
    ``largest_within < threshold < smallest_between`` with room on both
    sides.

    Parameters
    ----------
    states : sequence of np.ndarray
        End states, each of shape ``(n_cells,)``.
    threshold : float
        Set this from replicate noise, not from the states being grouped.
    """
    if threshold <= 0:
        raise ValueError("threshold must be positive")
    distances = pairwise_max_difference(states)
    n = distances.shape[0]
    labels = np.full(n, -1, dtype=int)
    next_label = 0
    for i in range(n):
        if labels[i] >= 0:
            continue
        labels[i] = next_label
        frontier: List[int] = [i]
        while frontier:
            j = frontier.pop()
            for k in np.flatnonzero((distances[j] < threshold) & (labels < 0)):
                labels[k] = next_label
                frontier.append(int(k))
        next_label += 1

    same = labels[:, None] == labels[None, :]
    off_diagonal = ~np.eye(n, dtype=bool)
    within = distances[same & off_diagonal]
    between = distances[~same]
    return DistinctStates(
        n_states=next_label,
        labels=labels,
        distances=distances,
        threshold=float(threshold),
        largest_within=float(within.max()) if within.size else 0.0,
        smallest_between=float(between.min()) if between.size else float("inf"),
    )
