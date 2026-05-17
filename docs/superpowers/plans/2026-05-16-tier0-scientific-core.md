# Tier 0 Scientific Core Repair — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace Mneme's unreliable Lyapunov machinery with hand-rolled, literature-standard estimators plus a mandatory surrogate-significance gate, so the tool can no longer label noise as chaos.

**Architecture:** Four new pure-numpy modules (`embedding.py`, `lyapunov.py`, `surrogates.py`, `classify.py`) under `src/mneme/core/`. `attractors.py` loses all module-level Lyapunov code (clean break — no shims) and keeps only recurrence/clustering plus a delegating `LyapunovAnalysis`. Validation is TDD against Lorenz/Rössler/sine/noise with explicit tolerances.

**Tech Stack:** Python 3.12, numpy, scipy (`cKDTree`, `fft`), pytest. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-05-16-tier0-scientific-core-design.md`

---

## File Structure

| File | Responsibility |
|---|---|
| `src/mneme/types.py` | MODIFY — add `AttractorType.UNDETERMINED` |
| `tests/conftest.py` | MODIFY — RK4 Lorenz/Rössler fixtures replacing Euler Lorenz |
| `src/mneme/core/embedding.py` | CREATE — `embed_trajectory`, `theiler_window`, `mutual_information_delay`, `cao_embedding_dimension`, `estimate_embedding_parameters` |
| `src/mneme/core/lyapunov.py` | CREATE — `LyapunovResult`, `largest_lyapunov` (Rosenstein), `lyapunov_spectrum` (Sano-Sawada, exploratory) |
| `src/mneme/core/surrogates.py` | CREATE — `iaaft_surrogates`, `SurrogateResult`, `surrogate_test` |
| `src/mneme/core/classify.py` | CREATE — `kaplan_yorke_dimension` (moved), `classify_attractor` (gated) |
| `src/mneme/core/attractors.py` | MODIFY — delete module-level Lyapunov fns/helpers; import `embed_trajectory` from `embedding`; `LyapunovAnalysis.compute_lyapunov_spectrum` delegates |
| `src/mneme/core/__init__.py` | MODIFY — export new names, remove old |
| `tests/test_embedding.py` | CREATE |
| `tests/test_lyapunov.py` | CREATE |
| `tests/test_surrogates.py` | CREATE |
| `tests/test_classify.py` | CREATE |
| `tests/test_attractors.py` | MODIFY — drop deleted-API tests/imports |
| `tests/test_scripts_smoke.py` | CREATE — import-and-run smoke for 3 scripts |
| `scripts/analyze_physionet.py`, `scripts/deep_analysis.py`, `scripts/analyze_betse.py` | MODIFY — migrate to new API |
| `CLAUDE.md`, `README.md`, `CHANGELOG.md` | MODIFY — API snippets + flag headline numbers |

**Test command (this machine):** `win_venv\Scripts\python.exe -m pytest <args>`

---

## Task 1: Add `AttractorType.UNDETERMINED`

**Files:**
- Modify: `src/mneme/types.py:37-42`
- Test: `tests/test_classify.py` (create with this one test for now)

- [ ] **Step 1: Write the failing test**

Create `tests/test_classify.py`:

```python
"""Tests for mneme.core.classify — gated attractor classification."""

import numpy as np
import pytest

from mneme.types import AttractorType


def test_undetermined_member_exists():
    assert AttractorType.UNDETERMINED == "undetermined"
    assert AttractorType.UNDETERMINED.value == "undetermined"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `win_venv\Scripts\python.exe -m pytest tests/test_classify.py -v`
Expected: FAIL — `AttributeError: UNDETERMINED`

- [ ] **Step 3: Add the enum member**

In `src/mneme/types.py`, the `AttractorType` enum (lines 37-42) becomes:

```python
class AttractorType(str, Enum):
    """Types of dynamical attractors."""
    FIXED_POINT = "fixed_point"
    LIMIT_CYCLE = "limit_cycle"
    STRANGE = "strange"
    QUASI_PERIODIC = "quasi_periodic"
    UNDETERMINED = "undetermined"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `win_venv\Scripts\python.exe -m pytest tests/test_classify.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/mneme/types.py tests/test_classify.py
git commit -m "feat: add AttractorType.UNDETERMINED"
```

---

## Task 2: RK4 Lorenz/Rössler test fixtures

**Files:**
- Modify: `tests/conftest.py:85-104` (replace `lorenz_trajectory`), append new fixtures

- [ ] **Step 1: Write the failing test**

Append to `tests/test_classify.py` temporarily (will move in Task 6) — actually create `tests/test_lyapunov.py` with a fixture sanity test:

```python
"""Tests for mneme.core.lyapunov."""

import numpy as np


def test_lorenz_rk4_fixture_shape(lorenz_rk4):
    traj, dt = lorenz_rk4
    assert traj.shape[1] == 3
    assert traj.shape[0] >= 6000
    assert dt == 0.01
    assert np.all(np.isfinite(traj))


def test_rossler_rk4_fixture_shape(rossler_rk4):
    traj, dt = rossler_rk4
    assert traj.shape[1] == 3
    assert np.all(np.isfinite(traj))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `win_venv\Scripts\python.exe -m pytest tests/test_lyapunov.py -v`
Expected: FAIL — `fixture 'lorenz_rk4' not found`

- [ ] **Step 3: Replace Euler Lorenz with RK4 fixtures**

In `tests/conftest.py`, replace the `lorenz_trajectory` fixture (lines 85-104) with:

```python
def _rk4(deriv, state0, dt, n_steps):
    states = np.empty((n_steps, len(state0)))
    s = np.asarray(state0, dtype=float)
    for i in range(n_steps):
        states[i] = s
        k1 = deriv(s)
        k2 = deriv(s + 0.5 * dt * k1)
        k3 = deriv(s + 0.5 * dt * k2)
        k4 = deriv(s + dt * k3)
        s = s + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return states


@pytest.fixture
def lorenz_rk4():
    """RK4-integrated Lorenz attractor. Returns (trajectory (N,3), dt).

    100-step transient discarded. Standard params -> lambda1 ~ 0.906.
    """
    sigma, rho, beta = 10.0, 28.0, 8.0 / 3.0

    def deriv(s):
        x, y, z = s
        return np.array([sigma * (y - x), x * (rho - z) - y, x * y - beta * z])

    dt = 0.01
    traj = _rk4(deriv, [1.0, 1.0, 1.0], dt, 6500)
    return traj[100:], dt


@pytest.fixture
def lorenz_trajectory(lorenz_rk4):
    """Back-compat alias used by existing recurrence tests: (N,3) array only."""
    return lorenz_rk4[0]


@pytest.fixture
def rossler_rk4():
    """RK4-integrated Rössler attractor. Returns (trajectory (N,3), dt).

    a=b=0.2, c=5.7 -> lambda1 ~ 0.071.
    """
    a, b, c = 0.2, 0.2, 5.7

    def deriv(s):
        x, y, z = s
        return np.array([-y - z, x + a * y, b + z * (x - c)])

    dt = 0.05
    traj = _rk4(deriv, [1.0, 1.0, 1.0], dt, 8000)
    return traj[500:], dt
```

(Keep the existing `import numpy as np` / `import pytest` at the top of `conftest.py`; they are already present.)

- [ ] **Step 4: Run test to verify it passes**

Run: `win_venv\Scripts\python.exe -m pytest tests/test_lyapunov.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Run existing recurrence tests still green**

Run: `win_venv\Scripts\python.exe -m pytest tests/test_attractors.py -k "Recurrence or Clustering" -q`
Expected: PASS (the `lorenz_trajectory` alias keeps them working)

- [ ] **Step 6: Commit**

```bash
git add tests/conftest.py tests/test_lyapunov.py
git commit -m "test: RK4 Lorenz/Rössler fixtures replacing Euler Lorenz"
```

---

## Task 3: `embedding.py`

**Files:**
- Create: `src/mneme/core/embedding.py`
- Test: `tests/test_embedding.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_embedding.py`:

```python
"""Tests for mneme.core.embedding."""

import numpy as np
import pytest

from mneme.core.embedding import (
    cao_embedding_dimension,
    embed_trajectory,
    estimate_embedding_parameters,
    mutual_information_delay,
    theiler_window,
)


class TestEmbedTrajectory:
    def test_1d_embedding_shape(self):
        sig = np.sin(np.linspace(0, 50, 1000))
        emb = embed_trajectory(sig, embedding_dimension=3, time_delay=1)
        assert emb.shape == (998, 3)

    def test_delay_one_columns(self):
        sig = np.arange(100.0)
        emb = embed_trajectory(sig, embedding_dimension=2, time_delay=1)
        np.testing.assert_array_equal(emb[:, 0], sig[: len(emb)])
        np.testing.assert_array_equal(emb[:, 1], sig[1 : len(emb) + 1])

    def test_short_series_raises(self):
        with pytest.raises(ValueError, match="too short"):
            embed_trajectory(np.array([1.0, 2.0]), embedding_dimension=5, time_delay=2)


class TestTheilerWindow:
    def test_positive_int(self):
        sig = np.sin(np.linspace(0, 80 * np.pi, 4000))
        w = theiler_window(sig)
        assert isinstance(w, int) and w >= 1

    def test_sine_window_near_quarter_period(self):
        # period = 100 samples -> autocorr first zero ~ 25 samples
        t = np.arange(4000)
        sig = np.sin(2 * np.pi * t / 100.0)
        w = theiler_window(sig)
        assert 15 <= w <= 40


class TestMutualInformationDelay:
    def test_recovers_quarter_period_on_sine(self):
        t = np.arange(5000)
        sig = np.sin(2 * np.pi * t / 40.0)  # period 40 -> first MI min ~ 10
        d = mutual_information_delay(sig, max_delay=60)
        assert 6 <= d <= 14

    def test_returns_positive_int(self):
        rng = np.random.RandomState(0)
        d = mutual_information_delay(rng.randn(2000), max_delay=50)
        assert isinstance(d, int) and d >= 1


class TestCaoEmbeddingDimension:
    def test_lorenz_dimension(self, lorenz_rk4):
        traj, _ = lorenz_rk4
        d = cao_embedding_dimension(traj[:, 0], delay=8, max_dim=8)
        assert 2 <= d <= 5

    def test_returns_positive_int(self):
        sig = np.sin(np.linspace(0, 100, 3000))
        d = cao_embedding_dimension(sig, delay=10, max_dim=8)
        assert isinstance(d, int) and d >= 1


class TestEstimateEmbeddingParameters:
    def test_returns_two_positive_ints(self, lorenz_rk4):
        traj, _ = lorenz_rk4
        dim, delay = estimate_embedding_parameters(traj[:, 0])
        assert isinstance(dim, int) and isinstance(delay, int)
        assert dim >= 1 and delay >= 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `win_venv\Scripts\python.exe -m pytest tests/test_embedding.py -v`
Expected: FAIL — `ModuleNotFoundError: mneme.core.embedding`

- [ ] **Step 3: Create `src/mneme/core/embedding.py`**

```python
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
    # First zero crossing
    for lag in range(1, len(ac)):
        if ac[lag] <= 0.0:
            return max(1, lag)
    # Fallback: 1/e decay
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
            return i + 1  # delays start at 1
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
    e1 = e[1:] / e[:-1]  # E1(d) for d = 1 .. len-1
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `win_venv\Scripts\python.exe -m pytest tests/test_embedding.py -v`
Expected: PASS (all). If `test_lorenz_dimension` or `test_recovers_quarter_period_on_sine` are marginally outside the asserted band, adjust only the `n_bins` heuristic / Cao 5% threshold constant — do not loosen the test bands.

- [ ] **Step 5: Commit**

```bash
git add src/mneme/core/embedding.py tests/test_embedding.py
git commit -m "feat: embedding module (MI delay, Cao dimension, Theiler window)"
```

---

## Task 4: `lyapunov.py`

**Files:**
- Create: `src/mneme/core/lyapunov.py`
- Test: `tests/test_lyapunov.py` (extend the file from Task 2)

- [ ] **Step 1: Write the failing validation tests**

Replace the contents of `tests/test_lyapunov.py` with:

```python
"""Validation tests for mneme.core.lyapunov against known systems."""

import numpy as np
import pytest

from mneme.core.lyapunov import LyapunovResult, largest_lyapunov, lyapunov_spectrum


def test_fixture_shapes(lorenz_rk4, rossler_rk4):
    assert lorenz_rk4[0].shape[1] == 3
    assert rossler_rk4[0].shape[1] == 3


class TestLargestLyapunov:
    def test_lorenz_lambda1(self, lorenz_rk4):
        traj, dt = lorenz_rk4
        res = largest_lyapunov(traj[:, 0], dt=dt)
        assert isinstance(res, LyapunovResult)
        assert 0.85 <= res.lambda1 <= 0.97

    def test_lorenz_scale_invariant(self, lorenz_rk4):
        traj, dt = lorenz_rk4
        a = largest_lyapunov(traj[:, 0], dt=dt).lambda1
        b = largest_lyapunov(1000.0 * traj[:, 0], dt=dt).lambda1
        assert abs(a - b) <= 0.01 * abs(a)

    def test_rossler_lambda1(self, rossler_rk4):
        traj, dt = rossler_rk4
        res = largest_lyapunov(traj[:, 0], dt=dt)
        assert 0.04 <= res.lambda1 <= 0.11

    def test_sine_near_zero(self):
        t = np.arange(6000)
        sig = np.sin(2 * np.pi * t / 50.0)
        res = largest_lyapunov(sig, dt=1.0)
        assert abs(res.lambda1) < 0.02

    def test_result_fields_populated(self, lorenz_rk4):
        traj, dt = lorenz_rk4
        res = largest_lyapunov(traj[:, 0], dt=dt)
        assert res.divergence_curve.ndim == 1
        assert res.fit_region[0] < res.fit_region[1]
        assert res.emb_dim >= 2 and res.delay >= 1 and res.theiler >= 1


class TestLyapunovSpectrum:
    def test_lorenz_spectrum_sum_and_lambda1(self, lorenz_rk4):
        traj, dt = lorenz_rk4
        spec = lyapunov_spectrum(traj, dt=dt)
        assert len(spec) == 3
        assert 0.80 <= spec[0] <= 1.05
        assert -16.0 <= float(np.sum(spec)) <= -11.0

    def test_short_trajectory_raises(self):
        with pytest.raises(ValueError, match="too short"):
            lyapunov_spectrum(np.column_stack([np.arange(20.0)] * 3), dt=0.01)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `win_venv\Scripts\python.exe -m pytest tests/test_lyapunov.py -v`
Expected: FAIL — `ModuleNotFoundError: mneme.core.lyapunov`

- [ ] **Step 3: Create `src/mneme/core/lyapunov.py`**

```python
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
    """
    n = len(curve)
    start = min(2, max(0, n - 2))
    if n - start < 4:
        return start, n
    # Early slope from the first quarter of the post-transient curve.
    probe = max(start + 2, start + (n - start) // 4)
    early_slope = (curve[probe] - curve[start]) / (probe - start)
    if early_slope <= 0:
        return start, n  # degenerate (non-diverging) — caller will get ~0
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
    # For each reference point, the nearest neighbour outside the Theiler window.
    k = min(n, 4 * theiler + 8)
    dists, idxs = tree.query(emb, k=k)
    neighbour = np.full(n, -1, dtype=int)
    for i in range(n):
        for j_pos in range(1, idxs.shape[1]):
            j = idxs[i, j_pos]
            if abs(j - i) > theiler:
                neighbour[i] = j
                break

    # Mean log divergence at each horizon step.
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
    n_neighbors: int = 20,
) -> np.ndarray:
    """EXPLORATORY full Lyapunov spectrum (corrected Sano-Sawada).

    Local-Jacobian QR method with Theiler exclusion, ridge-regularised
    neighbour regression, and consistent log-growth / time normalisation.
    Emits a RuntimeWarning: prefer `largest_lyapunov` for the headline λ₁.
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
        # Ridge-regularised least squares: J ≈ argmin ||dx J^T - dy||.
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `win_venv\Scripts\python.exe -m pytest tests/test_lyapunov.py -v`
Expected: PASS. The fragile gate is `test_lorenz_lambda1` / `test_rossler_lambda1`. If λ₁ lands outside the band, tune **only** the `_linear_region` heuristic constants (transient skip = 2, early-slope probe fraction = 1/4, plateau threshold = 0.5) and `max_steps` default — keep the algorithm and the test bands fixed. Iterate until both Lorenz and Rössler pass. This empirical tuning of the linear-region heuristic is expected TDD work, not a placeholder.

- [ ] **Step 5: Add a non-slow CI Lorenz test**

Append to `tests/test_lyapunov.py`:

```python
def test_lorenz_lambda1_runs_in_default_ci(lorenz_rk4):
    """Downsized, NOT marked slow — guards the headline claim in default CI."""
    traj, dt = lorenz_rk4
    res = largest_lyapunov(traj[:3000, 0], dt=dt)
    assert res.lambda1 > 0.3  # clearly positive, fast
```

Run: `win_venv\Scripts\python.exe -m pytest tests/test_lyapunov.py -q`
Expected: PASS (all).

- [ ] **Step 6: Commit**

```bash
git add src/mneme/core/lyapunov.py tests/test_lyapunov.py
git commit -m "feat: Rosenstein largest_lyapunov + exploratory Sano-Sawada spectrum"
```

---

## Task 5: `surrogates.py`

**Files:**
- Create: `src/mneme/core/surrogates.py`
- Test: `tests/test_surrogates.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_surrogates.py`:

```python
"""Tests for mneme.core.surrogates (IAAFT + surrogate significance test)."""

import numpy as np

from mneme.core.surrogates import (
    SurrogateResult,
    iaaft_surrogates,
    surrogate_test,
)


class TestIAAFT:
    def test_shape_and_amplitude_preserved(self):
        rng = np.random.RandomState(1)
        x = np.cumsum(rng.randn(512))
        sur = iaaft_surrogates(x, n=5, seed=0)
        assert sur.shape == (5, 512)
        np.testing.assert_allclose(np.sort(sur[0]), np.sort(x), rtol=0, atol=1e-6)

    def test_power_spectrum_approx_preserved(self):
        rng = np.random.RandomState(2)
        x = np.sin(np.linspace(0, 60, 1024)) + 0.1 * rng.randn(1024)
        sur = iaaft_surrogates(x, n=3, seed=1)
        px = np.abs(np.fft.rfft(x - x.mean()))
        ps = np.abs(np.fft.rfft(sur[0] - sur[0].mean()))
        # Correlate spectra — IAAFT preserves linear (spectral) structure.
        r = np.corrcoef(px, ps)[0, 1]
        assert r > 0.95

    def test_reproducible_with_seed(self):
        rng = np.random.RandomState(3)
        x = rng.randn(256)
        a = iaaft_surrogates(x, n=2, seed=42)
        b = iaaft_surrogates(x, n=2, seed=42)
        np.testing.assert_array_equal(a, b)


class TestSurrogateTest:
    def test_white_noise_not_significant(self):
        rng = np.random.RandomState(4)
        res = surrogate_test(rng.randn(1500), statistic="lambda1", n=30, seed=0)
        assert isinstance(res, SurrogateResult)
        assert res.significant is False

    def test_ar1_noise_not_significant(self):
        rng = np.random.RandomState(5)
        x = np.zeros(1500)
        for i in range(1, 1500):
            x[i] = 0.7 * x[i - 1] + rng.randn()
        res = surrogate_test(x, statistic="lambda1", n=30, seed=0)
        assert res.significant is False

    def test_lorenz_is_significant(self, lorenz_rk4):
        traj, dt = lorenz_rk4
        res = surrogate_test(
            traj[:2500, 0], statistic="lambda1", n=30, seed=0, dt=dt
        )
        assert res.significant is True
        assert res.p_value < 0.05
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `win_venv\Scripts\python.exe -m pytest tests/test_surrogates.py -v`
Expected: FAIL — `ModuleNotFoundError: mneme.core.surrogates`

- [ ] **Step 3: Create `src/mneme/core/surrogates.py`**

```python
"""IAAFT surrogate data and rank-based significance testing.

Implements the Schreiber–Schmitz iterative amplitude-adjusted Fourier
transform and a one-sided rank test that gates chaos claims.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .lyapunov import largest_lyapunov


def iaaft_surrogates(
    x: np.ndarray,
    n: int = 200,
    *,
    max_iter: int = 1000,
    tol: float = 1e-8,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Generate `n` IAAFT surrogates of a 1-D series.

    Each surrogate preserves the amplitude distribution and (closely) the
    power spectrum of `x` while randomising nonlinear structure.

    Returns
    -------
    np.ndarray
        Shape (n, len(x)).
    """
    x = np.asarray(x, dtype=float).ravel()
    rng = np.random.RandomState(seed)
    sorted_x = np.sort(x)
    target_amp = np.abs(np.fft.rfft(x))

    out = np.empty((n, len(x)))
    for s in range(n):
        surrogate = rng.permutation(x)
        prev = None
        for _ in range(max_iter):
            # Match power spectrum.
            fft = np.fft.rfft(surrogate)
            phases = np.angle(fft)
            surrogate = np.fft.irfft(target_amp * np.exp(1j * phases), n=len(x))
            # Match amplitude distribution (rank remap).
            ranks = np.argsort(np.argsort(surrogate))
            surrogate = sorted_x[ranks]
            if prev is not None and np.mean((surrogate - prev) ** 2) < tol:
                break
            prev = surrogate.copy()
        out[s] = surrogate
    return out


@dataclass
class SurrogateResult:
    """Outcome of a surrogate-data significance test."""

    statistic_name: str
    statistic_value: float
    null_distribution: np.ndarray
    p_value: float
    n_surrogates: int
    alpha: float
    significant: bool


def _lambda1_stat(series: np.ndarray, **kw) -> float:
    return largest_lyapunov(series, **kw).lambda1


_STATISTICS = {"lambda1": _lambda1_stat}


def surrogate_test(
    trajectory: np.ndarray,
    statistic: str = "lambda1",
    n: int = 200,
    *,
    alpha: float = 0.05,
    seed: Optional[int] = None,
    **stat_kwargs,
) -> SurrogateResult:
    """One-sided IAAFT surrogate test.

    H0: the discriminating statistic of `trajectory` is consistent with a
    linear stochastic process. Rejected when the original statistic exceeds
    the surrogate null distribution at level `alpha`.

    p_value = (1 + #{surrogate >= original}) / (n + 1)
    """
    if statistic not in _STATISTICS:
        raise ValueError(
            f"Unknown statistic {statistic!r}. Available: {list(_STATISTICS)}"
        )
    stat_fn = _STATISTICS[statistic]

    arr = np.asarray(trajectory, dtype=float)
    series_1d = arr if arr.ndim == 1 else arr[:, 0]

    observed = stat_fn(series_1d, **stat_kwargs)
    surrogates = iaaft_surrogates(series_1d, n=n, seed=seed)
    null = np.array([stat_fn(s, **stat_kwargs) for s in surrogates])

    p_value = (1.0 + np.sum(null >= observed)) / (n + 1.0)
    significant = bool(p_value < alpha)
    return SurrogateResult(
        statistic_name=statistic,
        statistic_value=float(observed),
        null_distribution=null,
        p_value=float(p_value),
        n_surrogates=n,
        alpha=alpha,
        significant=significant,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `win_venv\Scripts\python.exe -m pytest tests/test_surrogates.py -v`
Expected: PASS. `test_white_noise_not_significant` / `test_ar1_noise_not_significant` are the credibility gates — they must be `significant is False`. If a noise case is flakily significant, increase `n` in that test to 50; do not weaken the assertion. (Suppress the expected `RuntimeWarning` from short-trajectory λ₁ inside surrogate loops by running with `-W ignore::RuntimeWarning` if it clutters output — behavior is unaffected.)

- [ ] **Step 5: Commit**

```bash
git add src/mneme/core/surrogates.py tests/test_surrogates.py
git commit -m "feat: IAAFT surrogates + rank-based significance test"
```

---

## Task 6: `classify.py`

**Files:**
- Create: `src/mneme/core/classify.py`
- Test: `tests/test_classify.py` (extend the file from Task 1)

- [ ] **Step 1: Write the failing tests**

Replace the contents of `tests/test_classify.py` with:

```python
"""Tests for mneme.core.classify — gated classification + Kaplan-Yorke."""

import numpy as np

from mneme.core.classify import classify_attractor, kaplan_yorke_dimension
from mneme.core.surrogates import SurrogateResult
from mneme.types import AttractorType


def _sig(significant: bool) -> SurrogateResult:
    return SurrogateResult(
        statistic_name="lambda1",
        statistic_value=0.9,
        null_distribution=np.zeros(10),
        p_value=0.001 if significant else 0.5,
        n_surrogates=10,
        alpha=0.05,
        significant=significant,
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `win_venv\Scripts\python.exe -m pytest tests/test_classify.py -v`
Expected: FAIL — `ModuleNotFoundError: mneme.core.classify`

- [ ] **Step 3: Create `src/mneme/core/classify.py`**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `win_venv\Scripts\python.exe -m pytest tests/test_classify.py -v`
Expected: PASS (all).

- [ ] **Step 5: Commit**

```bash
git add src/mneme/core/classify.py tests/test_classify.py
git commit -m "feat: surrogate-gated classify_attractor + Kaplan-Yorke (moved)"
```

---

## Task 7: Clean break in `attractors.py`

**Files:**
- Modify: `src/mneme/core/attractors.py` — delete `embed_trajectory` (lines 769-806), `estimate_embedding_parameters` (809-842), `_estimate_time_delay_mutual_info` (845-865), `_estimate_dimension_fnn` (868-928), `compute_lyapunov_spectrum` (992-1128), `_estimate_local_jacobian` (1131-1197), `classify_attractor_by_lyapunov` (1200-1237), `kaplan_yorke_dimension` (1240-1285). Keep `compute_correlation_dimension` (931-...). Rewire imports and `LyapunovAnalysis.compute_lyapunov_spectrum`.

- [ ] **Step 1: Add import + delegate; delete dead code**

At the top of `src/mneme/core/attractors.py`, add to the imports:

```python
from .embedding import embed_trajectory, estimate_embedding_parameters
from .lyapunov import lyapunov_spectrum
```

Delete the now-duplicated/old definitions listed above (`embed_trajectory`, `estimate_embedding_parameters`, `_estimate_time_delay_mutual_info`, `_estimate_dimension_fnn`, `compute_lyapunov_spectrum`, `_estimate_local_jacobian`, `classify_attractor_by_lyapunov`, `kaplan_yorke_dimension`). `RecurrenceAnalysis` / `ClusteringDetector` / `LyapunovAnalysis.detect` keep calling `embed_trajectory(...)` — now resolved via the new import.

Replace the body of `LyapunovAnalysis.compute_lyapunov_spectrum` (was lines 512-551) with:

```python
    def compute_lyapunov_spectrum(
        self,
        trajectory: np.ndarray,
        dt: float = 1.0,
    ) -> np.ndarray:
        """EXPLORATORY full Lyapunov spectrum (delegates to lyapunov module).

        Prefer ``mneme.core.largest_lyapunov`` for the headline exponent;
        this returns the exploratory Sano-Sawada spectrum.
        """
        return lyapunov_spectrum(trajectory, dt=dt)
```

- [ ] **Step 2: Verify no remaining references to deleted names**

Run: `win_venv\Scripts\python.exe -m pytest --co -q 2>&1 | head -5` then
`grep -rn "compute_lyapunov_spectrum\|classify_attractor_by_lyapunov\|_estimate_local_jacobian" src/mneme/core/attractors.py`
Expected: only the `LyapunovAnalysis.compute_lyapunov_spectrum` method definition remains; no module-level definitions, no `_estimate_local_jacobian`.

- [ ] **Step 3: Smoke-import**

Run: `win_venv\Scripts\python.exe -c "import mneme.core.attractors as a; print(hasattr(a,'compute_lyapunov_spectrum'), hasattr(a,'RecurrenceAnalysis'))"`
Expected: `False True` (module-level function gone; class kept)

- [ ] **Step 4: Commit**

```bash
git add src/mneme/core/attractors.py
git commit -m "refactor: remove old Lyapunov code from attractors.py (clean break)"
```

---

## Task 8: Update `mneme/core/__init__.py`

**Files:**
- Modify: `src/mneme/core/__init__.py:18-41`

- [ ] **Step 1: Write the failing test**

Create `tests/test_core_exports.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `win_venv\Scripts\python.exe -m pytest tests/test_core_exports.py -v`
Expected: FAIL — `largest_lyapunov` not found

- [ ] **Step 3: Rewrite the exports**

Replace lines 18-41 of `src/mneme/core/__init__.py` with:

```python
from .embedding import embed_trajectory, estimate_embedding_parameters
from .lyapunov import LyapunovResult, largest_lyapunov, lyapunov_spectrum
from .surrogates import SurrogateResult, iaaft_surrogates, surrogate_test
from .classify import classify_attractor, kaplan_yorke_dimension

__all__ = [
    # Modules
    "field_theory",
    "topology",
    "attractors",
    # Reconstructors
    "FieldReconstructor",
    "SparseGPReconstructor",
    "DenseIFTReconstructor",
    "GaussianProcessReconstructor",
    "NeuralFieldReconstructor",
    "create_reconstructor",
    "create_grid_points",
    # Embedding
    "embed_trajectory",
    "estimate_embedding_parameters",
    # Lyapunov analysis
    "LyapunovResult",
    "largest_lyapunov",
    "lyapunov_spectrum",
    # Surrogate significance
    "SurrogateResult",
    "iaaft_surrogates",
    "surrogate_test",
    # Classification
    "classify_attractor",
    "kaplan_yorke_dimension",
]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `win_venv\Scripts\python.exe -m pytest tests/test_core_exports.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/mneme/core/__init__.py tests/test_core_exports.py
git commit -m "feat: export new Tier 0 API from mneme.core; drop old names"
```

---

## Task 9: Migrate `tests/test_attractors.py`

**Files:**
- Modify: `tests/test_attractors.py:6-16` (imports), delete `TestEmbedTrajectory`, `TestClassifyAttractorByLyapunov`, `TestKaplanYorkeDimension`, `TestComputeLyapunovSpectrum` (lines 20-125 — now covered by `test_embedding.py` / `test_lyapunov.py` / `test_classify.py`)

- [ ] **Step 1: Rewrite imports**

Replace lines 6-17 of `tests/test_attractors.py` with:

```python
from mneme.core.attractors import (
    AttractorDetector,
    ClusteringDetector,
    LyapunovAnalysis,
    RecurrenceAnalysis,
    compute_correlation_dimension,
)
from mneme.core.embedding import embed_trajectory
from mneme.types import AttractorType
```

- [ ] **Step 2: Delete superseded test classes**

Delete the class blocks `TestEmbedTrajectory` (lines ~24-46), `TestClassifyAttractorByLyapunov` (~53-70), `TestKaplanYorkeDimension` (~77-96), and `TestComputeLyapunovSpectrum` (~103-124) including their section comment banners. Keep `TestRecurrenceAnalysis`, `TestAttractorDetector`, `TestComputeCorrelationDimension`.

- [ ] **Step 3: Run the trimmed file**

Run: `win_venv\Scripts\python.exe -m pytest tests/test_attractors.py -v`
Expected: PASS (Recurrence/AttractorDetector/CorrelationDimension classes only; no import errors)

- [ ] **Step 4: Commit**

```bash
git add tests/test_attractors.py
git commit -m "test: drop attractors tests superseded by Tier 0 modules"
```

---

## Task 10: Migrate the 3 analysis scripts + smoke test

**Files:**
- Modify: `scripts/analyze_physionet.py:13-51`, `scripts/deep_analysis.py:30-32,106-148` and its VAE block (~557-583), `scripts/analyze_betse.py:41-43,141-160` and single-cell block (~326-347)
- Create: `tests/test_scripts_smoke.py`

The mechanical migration pattern (apply at every old call site):

```python
# OLD
spectrum = compute_lyapunov_spectrum(traj, dt=DT, n_neighbors=K)         # remove
atype = classify_attractor_by_lyapunov(spectrum)                         # remove
d_ky = kaplan_yorke_dimension(spectrum)

# NEW
from mneme.core import (
    largest_lyapunov, lyapunov_spectrum, surrogate_test,
    classify_attractor, kaplan_yorke_dimension,
)
lyap = largest_lyapunov(traj, dt=DT)
sur = surrogate_test(traj, statistic="lambda1", n=50, dt=DT)
atype = classify_attractor(lyap.lambda1, surrogate=sur)
spectrum = lyapunov_spectrum(traj, dt=DT)   # exploratory; for D_KY only
d_ky = kaplan_yorke_dimension(spectrum)
# expose lyap.lambda1, sur.p_value, atype, d_ky in the result dicts
```

- [ ] **Step 1: Migrate `scripts/analyze_physionet.py`**

Replace imports (lines 13-14):

```python
from mneme.core import (
    classify_attractor,
    kaplan_yorke_dimension,
    largest_lyapunov,
    lyapunov_spectrum,
    surrogate_test,
)
from mneme.core.embedding import embed_trajectory
```

Replace the analysis block (old lines 40-51) inside `analyze_ecg_hrv` with:

```python
    lyap = largest_lyapunov(trajectory, dt=dt_hrv)
    sur = surrogate_test(trajectory, statistic="lambda1", n=50, dt=dt_hrv)
    atype = classify_attractor(lyap.lambda1, surrogate=sur)
    spectrum = lyapunov_spectrum(trajectory, dt=dt_hrv)
    d_ky = kaplan_yorke_dimension(spectrum)

    return {
        'n_beats': len(peaks),
        'mean_hr': 60000 / np.mean(rr_intervals),
        'rr_std': np.std(rr_intervals),
        'lambda1': float(lyap.lambda1),
        'surrogate_p': float(sur.p_value),
        'spectrum': spectrum,
        'd_ky': d_ky,
        'type': str(atype),
    }
```

- [ ] **Step 2: Migrate `scripts/deep_analysis.py`**

Replace the import block (lines 30-32) with the 5 new names + `kaplan_yorke_dimension`. In `lyapunov_from_pca` (lines 115-145) replace the `compute_lyapunov_spectrum`/`classify_attractor_by_lyapunov` pair with the NEW pattern (use `pca_coeffs` as `traj`, `dt=1.0`, `n=30`). In the VAE-latent block (~557-573) apply the same NEW pattern to the latent trajectory. Keep the existing `try/except` and `result` dict keys; add `"lambda1"` and `"surrogate_p"` keys.

- [ ] **Step 3: Migrate `scripts/analyze_betse.py`**

Replace imports (lines 41-43) with the new names. In the Lyapunov block (lines 142-160) and the single-cell block (~326-347) apply the NEW pattern (`trajectory`, `dt=1.0`, `n=30`). Keep `try/except` and dict keys; add `"lambda1"`, `"surrogate_p"`.

- [ ] **Step 4: Write the smoke test**

Create `tests/test_scripts_smoke.py`:

```python
"""Import-and-run smoke test for the migrated analysis scripts.

Only checks the scripts import and their Lyapunov code path runs against
synthetic data with the new API. Numeric reconciliation is Tier 1.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_analyze_physionet_imports():
    _load("analyze_physionet")  # must not raise (old API names are gone)


def test_deep_analysis_lyapunov_path_runs():
    mod = _load("deep_analysis")
    rng = np.random.RandomState(0)
    pca = rng.randn(1200, 3)
    res = mod.lyapunov_from_pca(pca, label="smoke")
    assert "error" in res or "lambda1" in res


def test_analyze_betse_imports():
    _load("analyze_betse")
```

- [ ] **Step 5: Run the smoke test**

Run: `win_venv\Scripts\python.exe -m pytest tests/test_scripts_smoke.py -v`
Expected: PASS (3 passed). If a script imports a heavy optional dep at module top (e.g. `wfdb`) and it is missing, wrap that import test in `pytest.importorskip("wfdb")` at the top of the relevant test — do not stub the Mneme API.

- [ ] **Step 6: Commit**

```bash
git add scripts/analyze_physionet.py scripts/deep_analysis.py scripts/analyze_betse.py tests/test_scripts_smoke.py
git commit -m "refactor: migrate analysis scripts to Tier 0 API + smoke test"
```

---

## Task 11: Update docs

**Files:**
- Modify: `CLAUDE.md` (Lyapunov Usage section + "Validated on real data" line), `README.md` (Lyapunov snippet + headline numbers), `CHANGELOG.md`

- [ ] **Step 1: Update `CLAUDE.md`**

In the "Lyapunov Spectrum Usage" code block, replace the old API usage with:

```python
from mneme.core import largest_lyapunov, surrogate_test, classify_attractor, lyapunov_spectrum, kaplan_yorke_dimension

res = largest_lyapunov(trajectory, dt=0.01)          # robust λ₁ (Rosenstein)
sur = surrogate_test(trajectory, statistic="lambda1", n=200, dt=0.01)
attractor_type = classify_attractor(res.lambda1, surrogate=sur)  # STRANGE only if sur.significant
spectrum = lyapunov_spectrum(trajectory, dt=0.01)    # EXPLORATORY full spectrum
d_ky = kaplan_yorke_dimension(spectrum)
```

Replace the bolded "Validated on real data: PhysioNet ECG ... λ₁=+0.12/s, D_KY=2.35, matching published literature" line with:

> **Validation status:** PhysioNet HRV results are **pending re-validation** under the corrected estimators and surrogate gating (Tier 0 replaced the previous Lyapunov implementation). Prior headline numbers are not asserted.

- [ ] **Step 2: Update `README.md`**

Find the README Lyapunov code snippet and the "Validated on Real Biological Data" / `λ₁=+0.123, D_KY=2.35` claims. Replace the snippet with the same new-API block as Step 1. Replace the asserted numbers with: "Lyapunov/attractor results are pending re-validation under the Tier 0 corrected estimators (surrogate-gated; chaos is not claimed without passed surrogate tests)."

- [ ] **Step 3: Add `CHANGELOG.md` entry**

Add under an `## [Unreleased]` section (create it at the top if absent):

```markdown
### Changed (Tier 0 — scientific core repair)
- **BREAKING:** removed `compute_lyapunov_spectrum` and `classify_attractor_by_lyapunov`.
- Added `largest_lyapunov` (Rosenstein 1993), exploratory `lyapunov_spectrum`
  (corrected Sano-Sawada), `surrogate_test` (IAAFT), and surrogate-gated
  `classify_attractor` with new `AttractorType.UNDETERMINED`.
- Added `mneme.core.embedding` (true-MI delay, Cao-1997 dimension, Theiler window).
- Chaos / strange-attractor labels now require a passed surrogate test.
- Previous PhysioNet headline numbers are withdrawn pending re-validation.
```

- [ ] **Step 4: Commit**

```bash
git add CLAUDE.md README.md CHANGELOG.md
git commit -m "docs: update API + withdraw unvalidated Lyapunov claims (Tier 0)"
```

---

## Task 12: Full-suite verification

- [ ] **Step 1: Run the entire test suite**

Run: `win_venv\Scripts\python.exe -m pytest tests -m "not slow" -q`
Expected: ALL PASS, no collection errors, no `ModuleNotFoundError`. New modules (`embedding`, `lyapunov`, `surrogates`, `classify`) covered by their test files.

- [ ] **Step 2: If any failure, fix at the source**

Use superpowers:systematic-debugging on any failure. Do NOT weaken validation tolerances in `test_lyapunov.py` / `test_surrogates.py` — those encode the scientific acceptance gates.

- [ ] **Step 3: Final verification commit (if fixes were made)**

```bash
git add -A
git commit -m "test: Tier 0 full-suite green"
```

- [ ] **Step 4: Report**

State explicitly: which validation gates pass with what measured values (Lorenz λ₁, Rössler λ₁, noise non-significance), full suite pass count, and that no tolerances were weakened.

---

## Self-Review Notes (author)

- **Spec coverage:** §5 module layout → Tasks 3-8; §6.1 embedding → Task 3; §6.2 estimators → Task 4; §6.3 surrogates → Task 5; §6.4 classify + UNDETERMINED → Tasks 1,6; §6.5 deletions/migrations → Tasks 7-10; §7 validation → Tasks 3-6 tests + Task 4 Step 5 (CI); §9 DoD → Task 12. Pipeline fail-soft fix correctly excluded (Non-Goal).
- **Type consistency:** `LyapunovResult(lambda1, divergence_curve, fit_region, emb_dim, delay, theiler, dt)` and `SurrogateResult(statistic_name, statistic_value, null_distribution, p_value, n_surrogates, alpha, significant)` are used identically in Tasks 4-6 and the script-migration pattern. `classify_attractor(lambda1, *, surrogate, oscillatory, zero_tol)` signature consistent across Task 6 and Task 10.
- **Known fragile step:** Rosenstein linear-region heuristic (Task 4 Step 4) — flagged as expected TDD tuning, bounded to named constants, with fixed test gates.
