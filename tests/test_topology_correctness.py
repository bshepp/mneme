"""Correctness tests for persistence and diagram distances.

These check answers that are known in closed form, and check the built-in
fallbacks against GUDHI where GUDHI is installed.
"""

import warnings

import numpy as np
import pytest

from mneme.core import topology
from mneme.core.topology import (
    PersistentHomology,
    _h0_sublevel_persistence,
    compute_bottleneck_distance,
    compute_wasserstein_distance,
)
from mneme.types import PersistenceDiagram


def _two_pits() -> np.ndarray:
    """Flat plateau at 0 with pits of depth -3 and -1, no peaks."""
    field = np.zeros((9, 15))
    field[4, 3] = -3.0
    field[4, 11] = -1.0
    return field


def _finite(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    points = points[np.all(np.isfinite(points), axis=1)]
    return points[np.lexsort((points[:, 1], points[:, 0]))]


def _diagram(points) -> PersistenceDiagram:
    return PersistenceDiagram(
        points=np.asarray(points, dtype=float).reshape(-1, 2), dimension=0, threshold=0.0
    )


class TestH0UnionFind:
    def test_two_pits_sublevel(self):
        points = _h0_sublevel_persistence(_two_pits())
        # Deep pit is essential; shallow pit is born at -1 and dies at 0.
        np.testing.assert_allclose(_finite(points), [[-1.0, 0.0]])
        essential = points[~np.isfinite(points[:, 1])]
        np.testing.assert_allclose(essential[:, 0], [-3.0])

    def test_constant_field_has_only_the_essential_class(self):
        points = _h0_sublevel_persistence(np.ones((5, 7)))
        assert points.shape == (1, 2)
        assert points[0, 0] == 1.0 and np.isinf(points[0, 1])

    def test_diagonal_pixels_are_connected(self):
        # Two minima touching only at a corner form ONE component.
        field = np.ones((4, 4))
        field[1, 1] = 0.0
        field[2, 2] = 0.0
        assert len(_finite(_h0_sublevel_persistence(field))) == 0


class TestFiltrationDirection:
    """`sublevel` must find pits; `superlevel` must find peaks."""

    def test_sublevel_finds_pits(self):
        ph = PersistentHomology(max_dimension=1, filtration="sublevel", persistence_threshold=0.0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            h0 = ph.compute_persistence(_two_pits())[0]
        np.testing.assert_allclose(_finite(h0.points), [[-1.0, 0.0]])

    def test_superlevel_of_pits_has_no_finite_h0(self):
        ph = PersistentHomology(max_dimension=1, filtration="superlevel", persistence_threshold=0.0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            h0 = ph.compute_persistence(_two_pits())[0]
        assert len(_finite(h0.points)) == 0

    def test_superlevel_finds_peaks(self):
        ph = PersistentHomology(max_dimension=1, filtration="superlevel", persistence_threshold=0.0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            h0 = ph.compute_persistence(-_two_pits())[0]
        # Expressed in units of -field: born at -1, dies at 0.
        np.testing.assert_allclose(_finite(h0.points), [[-1.0, 0.0]])

    def test_nan_field_raises(self):
        field = _two_pits()
        field[0, 0] = np.nan
        with pytest.raises(ValueError, match="NaN"):
            PersistentHomology().compute_persistence(field)

    def test_cycles_not_implemented(self):
        with pytest.raises(NotImplementedError):
            PersistentHomology(compute_cycles=True)


class TestAgainstGudhi:
    """The GUDHI path and the independent union-find must agree on H0."""

    @pytest.mark.parametrize("shape", [(12, 12), (6, 17), (17, 6)])
    def test_h0_matches_union_find(self, shape):
        pytest.importorskip("gudhi")
        field = np.random.RandomState(3).standard_normal(shape)
        ph = PersistentHomology(max_dimension=1, filtration="sublevel", persistence_threshold=0.0)
        h0 = ph.compute_persistence(field)[0]
        np.testing.assert_allclose(
            _finite(h0.points), _finite(_h0_sublevel_persistence(field))
        )

    def test_ring_has_one_h1_class(self):
        pytest.importorskip("gudhi")
        field = np.ones((11, 11))
        field[3:8, 3:8] = 0.0   # low ring ...
        field[4:7, 4:7] = 2.0   # ... around a high centre
        ph = PersistentHomology(max_dimension=1, filtration="sublevel", persistence_threshold=0.0)
        h1 = ph.compute_persistence(field)[1]
        np.testing.assert_allclose(_finite(h1.points), [[0.0, 2.0]])


class TestFallbackWithoutGudhi:
    def test_fallback_warns_and_returns_all_dimensions(self, monkeypatch):
        import builtins

        real_import = builtins.__import__

        def no_gudhi(name, *args, **kwargs):
            if name == "gudhi" or name.startswith("gudhi."):
                raise ImportError("gudhi blocked for test")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", no_gudhi)
        ph = PersistentHomology(max_dimension=2, filtration="sublevel", persistence_threshold=0.0)
        with pytest.warns(RuntimeWarning, match="H0 only"):
            diagrams = ph.compute_persistence(_two_pits())
        assert [d.dimension for d in diagrams] == [0, 1, 2]
        np.testing.assert_allclose(_finite(diagrams[0].points), [[-1.0, 0.0]])


class TestDistances:
    """Closed-form cases, L-infinity ground metric."""

    @pytest.fixture
    def builtin_only(self, monkeypatch):
        import builtins

        real_import = builtins.__import__

        def no_gudhi(name, *args, **kwargs):
            if name == "gudhi" or name.startswith("gudhi."):
                raise ImportError("gudhi blocked for test")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", no_gudhi)

    def test_identical_diagrams_are_at_zero(self, builtin_only):
        d = _diagram([[0.0, 1.0], [0.5, 3.0]])
        with pytest.warns(RuntimeWarning):
            assert compute_wasserstein_distance(d, d) == pytest.approx(0.0)
        with pytest.warns(RuntimeWarning):
            assert compute_bottleneck_distance(d, d) == pytest.approx(0.0)

    def test_point_against_empty_costs_half_its_persistence(self, builtin_only):
        d, empty = _diagram([[0.0, 2.0]]), _diagram([])
        with pytest.warns(RuntimeWarning):
            assert compute_wasserstein_distance(d, empty, p=2.0) == pytest.approx(1.0)
        with pytest.warns(RuntimeWarning):
            assert compute_bottleneck_distance(d, empty) == pytest.approx(1.0)

    def test_shifted_point(self, builtin_only):
        a, b = _diagram([[0.0, 10.0]]), _diagram([[1.0, 10.5]])
        with pytest.warns(RuntimeWarning):
            assert compute_wasserstein_distance(a, b, p=2.0) == pytest.approx(1.0)
        with pytest.warns(RuntimeWarning):
            assert compute_bottleneck_distance(a, b) == pytest.approx(1.0)

    def test_two_points_wasserstein_is_root_sum_of_squares(self, builtin_only):
        a = _diagram([[0.0, 10.0], [20.0, 30.0]])
        b = _diagram([[3.0, 10.0], [20.0, 34.0]])
        with pytest.warns(RuntimeWarning):
            assert compute_wasserstein_distance(a, b, p=2.0) == pytest.approx(5.0)
        with pytest.warns(RuntimeWarning):
            assert compute_bottleneck_distance(a, b) == pytest.approx(4.0)

    def test_essential_classes_are_ignored(self, builtin_only):
        a = _diagram([[0.0, np.inf], [0.0, 2.0]])
        b = _diagram([[5.0, np.inf]])
        with pytest.warns(RuntimeWarning):
            assert compute_wasserstein_distance(a, b) == pytest.approx(1.0)

    def test_builtin_matches_gudhi(self):
        pytest.importorskip("gudhi")
        pytest.importorskip("ot")
        rng = np.random.RandomState(5)
        births = rng.uniform(0, 5, (2, 9))
        a = _diagram(np.column_stack([births[0], births[0] + rng.uniform(0.1, 3, 9)]))
        b = _diagram(np.column_stack([births[1], births[1] + rng.uniform(0.1, 3, 9)])[:6])
        via_gudhi = compute_wasserstein_distance(a, b, p=2.0)
        cost = topology._matching_costs(a.points, b.points) ** 2.0
        from scipy.optimize import linear_sum_assignment

        allowed = np.isfinite(cost)
        rows, cols = linear_sum_assignment(np.where(allowed, cost, 1e9))
        assert via_gudhi == pytest.approx(np.sqrt(cost[rows, cols].sum()), rel=1e-6)
