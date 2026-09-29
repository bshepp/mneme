"""Steady-state analysis, tested on systems whose answers are known."""

import numpy as np
import pytest

from mneme.analysis.steady_state import (
    assess_steady_state,
    convergence_rate,
    count_distinct_states,
    pairwise_max_difference,
)


def _relaxation(end, start, tau, times):
    """x(t) = end + (start - end) * exp(-t / tau), one column per cell."""
    end, start = np.asarray(end, float), np.asarray(start, float)
    return end[None, :] + (start - end)[None, :] * np.exp(-times[:, None] / tau)


def _double_well(x0, times, dt=0.01):
    """Overdamped motion in V(x) = (x^2 - 1)^2 / 4: stable at -1 and +1."""
    x = np.array(x0, dtype=float)
    out = np.empty((len(times), len(x)))
    t = 0.0
    for i, target in enumerate(times):
        while t < target - 1e-12:
            x = x + dt * (x - x**3)
            t += dt
        out[i] = x
    return out


class TestConvergenceRate:
    def test_linear_drift_has_constant_rate(self):
        times = np.arange(0.0, 50.0, 5.0)
        values = np.outer(times, [0.2, -0.5])  # cell rates 0.2 and 0.5
        np.testing.assert_allclose(convergence_rate(values, times), 0.5)

    def test_rate_is_independent_of_sampling_interval(self):
        coarse = np.arange(0.0, 100.0, 10.0)
        fine = np.arange(0.0, 100.0, 1.0)
        rc = convergence_rate(np.outer(coarse, [0.3]), coarse)
        rf = convergence_rate(np.outer(fine, [0.3]), fine)
        assert rc[0] == pytest.approx(rf[0])

    @pytest.mark.parametrize(
        "values, times, message",
        [
            (np.zeros(10), None, "shape"),
            (np.zeros((2, 3)), None, "at least 3"),
            (np.full((5, 2), np.nan), None, "NaN"),
            (np.zeros((4, 2)), np.array([0.0, 1.0, 1.0, 2.0]), "increasing"),
        ],
    )
    def test_bad_input_raises(self, values, times, message):
        with pytest.raises(ValueError, match=message):
            convergence_rate(values, times)


class TestAssessSteadyState:
    TAU = 100.0

    def _run(self, duration):
        times = np.arange(0.0, duration + 1.0, 5.0)
        values = _relaxation([-60.0, -40.0], [-10.0, -10.0], self.TAU, times)
        return values, times

    def test_recovers_the_time_constant(self):
        values, times = self._run(400.0)
        report = assess_steady_state(
            values, times, rate_tolerance=1.0, drift_tolerance=1.0
        )
        assert report.time_constant == pytest.approx(self.TAU, rel=0.02)

    def test_remaining_drift_matches_the_true_distance(self):
        values, times = self._run(300.0)
        true_remaining = np.max(np.abs(values[-1] - np.array([-60.0, -40.0])))
        report = assess_steady_state(
            values, times, rate_tolerance=1.0, drift_tolerance=1.0
        )
        assert report.remaining_drift == pytest.approx(true_remaining, rel=0.05)

    def test_short_run_is_not_settled(self):
        values, times = self._run(200.0)  # two time constants: 6.8 mV to go
        report = assess_steady_state(
            values, times, rate_tolerance=0.1, drift_tolerance=0.1
        )
        assert report.settled is False
        assert report.remaining_drift > 5.0

    def test_long_run_is_settled(self):
        values, times = self._run(1200.0)  # twelve time constants
        report = assess_steady_state(
            values, times, rate_tolerance=1e-3, drift_tolerance=0.1
        )
        assert report.settled is True
        assert report.remaining_drift < 0.01

    def test_slow_drift_with_low_rate_is_not_settled(self):
        """A small rate alone must not pass: the run may still be far away."""
        times = np.arange(0.0, 2000.0, 10.0)
        values = _relaxation([-60.0], [-10.0], 5000.0, times)
        report = assess_steady_state(
            values, times, rate_tolerance=0.02, drift_tolerance=0.5
        )
        assert np.all(report.rate < 0.02)      # the rate test alone would pass
        assert report.settled is False
        assert report.remaining_drift > 30.0   # true distance is 33.5

    def test_constant_run_is_settled(self):
        report = assess_steady_state(
            np.full((30, 4), -55.0), rate_tolerance=1e-6, drift_tolerance=1e-6
        )
        assert report.settled is True
        assert report.remaining_drift == 0.0

    def test_steady_linear_drift_is_not_settled(self):
        times = np.arange(0.0, 100.0, 1.0)
        report = assess_steady_state(
            np.outer(times, [0.001]), times, rate_tolerance=0.01, drift_tolerance=1.0
        )
        assert report.settled is False


class TestCountDistinctStates:
    def test_single_state(self):
        rng = np.random.RandomState(0)
        base = rng.uniform(-70, -20, 30)
        states = [base + 0.001 * rng.standard_normal(30) for _ in range(6)]
        found = count_distinct_states(states, threshold=0.05)
        assert found.n_states == 1
        assert found.smallest_between == np.inf
        assert found.largest_within < 0.05

    def test_two_states_with_known_membership(self):
        rng = np.random.RandomState(1)
        a = rng.uniform(-70, -20, 30)
        b = a.copy()
        b[:10] += 8.0
        truth = np.array([0, 1, 0, 0, 1, 1, 0])
        states = [(a, b)[t] + 0.001 * rng.standard_normal(30) for t in truth]
        found = count_distinct_states(states, threshold=0.05)
        assert found.n_states == 2
        np.testing.assert_array_equal(found.labels, truth)
        assert found.largest_within < 0.05 < found.smallest_between
        assert found.smallest_between == pytest.approx(8.0, abs=0.01)

    def test_difference_in_one_cell_is_enough(self):
        a = np.zeros(50)
        b = np.zeros(50)
        b[17] = 1.0
        assert count_distinct_states([a, b], threshold=0.5).n_states == 2

    def test_threshold_must_be_positive(self):
        with pytest.raises(ValueError, match="positive"):
            count_distinct_states([np.zeros(3), np.ones(3)], threshold=0.0)

    def test_pairwise_matrix(self):
        d = pairwise_max_difference([np.array([0.0, 0.0]), np.array([3.0, -4.0])])
        np.testing.assert_allclose(d, [[0.0, 4.0], [4.0, 0.0]])


class TestOnASystemWithKnownMultistability:
    """The double well has exactly two stable states, at -1 and +1."""

    def test_finds_both_wells_and_assigns_runs_correctly(self):
        times = np.arange(0.0, 30.0, 0.5)
        starts = [-1.8, -0.6, -0.05, 0.05, 0.4, 2.5]
        runs = [_double_well([s, s, s], times) for s in starts]

        for run in runs:
            report = assess_steady_state(
                run, times, rate_tolerance=1e-4, drift_tolerance=1e-3
            )
            assert report.settled is True

        found = count_distinct_states([run[-1] for run in runs], threshold=0.01)
        assert found.n_states == 2
        np.testing.assert_array_equal(found.labels, [0, 0, 0, 1, 1, 1])
        assert found.smallest_between == pytest.approx(2.0, abs=1e-3)

    def test_single_well_gives_one_state(self):
        times = np.arange(0.0, 30.0, 0.5)
        runs = [_double_well([s, s], times) for s in (0.1, 0.5, 1.5, 3.0)]
        found = count_distinct_states([run[-1] for run in runs], threshold=0.01)
        assert found.n_states == 1

    def test_unsettled_runs_are_flagged(self):
        """Started next to the ridge, a short run has not yet chosen a well."""
        times = np.arange(0.0, 3.0, 0.1)
        run = _double_well([0.01, 0.01], times)
        report = assess_steady_state(
            run, times, rate_tolerance=1e-4, drift_tolerance=1e-3
        )
        assert report.settled is False
