import numpy as np
import pytest
from scipy.linalg import expm

from viprodyne.core.rate_edges import wrap_column_generator
from viprodyne.core.tilted_ctmc import TiltedCTMC


def two_state_generator(kon: float, koff: float) -> np.ndarray:
    # Column convention: Q[1, 0] is 0->1 and Q[0, 1] is 1->0.
    return wrap_column_generator(np.array([koff, kon]), n_states=2)


def test_two_state_forward_matches_analytic_solution():
    kon = 0.7
    koff = 0.2
    total = kon + koff
    time_grid = np.linspace(0.0, 3.0, 7)
    solution = TiltedCTMC(
        generator=two_state_generator(kon, koff),
        time_grid=time_grid,
        initial_probabilities=np.array([1.0, 0.0]),
    ).solve()

    expected_on = kon / total * (1.0 - np.exp(-total * time_grid))
    expected_off = 1.0 - expected_on
    expected = np.stack([expected_off, expected_on], axis=-1)

    assert solution.alpha.dtype == np.float32
    assert solution.log_partition[0] == pytest.approx(0.0, abs=2e-7)
    np.testing.assert_allclose(solution.alpha[0], expected, rtol=3e-6, atol=3e-7)
    np.testing.assert_allclose(solution.posterior[0], expected, rtol=3e-6, atol=3e-7)


def test_two_state_expected_occupancy_and_jumps_match_analytic_integrals():
    kon = 0.8
    koff = 0.3
    total = kon + koff
    duration = 2.5
    solution = TiltedCTMC(
        generator=two_state_generator(kon, koff),
        time_grid=np.array([0.0, duration]),
        initial_probabilities=np.array([1.0, 0.0]),
    ).solve()

    off_integral = (koff / total) * duration + (kon / total) * (
        1.0 - np.exp(-total * duration)
    ) / total
    on_integral = duration - off_integral
    occupancy = solution.expected_occupancy()[0, 0]
    jumps = solution.expected_jumps()[0, 0]

    np.testing.assert_allclose(occupancy, [off_integral, on_integral], rtol=3e-6)
    assert jumps[1, 0] == pytest.approx(kon * off_integral, rel=3e-6)
    assert jumps[0, 1] == pytest.approx(koff * on_integral, rel=3e-6)
    assert jumps[0, 0] == 0.0
    assert jumps[1, 1] == 0.0


def test_constant_potential_partition_matches_matrix_exponential():
    generator = two_state_generator(kon=0.4, koff=0.6)
    potential = np.array([0.2, -0.5])
    initial = np.array([0.3, 0.7])
    duration = 1.7
    solution = TiltedCTMC(
        generator=generator,
        time_grid=np.array([0.0, duration]),
        initial_probabilities=initial,
        potentials=potential,
    ).solve()

    tilted = generator + np.diag(potential)
    expected_z = np.sum(expm(tilted * duration) @ initial)

    assert solution.log_partition[0] == pytest.approx(np.log(expected_z), rel=3e-6)
    np.testing.assert_allclose(
        solution.marginal_at(0.0),
        solution.posterior[:, 0],
        rtol=3e-6,
        atol=3e-7,
    )


def test_scaled_forward_backward_avoids_float32_underflow():
    n_intervals = 400
    time_grid = np.linspace(0.0, 200.0, n_intervals + 1, dtype=np.float32)
    generator = np.array([[0.0]], dtype=np.float32)
    potential = np.full((n_intervals, 1), -5.0, dtype=np.float32)

    solution = TiltedCTMC(
        generator=generator,
        time_grid=time_grid,
        initial_probabilities=np.array([1.0], dtype=np.float32),
        potentials=potential,
    ).solve()

    assert np.isfinite(np.asarray(solution.log_partition)[0])
    assert solution.log_partition[0] == pytest.approx(-1000.0, rel=1e-6)
    np.testing.assert_allclose(solution.posterior[0], np.ones((n_intervals + 1, 1)))
    np.testing.assert_allclose(
        solution.expected_occupancy()[0, :, 0],
        np.diff(time_grid),
        rtol=2e-5,
    )


def test_piecewise_batch_shapes_and_transition_columns():
    generator = two_state_generator(kon=0.5, koff=0.1)
    time_grid = np.array([0.0, 1.0, 1.5])
    initial = np.array([[1.0, 0.0], [0.0, 1.0]])
    potentials = np.array([[[0.0, 0.0], [0.1, -0.2]], [[0.2, 0.0], [0.0, 0.3]]])

    solution = TiltedCTMC(
        generator=generator,
        time_grid=time_grid,
        initial_probabilities=initial,
        potentials=potentials,
    ).solve()

    assert solution.posterior.shape == (2, 3, 2)
    assert solution.expected_occupancy().shape == (2, 2, 2)
    assert solution.expected_jumps().shape == (2, 2, 2, 2)
    transitions = solution.posterior_grid_transition_matrices()
    np.testing.assert_allclose(np.sum(transitions, axis=2), np.ones((2, 2, 2)), rtol=3e-6)
