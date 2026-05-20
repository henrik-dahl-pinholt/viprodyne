import numpy as np
import pytest
from scipy.linalg import expm

from viprodyne.core.rate_edges import wrap_column_generator
from viprodyne.core.simulation import (
    CTMCPath,
    generate_ms2_signal,
    proximal_ms2_kernel,
    sample_ctmc_path,
    sample_loading_events,
    simulate_ms2_trajectory,
    stationary_distribution,
)


def two_state_generator(kon: float, koff: float) -> np.ndarray:
    # Off-diagonal row-major order: Q[0, 1] = koff, Q[1, 0] = kon.
    return wrap_column_generator(np.array([koff, kon]), n_states=2)


def test_stationary_distribution_two_state_column_generator():
    kon = 0.7
    koff = 0.2
    generator = two_state_generator(kon, koff)

    stationary = stationary_distribution(generator)

    assert stationary.dtype == np.float32
    np.testing.assert_allclose(stationary, [koff / (kon + koff), kon / (kon + koff)])
    np.testing.assert_allclose(generator @ stationary, np.zeros(2), atol=1e-7)


def test_ctmc_path_summaries_use_to_from_transition_counts():
    path = CTMCPath(
        times=np.array([0.0, 0.25, 0.75, 1.0]),
        states=np.array([0, 1, 0, 1]),
        stop_time=1.5,
    )

    np.testing.assert_array_equal(path.state_at([0.0, 0.3, 1.25]), [0, 1, 1])
    assert path.times.dtype == np.float32
    assert path.durations.dtype == np.float32
    np.testing.assert_allclose(path.durations, [0.25, 0.5, 0.25, 0.5])
    dwell = path.dwell_times(n_states=2)
    counts = path.transition_counts(n_states=2)
    assert dwell.dtype == np.float32
    assert counts.dtype == np.float32
    np.testing.assert_allclose(dwell, [0.5, 1.0])
    np.testing.assert_allclose(counts, [[0.0, 1.0], [2.0, 0.0]])


def test_sample_ctmc_path_final_state_matches_two_state_analytic_distribution():
    kon = 0.8
    koff = 0.3
    generator = two_state_generator(kon, koff)
    initial = np.array([1.0, 0.0])
    stop_time = 1.4
    expected = expm(generator * stop_time) @ initial

    rng = np.random.default_rng(2026)
    n_samples = 20_000
    final_states = np.empty(n_samples, dtype=int)
    for sample in range(n_samples):
        final_states[sample] = sample_ctmc_path(
            generator,
            stop_time=stop_time,
            rng=rng,
            initial_probabilities=initial,
        ).state_at(stop_time)
    observed = np.bincount(final_states, minlength=2) / n_samples

    np.testing.assert_allclose(observed, expected, atol=0.012)


def test_sample_ctmc_path_absorbing_state_has_single_segment():
    generator = np.array([[0.0, 0.5], [0.0, -0.5]])

    path = sample_ctmc_path(generator, stop_time=10.0, initial_state=0, rng=np.random.default_rng(1))

    np.testing.assert_array_equal(path.times, [0.0])
    np.testing.assert_array_equal(path.states, [0])
    np.testing.assert_array_equal(path.state_at([0.0, 10.0]), [0, 0])


def test_loading_events_are_sorted_and_within_active_segments():
    path = CTMCPath(times=np.array([0.0, 2.0]), states=np.array([0, 1]), stop_time=3.0)

    empty = sample_loading_events(path, loading_rates=np.array([0.0, 0.0]), rng=np.random.default_rng(2))
    events = sample_loading_events(path, loading_rates=np.array([4.0, 7.0]), rng=np.random.default_rng(2))

    assert empty.dtype == np.float32
    assert events.dtype == np.float32
    assert empty.size == 0
    assert np.all(np.diff(events) >= 0)
    assert np.all(events >= 0.0)
    assert np.all(events <= 3.0)


def test_proximal_kernel_and_ms2_signal_without_noise_are_exact():
    offsets = np.array([-1.0, 0.0, 0.5, 1.0, 2.5, 3.1])
    kernel_values = proximal_ms2_kernel(offsets, rise_time=1.0, plateau_time=2.0, max_intensity=4.0)
    assert kernel_values.dtype == np.float32
    np.testing.assert_allclose(kernel_values, [0.0, 0.0, 2.0, 4.0, 4.0, 0.0])

    sampling_times = np.array([0.0, 1.0, 2.0])
    loading_times = np.array([0.0, 1.0])
    clean, noisy = generate_ms2_signal(
        sampling_times,
        loading_times,
        kernel=lambda t: proximal_ms2_kernel(t, rise_time=1.0, plateau_time=1.0),
        noise_std=0.0,
        rng=np.random.default_rng(3),
    )

    assert clean.dtype == np.float32
    assert noisy.dtype == np.float32
    np.testing.assert_allclose(clean, [0.0, 1.0, 2.0])
    np.testing.assert_allclose(noisy, clean)


def test_simulate_ms2_trajectory_shapes_and_hidden_variables():
    rng = np.random.default_rng(4)
    sampling_times = np.linspace(0.0, 3.0, 7)
    trajectory = simulate_ms2_trajectory(
        generator=two_state_generator(kon=0.5, koff=0.2),
        loading_rates=np.array([0.0, 2.0]),
        sampling_times=sampling_times,
        kernel=lambda t: proximal_ms2_kernel(t, rise_time=0.5, plateau_time=1.0),
        noise_std=0.1,
        rng=rng,
        initial_state=1,
        pad_time=1.5,
    )

    assert trajectory.clean_signal.shape == sampling_times.shape
    assert trajectory.noisy_signal.shape == sampling_times.shape
    assert trajectory.clean_signal.dtype == np.float32
    assert trajectory.noisy_signal.dtype == np.float32
    assert trajectory.loading_times.dtype == np.float32
    assert trajectory.promoter_path.stop_time == pytest.approx(4.5)
    assert np.all(trajectory.loading_times >= 0.0)
    assert np.all(trajectory.loading_times <= trajectory.promoter_path.stop_time)
