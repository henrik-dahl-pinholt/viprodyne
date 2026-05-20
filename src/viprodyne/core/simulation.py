"""Prior simulation utilities for column-sum-zero CTMC promoter models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from viprodyne.core.rate_edges import validate_column_generator


@dataclass(frozen=True)
class CTMCPath:
    """Piecewise-constant CTMC state path.

    ``times`` stores state-entry times. The first entry is always zero. The state
    in ``states[i]`` is active on ``[times[i], times[i + 1])`` or until
    ``stop_time`` for the last entry.
    """

    times: np.ndarray
    states: np.ndarray
    stop_time: float

    def __post_init__(self) -> None:
        times = np.asarray(self.times, dtype=float)
        states = np.asarray(self.states, dtype=int)
        if times.ndim != 1 or states.ndim != 1:
            raise ValueError("times and states must be one-dimensional.")
        if times.size != states.size:
            raise ValueError("times and states must have the same length.")
        if times.size == 0 or not np.isclose(times[0], 0.0):
            raise ValueError("times must start at zero.")
        if self.stop_time < 0:
            raise ValueError("stop_time must be non-negative.")
        if np.any(np.diff(times) <= 0):
            raise ValueError("times must be strictly increasing.")
        if times[-1] > self.stop_time:
            raise ValueError("state-entry times cannot exceed stop_time.")
        object.__setattr__(self, "times", times)
        object.__setattr__(self, "states", states)
        object.__setattr__(self, "stop_time", float(self.stop_time))

    @property
    def segment_ends(self) -> np.ndarray:
        """Return end time for each state segment."""
        return np.concatenate([self.times[1:], np.array([self.stop_time])])

    @property
    def durations(self) -> np.ndarray:
        """Return duration of each state segment."""
        return self.segment_ends - self.times

    def state_at(self, query_times: np.ndarray | float) -> np.ndarray:
        """Return active state at one or more times."""
        query_times = np.asarray(query_times, dtype=float)
        if np.any(query_times < 0) or np.any(query_times > self.stop_time):
            raise ValueError("query_times must lie within [0, stop_time].")
        indices = np.searchsorted(self.times, query_times, side="right") - 1
        return self.states[indices]

    def dwell_times(self, n_states: int | None = None) -> np.ndarray:
        """Return total dwell time in each state."""
        if n_states is None:
            n_states = int(np.max(self.states)) + 1
        dwell = np.zeros(n_states, dtype=float)
        np.add.at(dwell, self.states, self.durations)
        return dwell

    def transition_counts(self, n_states: int | None = None) -> np.ndarray:
        """Return transition counts using ``counts[to_state, from_state]``."""
        if n_states is None:
            n_states = int(np.max(self.states)) + 1
        counts = np.zeros((n_states, n_states), dtype=float)
        from_states = self.states[:-1]
        to_states = self.states[1:]
        np.add.at(counts, (to_states, from_states), 1.0)
        return counts


@dataclass(frozen=True)
class MS2Trajectory:
    """A simulated MS2 trajectory with hidden promoter and loading variables."""

    promoter_path: CTMCPath
    loading_times: np.ndarray
    clean_signal: np.ndarray
    noisy_signal: np.ndarray


def stationary_distribution(generator: np.ndarray) -> np.ndarray:
    """Compute the stationary distribution of a column-sum-zero CTMC."""
    generator = validate_column_generator(generator)
    n_states = generator.shape[0]
    matrix = np.vstack([generator, np.ones((1, n_states))])
    rhs = np.zeros(n_states + 1, dtype=float)
    rhs[-1] = 1.0
    stationary, *_ = np.linalg.lstsq(matrix, rhs, rcond=None)
    stationary = np.clip(stationary, 0.0, np.inf)
    total = np.sum(stationary)
    if total <= 0:
        raise FloatingPointError("stationary distribution has no positive mass.")
    return stationary / total


def sample_ctmc_path(
    generator: np.ndarray,
    stop_time: float,
    rng: np.random.Generator | None = None,
    initial_probabilities: np.ndarray | None = None,
    initial_state: int | None = None,
) -> CTMCPath:
    """Sample a CTMC state path by Gillespie simulation."""
    generator = validate_column_generator(generator)
    stop_time = float(stop_time)
    if stop_time < 0:
        raise ValueError("stop_time must be non-negative.")
    rng = np.random.default_rng() if rng is None else rng
    n_states = generator.shape[0]
    if initial_state is None:
        if initial_probabilities is None:
            initial_probabilities = stationary_distribution(generator)
        initial_probabilities = _normalize_probabilities(initial_probabilities, n_states)
        current_state = int(rng.choice(n_states, p=initial_probabilities))
    else:
        if not 0 <= initial_state < n_states:
            raise ValueError("initial_state is outside [0, n_states).")
        current_state = int(initial_state)

    times = [0.0]
    states = [current_state]
    time = 0.0
    while time < stop_time:
        exit_rate = -float(generator[current_state, current_state])
        if exit_rate <= 0.0:
            break
        wait_time = float(rng.exponential(1.0 / exit_rate))
        if time + wait_time > stop_time:
            break
        time += wait_time
        rates = generator[:, current_state].copy()
        rates[current_state] = 0.0
        next_state = int(rng.choice(n_states, p=rates / np.sum(rates)))
        times.append(time)
        states.append(next_state)
        current_state = next_state
    return CTMCPath(times=np.asarray(times), states=np.asarray(states), stop_time=stop_time)


def sample_loading_events(
    path: CTMCPath,
    loading_rates: np.ndarray,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Sample Poisson loading events conditional on a promoter path."""
    rng = np.random.default_rng() if rng is None else rng
    loading_rates = np.asarray(loading_rates, dtype=float)
    if loading_rates.ndim != 1:
        raise ValueError("loading_rates must be one-dimensional.")
    if np.any(loading_rates < 0):
        raise ValueError("loading_rates must be non-negative.")
    if np.max(path.states) >= loading_rates.size:
        raise ValueError("loading_rates must provide one rate per visited state.")
    intensities = loading_rates[path.states] * path.durations
    event_counts = rng.poisson(intensities)
    if np.sum(event_counts) == 0:
        return np.empty(0, dtype=float)
    segment_ids = np.repeat(np.arange(event_counts.size), event_counts)
    uniforms = rng.uniform(size=int(np.sum(event_counts)))
    event_times = path.times[segment_ids] + path.durations[segment_ids] * uniforms
    return np.sort(event_times)


def proximal_ms2_kernel(
    time_offsets: np.ndarray,
    rise_time: float,
    plateau_time: float,
    max_intensity: float = 1.0,
) -> np.ndarray:
    """Evaluate a ramp-then-plateau MS2 kernel."""
    time_offsets = np.asarray(time_offsets, dtype=float)
    if rise_time <= 0 or plateau_time < 0:
        raise ValueError("rise_time must be positive and plateau_time non-negative.")
    out = np.zeros_like(time_offsets, dtype=float)
    rising = (time_offsets >= 0.0) & (time_offsets < rise_time)
    plateau = (time_offsets >= rise_time) & (time_offsets <= rise_time + plateau_time)
    out[rising] = max_intensity * time_offsets[rising] / rise_time
    out[plateau] = max_intensity
    return out


def generate_ms2_signal(
    sampling_times: np.ndarray,
    loading_times: np.ndarray,
    kernel: Callable[[np.ndarray], np.ndarray],
    noise_std: float = 0.0,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate clean and noisy MS2 signals from loading events."""
    sampling_times = np.asarray(sampling_times, dtype=float)
    loading_times = np.asarray(loading_times, dtype=float)
    if noise_std < 0:
        raise ValueError("noise_std must be non-negative.")
    if loading_times.size == 0:
        clean = np.zeros_like(sampling_times, dtype=float)
    else:
        clean = kernel(sampling_times[:, None] - loading_times[None, :]).sum(axis=-1)
    rng = np.random.default_rng() if rng is None else rng
    noise = rng.normal(0.0, noise_std, size=sampling_times.shape)
    return clean, clean + noise


def simulate_ms2_trajectory(
    generator: np.ndarray,
    loading_rates: np.ndarray,
    sampling_times: np.ndarray,
    kernel: Callable[[np.ndarray], np.ndarray],
    noise_std: float = 0.0,
    rng: np.random.Generator | None = None,
    initial_probabilities: np.ndarray | None = None,
    initial_state: int | None = None,
    pad_time: float = 0.0,
) -> MS2Trajectory:
    """Simulate promoter states, Pol2 loading events, and an MS2 trace."""
    rng = np.random.default_rng() if rng is None else rng
    sampling_times = np.asarray(sampling_times, dtype=float)
    if sampling_times.ndim != 1 or sampling_times.size == 0:
        raise ValueError("sampling_times must be a non-empty one-dimensional array.")
    stop_time = float(np.max(sampling_times) + pad_time)
    path = sample_ctmc_path(
        generator,
        stop_time=stop_time,
        rng=rng,
        initial_probabilities=initial_probabilities,
        initial_state=initial_state,
    )
    loading_times = sample_loading_events(path, loading_rates, rng)
    clean, noisy = generate_ms2_signal(sampling_times, loading_times, kernel, noise_std, rng)
    return MS2Trajectory(
        promoter_path=path,
        loading_times=loading_times,
        clean_signal=clean,
        noisy_signal=noisy,
    )


def _normalize_probabilities(probabilities: np.ndarray, n_states: int) -> np.ndarray:
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.shape != (n_states,):
        raise ValueError("initial_probabilities must have shape (n_states,).")
    if np.any(probabilities < 0):
        raise ValueError("initial_probabilities must be non-negative.")
    total = np.sum(probabilities)
    if total <= 0:
        raise ValueError("initial_probabilities must have positive mass.")
    return probabilities / total
