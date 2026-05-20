"""Tilted continuous-time Markov chains with column-sum-zero generators."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import expm, expm_frechet

from viprodyne.core.rate_edges import validate_column_generator


@dataclass(frozen=True)
class TiltedCTMCSolution:
    """Forward-backward quantities for a tilted CTMC."""

    time_grid: np.ndarray
    generators: np.ndarray
    potentials: np.ndarray
    initial_probabilities: np.ndarray
    transition_matrices: np.ndarray
    alpha: np.ndarray
    beta: np.ndarray
    log_partition: np.ndarray

    @property
    def posterior(self) -> np.ndarray:
        """State posterior on grid points, shape ``(batch, n_times, n_states)``."""
        z = np.exp(self.log_partition)[:, None, None]
        return self.alpha * self.beta / z

    def marginal_at(self, time: float) -> np.ndarray:
        """Interpolate the state posterior at a single time."""
        interval = _interval_index(self.time_grid, float(time))
        left = self.time_grid[interval]
        right = self.time_grid[interval + 1]
        dt_left = float(time - left)
        dt_right = float(right - time)
        out = []
        z = np.exp(self.log_partition)
        for batch in range(self.generators.shape[0]):
            generator = self.generators[batch, interval]
            potential = self.potentials[batch, interval]
            tilted = generator + np.diag(potential)
            alpha_mid = expm(tilted * dt_left) @ self.alpha[batch, interval]
            beta_mid = expm(tilted.T * dt_right) @ self.beta[batch, interval + 1]
            out.append(alpha_mid * beta_mid / z[batch])
        return np.asarray(out)

    def expected_occupancy(self) -> np.ndarray:
        """Expected time spent in each state per interval.

        Returns an array with shape ``(batch, n_intervals, n_states)`` and units of
        time. This is exact for piecewise-constant generators and potentials.
        """
        batch_size, n_intervals, n_states, _ = self.generators.shape
        out = np.zeros((batch_size, n_intervals, n_states), dtype=float)
        z = np.exp(self.log_partition)
        for batch in range(batch_size):
            for interval in range(n_intervals):
                dt = self.time_grid[interval + 1] - self.time_grid[interval]
                tilted = self.generators[batch, interval] + np.diag(
                    self.potentials[batch, interval]
                )
                matrix = tilted * dt
                for state in range(n_states):
                    direction = np.zeros_like(tilted)
                    direction[state, state] = dt
                    frechet = expm_frechet(matrix, direction, compute_expm=False)
                    out[batch, interval, state] = (
                        self.beta[batch, interval + 1]
                        @ frechet
                        @ self.alpha[batch, interval]
                        / z[batch]
                    )
        return out

    def expected_jumps(self) -> np.ndarray:
        """Expected transition counts per interval for ``Q[to_state, from_state]``.

        The diagonal entries are always zero. The returned counts are integrated
        over each interval rather than multiplied by a downstream quadrature rule.
        """
        batch_size, n_intervals, n_states, _ = self.generators.shape
        out = np.zeros((batch_size, n_intervals, n_states, n_states), dtype=float)
        z = np.exp(self.log_partition)
        for batch in range(batch_size):
            for interval in range(n_intervals):
                dt = self.time_grid[interval + 1] - self.time_grid[interval]
                generator = self.generators[batch, interval]
                tilted = generator + np.diag(self.potentials[batch, interval])
                matrix = tilted * dt
                for to_state in range(n_states):
                    for from_state in range(n_states):
                        if to_state == from_state or generator[to_state, from_state] == 0.0:
                            continue
                        direction = np.zeros_like(tilted)
                        direction[to_state, from_state] = generator[to_state, from_state] * dt
                        frechet = expm_frechet(matrix, direction, compute_expm=False)
                        out[batch, interval, to_state, from_state] = (
                            self.beta[batch, interval + 1]
                            @ frechet
                            @ self.alpha[batch, interval]
                            / z[batch]
                        )
        return out

    def expected_jump_density(self) -> np.ndarray:
        """Expected jump counts divided by interval durations."""
        dt = np.diff(self.time_grid)[None, :, None, None]
        return self.expected_jumps() / dt

    def posterior_grid_transition_matrices(self) -> np.ndarray:
        """Return posterior transitions between grid points.

        Entry ``P[to_state, from_state]`` is the probability of the next grid
        state being ``to_state`` given current grid state ``from_state``.
        """
        numerator = self.transition_matrices * self.beta[:, 1:, :, None]
        denominator = self.beta[:, :-1, None, :]
        with np.errstate(divide="ignore", invalid="ignore"):
            transitions = numerator / denominator
        return np.nan_to_num(transitions, nan=0.0, posinf=0.0, neginf=0.0)

    def sample_grid_paths(
        self,
        n_samples: int,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """Sample posterior states on the time grid.

        Returns integer states with shape ``(batch, n_samples, n_times)``.
        """
        if n_samples <= 0:
            raise ValueError("n_samples must be positive.")
        rng = np.random.default_rng() if rng is None else rng
        batch_size, n_intervals, n_states, _ = self.generators.shape
        transitions = self.posterior_grid_transition_matrices()
        initial = self.posterior[:, 0]
        samples = np.zeros((batch_size, n_samples, n_intervals + 1), dtype=int)
        for batch in range(batch_size):
            samples[batch, :, 0] = rng.choice(n_states, size=n_samples, p=initial[batch])
            for interval in range(n_intervals):
                for sample in range(n_samples):
                    from_state = samples[batch, sample, interval]
                    probs = transitions[batch, interval, :, from_state]
                    probs = probs / np.sum(probs)
                    samples[batch, sample, interval + 1] = rng.choice(n_states, p=probs)
        return samples


class TiltedCTMC:
    """Solve a tilted CTMC with column-sum-zero generator convention.

    The generator convention is ``Q[to_state, from_state]`` with columns summing
    to zero. The tilted interval operator is

    ``expm((Q + diag(potential)) * dt)``.
    """

    def __init__(
        self,
        generator: np.ndarray,
        time_grid: np.ndarray,
        initial_probabilities: np.ndarray,
        potentials: np.ndarray | None = None,
    ) -> None:
        self.time_grid = _validate_time_grid(time_grid)
        self.n_intervals = self.time_grid.size - 1
        initial = _normalize_initial_probabilities(initial_probabilities)
        n_states = initial.shape[-1]
        generators = _normalize_generators(generator, self.n_intervals, n_states)
        potentials = _normalize_potentials(potentials, self.n_intervals, n_states)
        batch_size = max(initial.shape[0], generators.shape[0], potentials.shape[0])
        self.initial_probabilities = np.broadcast_to(initial, (batch_size, n_states)).copy()
        self.generators = np.broadcast_to(
            generators,
            (batch_size, self.n_intervals, n_states, n_states),
        ).copy()
        self.potentials = np.broadcast_to(
            potentials,
            (batch_size, self.n_intervals, n_states),
        ).copy()

    def solve(self) -> TiltedCTMCSolution:
        """Run forward-backward recursions."""
        batch_size, n_intervals, n_states, _ = self.generators.shape
        transition_matrices = np.zeros_like(self.generators)
        alpha = np.zeros((batch_size, n_intervals + 1, n_states), dtype=float)
        beta = np.zeros_like(alpha)
        alpha[:, 0] = self.initial_probabilities
        beta[:, -1] = 1.0

        for interval in range(n_intervals):
            dt = self.time_grid[interval + 1] - self.time_grid[interval]
            for batch in range(batch_size):
                tilted = self.generators[batch, interval] + np.diag(
                    self.potentials[batch, interval]
                )
                transition_matrices[batch, interval] = expm(tilted * dt)
                alpha[batch, interval + 1] = (
                    transition_matrices[batch, interval] @ alpha[batch, interval]
                )

        for interval in range(n_intervals - 1, -1, -1):
            for batch in range(batch_size):
                beta[batch, interval] = (
                    transition_matrices[batch, interval].T @ beta[batch, interval + 1]
                )

        partition = np.sum(alpha[:, -1], axis=-1)
        if np.any(partition <= 0) or np.any(~np.isfinite(partition)):
            raise FloatingPointError("CTMC partition function is not positive and finite.")
        return TiltedCTMCSolution(
            time_grid=self.time_grid,
            generators=self.generators,
            potentials=self.potentials,
            initial_probabilities=self.initial_probabilities,
            transition_matrices=transition_matrices,
            alpha=alpha,
            beta=beta,
            log_partition=np.log(partition),
        )


def _validate_time_grid(time_grid: np.ndarray) -> np.ndarray:
    time_grid = np.asarray(time_grid, dtype=float)
    if time_grid.ndim != 1 or time_grid.size < 2:
        raise ValueError("time_grid must be one-dimensional with at least two points.")
    if np.any(np.diff(time_grid) <= 0):
        raise ValueError("time_grid must be strictly increasing.")
    return time_grid


def _normalize_initial_probabilities(initial_probabilities: np.ndarray) -> np.ndarray:
    initial = np.asarray(initial_probabilities, dtype=float)
    if initial.ndim == 1:
        initial = initial[None, :]
    if initial.ndim != 2:
        raise ValueError("initial_probabilities must have shape (states,) or (batch, states).")
    if np.any(initial < 0):
        raise ValueError("initial_probabilities must be non-negative.")
    total = np.sum(initial, axis=-1, keepdims=True)
    if np.any(total <= 0):
        raise ValueError("initial_probabilities must have positive mass.")
    return initial / total


def _normalize_generators(
    generator: np.ndarray,
    n_intervals: int,
    n_states: int,
) -> np.ndarray:
    generator = np.asarray(generator, dtype=float)
    if generator.shape[-2:] != (n_states, n_states):
        raise ValueError("generator state dimensions do not match initial_probabilities.")
    if generator.ndim == 2:
        generator = generator[None, None, :, :]
    elif generator.ndim == 3:
        generator = generator[None, :, :, :]
    elif generator.ndim != 4:
        raise ValueError("generator must have shape (S,S), (T,S,S), or (B,T,S,S).")
    if generator.shape[1] not in (1, n_intervals):
        raise ValueError("generator interval dimension must be 1 or len(time_grid) - 1.")
    if generator.shape[1] == 1:
        generator = np.broadcast_to(
            generator,
            (generator.shape[0], n_intervals, n_states, n_states),
        )
    for matrix in generator.reshape((-1, n_states, n_states)):
        validate_column_generator(matrix)
    return generator


def _normalize_potentials(
    potentials: np.ndarray | None,
    n_intervals: int,
    n_states: int,
) -> np.ndarray:
    if potentials is None:
        return np.zeros((1, n_intervals, n_states), dtype=float)
    potentials = np.asarray(potentials, dtype=float)
    if potentials.shape[-1] != n_states:
        raise ValueError("potential state dimension does not match initial_probabilities.")
    if potentials.ndim == 1:
        potentials = potentials[None, None, :]
    elif potentials.ndim == 2:
        if potentials.shape[0] == n_intervals:
            potentials = potentials[None, :, :]
        else:
            potentials = potentials[:, None, :]
    elif potentials.ndim != 3:
        raise ValueError("potentials must have shape (S,), (T,S), (B,S), or (B,T,S).")
    if potentials.shape[1] not in (1, n_intervals):
        raise ValueError("potential interval dimension must be 1 or len(time_grid) - 1.")
    if potentials.shape[1] == 1:
        potentials = np.broadcast_to(potentials, (potentials.shape[0], n_intervals, n_states))
    return potentials


def _interval_index(time_grid: np.ndarray, time: float) -> int:
    if not time_grid[0] <= time <= time_grid[-1]:
        raise ValueError("time is outside the time grid.")
    if np.isclose(time, time_grid[-1]):
        return time_grid.size - 2
    return int(np.searchsorted(time_grid, time, side="right") - 1)
