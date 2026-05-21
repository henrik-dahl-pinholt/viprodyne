"""Tilted continuous-time Markov chains with column-sum-zero generators."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from viprodyne.core.rate_edges import validate_column_generator


@dataclass(frozen=True)
class TiltedCTMCSolution:
    """Forward-backward quantities for a tilted CTMC."""

    time_grid: jax.Array
    generators: jax.Array
    potentials: jax.Array
    initial_probabilities: jax.Array
    transition_matrices: jax.Array
    transition_scales: jax.Array
    alpha: jax.Array
    beta: jax.Array
    log_partition: jax.Array

    @property
    def posterior(self) -> jax.Array:
        """State posterior on grid points, shape ``(batch, n_times, n_states)``."""
        return self.alpha * self.beta

    def marginal_at(self, time: float) -> jax.Array:
        """Interpolate the state posterior at a single time."""
        interval = _interval_index(np.asarray(self.time_grid), float(time))
        left = self.time_grid[interval]
        right = self.time_grid[interval + 1]
        dt_left = jnp.asarray(float(time), dtype=jnp.float32) - left
        dt_right = right - jnp.asarray(float(time), dtype=jnp.float32)
        tilted = _tilted_generators(self.generators[:, interval], self.potentials[:, interval])

        def propagate_left(matrix, alpha_left):
            return jax.scipy.linalg.expm(matrix * dt_left) @ alpha_left

        def propagate_right(matrix, beta_right):
            return jax.scipy.linalg.expm(matrix.T * dt_right) @ beta_right

        alpha_mid = jax.vmap(propagate_left)(tilted, self.alpha[:, interval])
        beta_mid = jax.vmap(propagate_right)(tilted, self.beta[:, interval + 1])
        scale = self.transition_scales[:, interval, None]
        return alpha_mid * beta_mid / scale

    def expected_occupancy(self) -> jax.Array:
        """Expected time spent in each state per interval.

        Returns an array with shape ``(batch, n_intervals, n_states)`` and units of
        time. This is exact for piecewise-constant generators and potentials.
        """
        return _expected_occupancy_kernel(
            self.generators,
            self.potentials,
            self.transition_scales,
            self.alpha,
            self.beta,
            self.time_grid,
        )

    def expected_jumps(self) -> jax.Array:
        """Expected transition counts per interval for ``Q[to_state, from_state]``.

        The diagonal entries are always zero. The returned counts are integrated
        over each interval rather than multiplied by a downstream quadrature rule.
        """
        return _expected_jumps_kernel(
            self.generators,
            self.potentials,
            self.transition_scales,
            self.alpha,
            self.beta,
            self.time_grid,
        )

    def expected_jump_density(self) -> jax.Array:
        """Expected jump counts divided by interval durations."""
        dt = jnp.diff(self.time_grid)[None, :, None, None]
        return self.expected_jumps() / dt

    def posterior_grid_transition_matrices(self) -> jax.Array:
        """Return posterior transitions between grid points.

        Entry ``P[to_state, from_state]`` is the probability of the next grid
        state being ``to_state`` given current grid state ``from_state``.
        """
        numerator = self.transition_matrices * self.beta[:, 1:, :, None]
        denominator = self.transition_scales[:, :, None, None] * self.beta[:, :-1, None, :]
        transitions = numerator / denominator
        return jnp.nan_to_num(transitions, nan=0.0, posinf=0.0, neginf=0.0)

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
        transitions = np.asarray(self.posterior_grid_transition_matrices())
        initial = np.asarray(self.posterior[:, 0])
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
        time_grid = _validate_time_grid(time_grid)
        n_intervals = time_grid.size - 1
        initial = _normalize_initial_probabilities(initial_probabilities)
        n_states = initial.shape[-1]
        generators = _normalize_generators(generator, n_intervals, n_states)
        potentials = _normalize_potentials(potentials, n_intervals, n_states)
        batch_size = max(initial.shape[0], generators.shape[0], potentials.shape[0])

        self.time_grid = jnp.asarray(time_grid, dtype=jnp.float32)
        self.n_intervals = n_intervals
        self.initial_probabilities = jnp.asarray(
            np.broadcast_to(initial, (batch_size, n_states)).copy(),
            dtype=jnp.float32,
        )
        self.generators = jnp.asarray(
            np.broadcast_to(
                generators,
                (batch_size, n_intervals, n_states, n_states),
            ).copy(),
            dtype=jnp.float32,
        )
        self.potentials = jnp.asarray(
            np.broadcast_to(
                potentials,
                (batch_size, n_intervals, n_states),
            ).copy(),
            dtype=jnp.float32,
        )

    def solve(self) -> TiltedCTMCSolution:
        """Run forward-backward recursions."""
        transition_matrices, transition_scales, alpha, beta, log_partition = _solve_forward_backward(
            self.generators,
            self.potentials,
            self.initial_probabilities,
            self.time_grid,
        )
        log_partition_host = np.asarray(log_partition)
        if np.any(~np.isfinite(log_partition_host)):
            raise FloatingPointError("CTMC partition function is not positive and finite.")
        return TiltedCTMCSolution(
            time_grid=self.time_grid,
            generators=self.generators,
            potentials=self.potentials,
            initial_probabilities=self.initial_probabilities,
            transition_matrices=transition_matrices,
            transition_scales=transition_scales,
            alpha=alpha,
            beta=beta,
            log_partition=log_partition,
        )


@jax.jit
def _solve_forward_backward(
    generators: jax.Array,
    potentials: jax.Array,
    initial_probabilities: jax.Array,
    time_grid: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    dt = jnp.diff(time_grid)

    def interval_transition(generator, potential, interval_dt):
        tilted = _tilted_generator(generator, potential)
        return jax.scipy.linalg.expm(tilted * interval_dt)

    def batch_transitions(batch_generators, batch_potentials):
        return jax.vmap(interval_transition)(batch_generators, batch_potentials, dt)

    transition_matrices = jax.vmap(batch_transitions)(generators, potentials)

    def solve_batch(transitions, initial):
        def forward_step(alpha_t, transition):
            unscaled_next_alpha = transition @ alpha_t
            scale = jnp.sum(unscaled_next_alpha)
            safe_scale = jnp.maximum(scale, jnp.finfo(jnp.float32).tiny)
            next_alpha = unscaled_next_alpha / safe_scale
            return next_alpha, (next_alpha, safe_scale)

        _, (alpha_tail, transition_scales) = jax.lax.scan(forward_step, initial, transitions)
        alpha = jnp.concatenate([initial[None, :], alpha_tail], axis=0)

        terminal_beta = jnp.ones_like(initial)

        def backward_step(beta_t, inputs):
            transition, scale = inputs
            next_beta = (transition.T @ beta_t) / scale
            return next_beta, next_beta

        _, beta_tail_reversed = jax.lax.scan(
            backward_step,
            terminal_beta,
            (transitions[::-1], transition_scales[::-1]),
        )
        beta = jnp.concatenate([beta_tail_reversed[::-1], terminal_beta[None, :]], axis=0)
        log_partition = jnp.sum(jnp.log(transition_scales))
        return alpha, beta, transition_scales, log_partition

    alpha, beta, transition_scales, log_partition = jax.vmap(solve_batch)(
        transition_matrices,
        initial_probabilities,
    )
    return transition_matrices, transition_scales, alpha, beta, log_partition


@jax.jit
def _expected_occupancy_kernel(
    generators: jax.Array,
    potentials: jax.Array,
    transition_scales: jax.Array,
    alpha: jax.Array,
    beta: jax.Array,
    time_grid: jax.Array,
) -> jax.Array:
    dt = jnp.diff(time_grid)
    n_states = generators.shape[-1]
    state_basis = jnp.eye(n_states, dtype=jnp.float32)

    def interval_occupancy(generator, potential, transition_scale, alpha_left, beta_right, interval_dt):
        matrix = _tilted_generator(generator, potential) * interval_dt

        def one_state(direction_diag):
            direction = jnp.diag(direction_diag) * interval_dt
            frechet = jax.scipy.linalg.expm_frechet(matrix, direction, compute_expm=False)
            return beta_right @ frechet @ alpha_left / transition_scale

        return jax.vmap(one_state)(state_basis)

    def batch_occupancy(batch_generators, batch_potentials, batch_scales, batch_alpha, batch_beta):
        return jax.vmap(interval_occupancy, in_axes=(0, 0, 0, 0, 0, 0))(
            batch_generators,
            batch_potentials,
            batch_scales,
            batch_alpha[:-1],
            batch_beta[1:],
            dt,
        )

    return jax.vmap(batch_occupancy)(generators, potentials, transition_scales, alpha, beta)


@jax.jit
def _expected_jumps_kernel(
    generators: jax.Array,
    potentials: jax.Array,
    transition_scales: jax.Array,
    alpha: jax.Array,
    beta: jax.Array,
    time_grid: jax.Array,
) -> jax.Array:
    dt = jnp.diff(time_grid)
    n_states = generators.shape[-1]
    edge_basis = jnp.eye(n_states * n_states, dtype=jnp.float32).reshape(
        (n_states * n_states, n_states, n_states)
    )
    offdiag_mask = (1.0 - jnp.eye(n_states, dtype=jnp.float32)).reshape((n_states * n_states,))

    def interval_jumps(generator, potential, transition_scale, alpha_left, beta_right, interval_dt):
        matrix = _tilted_generator(generator, potential) * interval_dt

        def one_edge(basis_matrix, is_offdiag):
            direction = basis_matrix * generator * interval_dt
            frechet = jax.scipy.linalg.expm_frechet(matrix, direction, compute_expm=False)
            return is_offdiag * (beta_right @ frechet @ alpha_left) / transition_scale

        return jax.vmap(one_edge)(edge_basis, offdiag_mask).reshape((n_states, n_states))

    def batch_jumps(batch_generators, batch_potentials, batch_scales, batch_alpha, batch_beta):
        return jax.vmap(interval_jumps, in_axes=(0, 0, 0, 0, 0, 0))(
            batch_generators,
            batch_potentials,
            batch_scales,
            batch_alpha[:-1],
            batch_beta[1:],
            dt,
        )

    return jax.vmap(batch_jumps)(generators, potentials, transition_scales, alpha, beta)


def _tilted_generators(generators: jax.Array, potentials: jax.Array) -> jax.Array:
    eye = jnp.eye(generators.shape[-1], dtype=jnp.float32)
    return generators + eye[None, :, :] * potentials[:, None, :]


def _tilted_generator(generator: jax.Array, potential: jax.Array) -> jax.Array:
    eye = jnp.eye(generator.shape[-1], dtype=jnp.float32)
    return generator + eye * potential


def _validate_time_grid(time_grid: np.ndarray) -> np.ndarray:
    time_grid = np.asarray(time_grid, dtype=np.float32)
    if time_grid.ndim != 1 or time_grid.size < 2:
        raise ValueError("time_grid must be one-dimensional with at least two points.")
    if np.any(np.diff(time_grid) <= 0):
        raise ValueError("time_grid must be strictly increasing.")
    return time_grid


def _normalize_initial_probabilities(initial_probabilities: np.ndarray) -> np.ndarray:
    initial = np.asarray(initial_probabilities, dtype=np.float32)
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
    generator = np.asarray(generator, dtype=np.float32)
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
        validate_column_generator(matrix, atol=1e-6)
    return generator.astype(np.float32)


def _normalize_potentials(
    potentials: np.ndarray | None,
    n_intervals: int,
    n_states: int,
) -> np.ndarray:
    if potentials is None:
        return np.zeros((1, n_intervals, n_states), dtype=np.float32)
    potentials = np.asarray(potentials, dtype=np.float32)
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
    return potentials.astype(np.float32)


def _interval_index(time_grid: np.ndarray, time: float) -> int:
    if not time_grid[0] <= time <= time_grid[-1]:
        raise ValueError("time is outside the time grid.")
    if np.isclose(time, time_grid[-1]):
        return time_grid.size - 2
    return int(np.searchsorted(time_grid, time, side="right") - 1)
