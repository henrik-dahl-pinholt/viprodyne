"""JAX Bernoulli loading kernels for MS2 observations."""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp

MATMUL_PRECISION = jax.lax.Precision.HIGHEST


def build_ms2_design_matrix(
    sampling_times: jnp.ndarray,
    loading_grid: jnp.ndarray,
    kernel: Callable[[jnp.ndarray], jnp.ndarray],
) -> jnp.ndarray:
    """Build a linear MS2 observation matrix from loading times and a kernel."""
    sampling_times = jnp.asarray(sampling_times, dtype=jnp.float32)
    loading_grid = jnp.asarray(loading_grid, dtype=jnp.float32)
    if sampling_times.ndim != 1 or loading_grid.ndim != 1:
        raise ValueError("sampling_times and loading_grid must be one-dimensional.")
    return jnp.asarray(
        kernel(sampling_times[:, None] - loading_grid[None, :]),
        dtype=jnp.float32,
    )


def enumerate_binary_configurations(n_variables: int) -> jnp.ndarray:
    """Return all binary configurations, ordered by integer value."""
    if n_variables < 0:
        raise ValueError("n_variables must be non-negative.")
    states = jnp.arange(1 << n_variables, dtype=jnp.uint32)
    shifts = jnp.arange(n_variables - 1, -1, -1, dtype=jnp.uint32)
    return ((states[:, None] >> shifts[None, :]) & 1).astype(jnp.float32)


@jax.jit
def bernoulli_transfer_log_likelihood(
    observed: jnp.ndarray,
    prior_probabilities: jnp.ndarray,
    window_weights: jnp.ndarray,
    observation_starts: jnp.ndarray,
    noise_std: jnp.ndarray,
    finite_mask: jnp.ndarray,
) -> jnp.ndarray:
    """Exact Bernoulli loading log likelihood by sliding-window transfer.

    ``window_weights`` maps the active binary loading window to the MS2 signal.
    ``observation_starts[t]`` is the loading-grid index of the first bit in the
    window for observation ``t``. Starts must be monotone outside this JIT kernel.
    """
    observed = jnp.asarray(observed, dtype=jnp.float32)
    prior = jnp.clip(jnp.asarray(prior_probabilities, dtype=jnp.float32), 1e-7, 1.0 - 1e-7)
    window_weights = jnp.asarray(window_weights, dtype=jnp.float32)
    starts = jnp.asarray(observation_starts, dtype=jnp.int32)
    noise_by_observation = jnp.broadcast_to(
        jnp.asarray(noise_std, dtype=jnp.float32),
        observed.shape,
    )
    finite_mask = jnp.asarray(finite_mask, dtype=bool)

    window_size = window_weights.shape[0]
    n_states = 1 << window_size
    bits = enumerate_binary_configurations(window_size)
    emission_means = bits @ window_weights
    shifts = starts - jnp.concatenate([starts[:1], starts[:-1]])
    max_shift = jnp.max(shifts)

    initial_window = jax.lax.dynamic_slice(prior, (starts[0],), (window_size,))
    initial_logp = bits @ jnp.log(initial_window)
    initial_logp = initial_logp + (1.0 - bits) @ jnp.log1p(-initial_window)
    initial_logp = initial_logp - jax.scipy.special.logsumexp(initial_logp)
    alpha = jnp.exp(initial_logp)

    def append_state(alpha_t: jnp.ndarray, p_new: jnp.ndarray) -> jnp.ndarray:
        if window_size == 1:
            return jnp.asarray([1.0 - p_new, p_new], dtype=jnp.float32)
        collapsed = alpha_t.reshape((2, 1 << (window_size - 1))).sum(axis=0)
        return jnp.stack([collapsed * (1.0 - p_new), collapsed * p_new], axis=1).reshape(
            (n_states,)
        )

    def step(carry, inputs):
        alpha_t, loglik = carry
        y_t, is_finite_t, shift_t, start_t, noise_t = inputs

        def append_one(i, alpha_inner):
            append_ind = start_t + window_size - shift_t + i
            p_new = prior[append_ind]
            return jax.lax.cond(
                i < shift_t,
                lambda a: append_state(a, p_new),
                lambda a: a,
                alpha_inner,
            )

        alpha_t = jax.lax.fori_loop(0, max_shift, append_one, alpha_t)
        residual = y_t - emission_means
        obs_logp = -0.5 * (
            jnp.log(2.0 * jnp.pi * noise_t**2) + residual * residual / noise_t**2
        )
        joint_logp = jnp.log(jnp.maximum(alpha_t, jnp.finfo(jnp.float32).tiny)) + obs_logp
        local_loglik = jax.scipy.special.logsumexp(joint_logp)
        updated = jnp.exp(joint_logp - local_loglik)
        alpha_t = jnp.where(is_finite_t, updated, alpha_t)
        loglik = loglik + jnp.where(is_finite_t, local_loglik, 0.0)
        return (alpha_t, loglik), None

    (_, loglik), _ = jax.lax.scan(
        step,
        (alpha, jnp.asarray(0.0, dtype=jnp.float32)),
        (observed, finite_mask, shifts, starts, noise_by_observation),
    )
    return loglik


@jax.jit
def bernoulli_transfer_log_likelihood_batch(
    observed: jnp.ndarray,
    prior_probabilities: jnp.ndarray,
    window_weights: jnp.ndarray,
    observation_starts: jnp.ndarray,
    noise_std: jnp.ndarray,
    finite_mask: jnp.ndarray,
) -> jnp.ndarray:
    """Batch ``bernoulli_transfer_log_likelihood`` over trajectories."""
    return jax.vmap(
        bernoulli_transfer_log_likelihood,
        in_axes=(0, 0, None, None, 0, 0),
    )(
        observed,
        prior_probabilities,
        window_weights,
        observation_starts,
        noise_std,
        finite_mask,
    )


@jax.jit
def exact_bernoulli_posterior(
    observed: jnp.ndarray,
    prior_probabilities: jnp.ndarray,
    design_matrix: jnp.ndarray,
    noise_std: jnp.ndarray,
    finite_mask: jnp.ndarray,
    configurations: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Enumerate binary loading configurations and compute exact posterior moments.

    Returns ``(log_evidence, marginals, pairwise, predicted_signal, posterior_probs)``.
    This exact kernel is intended for small systems and transfer-matrix tests.
    """
    observed = jnp.asarray(observed, dtype=jnp.float32)
    prior = jnp.clip(jnp.asarray(prior_probabilities, dtype=jnp.float32), 1e-7, 1.0 - 1e-7)
    design_matrix = jnp.asarray(design_matrix, dtype=jnp.float32)
    configurations = jnp.asarray(configurations, dtype=jnp.float32)
    finite_mask = jnp.asarray(finite_mask, dtype=bool)
    noise_std = jnp.asarray(noise_std, dtype=jnp.float32)

    log_prior = configurations @ jnp.log(prior)
    log_prior = log_prior + (1.0 - configurations) @ jnp.log1p(-prior)
    means = jnp.matmul(configurations, design_matrix.T, precision=MATMUL_PRECISION)
    safe_observed = jnp.where(finite_mask, observed, 0.0)
    residuals = safe_observed[None, :] - means
    obs_terms = -0.5 * (
        jnp.log(2.0 * jnp.pi * noise_std**2) + residuals * residuals / noise_std**2
    )
    log_likelihood = jnp.sum(jnp.where(finite_mask[None, :], obs_terms, 0.0), axis=1)
    log_joint = log_prior + log_likelihood
    log_evidence = jax.scipy.special.logsumexp(log_joint)
    posterior_probabilities = jnp.exp(log_joint - log_evidence)
    marginal_probabilities = jnp.matmul(
        posterior_probabilities,
        configurations,
        precision=MATMUL_PRECISION,
    )
    pairwise_probabilities = jnp.matmul(
        (configurations * posterior_probabilities[:, None]).T,
        configurations,
        precision=MATMUL_PRECISION,
    )
    predicted_signal = jnp.matmul(
        design_matrix,
        marginal_probabilities,
        precision=MATMUL_PRECISION,
    )
    return (
        log_evidence,
        marginal_probabilities,
        pairwise_probabilities,
        predicted_signal,
        posterior_probabilities,
    )


@jax.jit
def mean_field_bernoulli_elbo(
    load_probabilities: jnp.ndarray,
    observed: jnp.ndarray,
    prior_probabilities: jnp.ndarray,
    design_matrix: jnp.ndarray,
    noise_std: jnp.ndarray,
    finite_mask: jnp.ndarray,
) -> jnp.ndarray:
    """Independent-Bernoulli variational ELBO for one trajectory."""
    q = jnp.clip(jnp.asarray(load_probabilities, dtype=jnp.float32), 1e-7, 1.0 - 1e-7)
    observed = jnp.asarray(observed, dtype=jnp.float32)
    prior = jnp.clip(jnp.asarray(prior_probabilities, dtype=jnp.float32), 1e-7, 1.0 - 1e-7)
    design_matrix = jnp.asarray(design_matrix, dtype=jnp.float32)
    noise_std = jnp.asarray(noise_std, dtype=jnp.float32)
    finite_mask = jnp.asarray(finite_mask, dtype=bool)

    mean_signal = jnp.matmul(design_matrix, q, precision=MATMUL_PRECISION)
    variance_signal = jnp.matmul(
        design_matrix * design_matrix,
        q * (1.0 - q),
        precision=MATMUL_PRECISION,
    )
    residual = observed - mean_signal
    obs_terms = -0.5 * (
        jnp.log(2.0 * jnp.pi * noise_std**2)
        + (residual * residual + variance_signal) / noise_std**2
    )
    obs_term = jnp.sum(jnp.where(finite_mask, obs_terms, 0.0))
    prior_term = jnp.sum(q * jnp.log(prior) + (1.0 - q) * jnp.log1p(-prior))
    entropy = -jnp.sum(q * jnp.log(q) + (1.0 - q) * jnp.log1p(-q))
    return obs_term + prior_term + entropy
