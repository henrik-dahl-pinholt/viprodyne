"""JAX Bernoulli loading kernels for MS2 observations."""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp


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
    means = configurations @ design_matrix.T
    safe_observed = jnp.where(finite_mask, observed, 0.0)
    residuals = safe_observed[None, :] - means
    obs_terms = -0.5 * (
        jnp.log(2.0 * jnp.pi * noise_std**2) + residuals * residuals / noise_std**2
    )
    log_likelihood = jnp.sum(jnp.where(finite_mask[None, :], obs_terms, 0.0), axis=1)
    log_joint = log_prior + log_likelihood
    log_evidence = jax.scipy.special.logsumexp(log_joint)
    posterior_probabilities = jnp.exp(log_joint - log_evidence)
    marginal_probabilities = posterior_probabilities @ configurations
    pairwise_probabilities = jnp.einsum(
        "c,ci,cj->ij",
        posterior_probabilities,
        configurations,
        configurations,
    )
    predicted_signal = design_matrix @ marginal_probabilities
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

    mean_signal = design_matrix @ q
    variance_signal = (design_matrix * design_matrix) @ (q * (1.0 - q))
    residual = observed - mean_signal
    obs_terms = -0.5 * (
        jnp.log(2.0 * jnp.pi * noise_std**2)
        + (residual * residual + variance_signal) / noise_std**2
    )
    obs_term = jnp.sum(jnp.where(finite_mask, obs_terms, 0.0))
    prior_term = jnp.sum(q * jnp.log(prior) + (1.0 - q) * jnp.log1p(-prior))
    entropy = -jnp.sum(q * jnp.log(q) + (1.0 - q) * jnp.log1p(-q))
    return obs_term + prior_term + entropy
