"""Exact finite Bernoulli loading posterior for MS2 observations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.special import logsumexp


@dataclass(frozen=True)
class ExactBernoulliPosterior:
    """Exact posterior moments over binary Pol2 loading variables."""

    log_evidence: float
    marginal_probabilities: np.ndarray
    pairwise_probabilities: np.ndarray
    predicted_signal: np.ndarray
    configurations: np.ndarray
    posterior_probabilities: np.ndarray

    @property
    def covariance(self) -> np.ndarray:
        """Return posterior covariance of binary loading variables."""
        return self.pairwise_probabilities - np.outer(
            self.marginal_probabilities,
            self.marginal_probabilities,
        )


def build_ms2_design_matrix(
    sampling_times: np.ndarray,
    loading_grid: np.ndarray,
    kernel: Callable[[np.ndarray], np.ndarray],
) -> np.ndarray:
    """Build a linear MS2 observation matrix from loading times and a kernel."""
    sampling_times = np.asarray(sampling_times, dtype=float)
    loading_grid = np.asarray(loading_grid, dtype=float)
    if sampling_times.ndim != 1 or loading_grid.ndim != 1:
        raise ValueError("sampling_times and loading_grid must be one-dimensional.")
    return np.asarray(kernel(sampling_times[:, None] - loading_grid[None, :]), dtype=float)


def exact_bernoulli_posterior(
    observed: np.ndarray,
    prior_probabilities: np.ndarray,
    design_matrix: np.ndarray,
    noise_std: float,
    mask: np.ndarray | None = None,
    max_loadings: int = 24,
) -> ExactBernoulliPosterior:
    """Enumerate all binary loading configurations and compute exact moments.

    This reference kernel is intended for small systems and tests. Optimized
    transfer-matrix code can be checked against it on short grids.
    """
    observed, prior_probabilities, design_matrix, finite_mask = _prepare_inputs(
        observed,
        prior_probabilities,
        design_matrix,
        noise_std,
        mask,
    )
    n_loadings = prior_probabilities.size
    if n_loadings > max_loadings:
        raise ValueError(
            f"exact enumeration requested for {n_loadings} variables; "
            f"max_loadings is {max_loadings}."
        )
    configurations = enumerate_binary_configurations(n_loadings)
    log_prior = _log_bernoulli_prior(configurations, prior_probabilities)
    log_likelihood = _gaussian_log_likelihood(
        configurations,
        observed,
        design_matrix,
        float(noise_std),
        finite_mask,
    )
    log_joint = log_prior + log_likelihood
    log_evidence = float(logsumexp(log_joint))
    posterior_probabilities = np.exp(log_joint - log_evidence)
    marginal_probabilities = posterior_probabilities @ configurations
    pairwise_probabilities = np.einsum(
        "c,ci,cj->ij",
        posterior_probabilities,
        configurations,
        configurations,
    )
    predicted_signal = design_matrix @ marginal_probabilities
    return ExactBernoulliPosterior(
        log_evidence=log_evidence,
        marginal_probabilities=marginal_probabilities,
        pairwise_probabilities=pairwise_probabilities,
        predicted_signal=predicted_signal,
        configurations=configurations,
        posterior_probabilities=posterior_probabilities,
    )


def enumerate_binary_configurations(n_variables: int) -> np.ndarray:
    """Return all binary configurations, ordered by integer value."""
    if n_variables < 0:
        raise ValueError("n_variables must be non-negative.")
    states = np.arange(1 << n_variables, dtype=np.uint64)
    shifts = np.arange(n_variables - 1, -1, -1, dtype=np.uint64)
    return ((states[:, None] >> shifts[None, :]) & 1).astype(float)


def mean_field_bernoulli_elbo(
    load_probabilities: np.ndarray,
    observed: np.ndarray,
    prior_probabilities: np.ndarray,
    design_matrix: np.ndarray,
    noise_std: float,
    mask: np.ndarray | None = None,
) -> float:
    """Compute the independent-Bernoulli variational ELBO for one trajectory."""
    observed, prior_probabilities, design_matrix, finite_mask = _prepare_inputs(
        observed,
        prior_probabilities,
        design_matrix,
        noise_std,
        mask,
    )
    load_probabilities = np.asarray(load_probabilities, dtype=float)
    if load_probabilities.shape != prior_probabilities.shape:
        raise ValueError("load_probabilities must match prior_probabilities.")
    if np.any((load_probabilities < 0.0) | (load_probabilities > 1.0)):
        raise ValueError("load_probabilities must lie in [0, 1].")

    q = np.clip(load_probabilities, 1e-12, 1.0 - 1e-12)
    prior = np.clip(prior_probabilities, 1e-12, 1.0 - 1e-12)
    mean_signal = design_matrix @ q
    variance_signal = (design_matrix * design_matrix) @ (q * (1.0 - q))
    residual = observed - mean_signal
    obs_term = -0.5 * np.sum(
        finite_mask
        * (
            np.log(2.0 * np.pi * noise_std**2)
            + (residual * residual + variance_signal) / noise_std**2
        )
    )
    prior_term = np.sum(q * np.log(prior) + (1.0 - q) * np.log1p(-prior))
    entropy = -np.sum(q * np.log(q) + (1.0 - q) * np.log1p(-q))
    return float(obs_term + prior_term + entropy)


def _prepare_inputs(
    observed: np.ndarray,
    prior_probabilities: np.ndarray,
    design_matrix: np.ndarray,
    noise_std: float,
    mask: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    observed = np.asarray(observed, dtype=float)
    prior_probabilities = np.asarray(prior_probabilities, dtype=float)
    design_matrix = np.asarray(design_matrix, dtype=float)
    if observed.ndim != 1:
        raise ValueError("observed must be one-dimensional.")
    if prior_probabilities.ndim != 1:
        raise ValueError("prior_probabilities must be one-dimensional.")
    if design_matrix.shape != (observed.size, prior_probabilities.size):
        raise ValueError("design_matrix must have shape (n_observations, n_loadings).")
    if noise_std <= 0:
        raise ValueError("noise_std must be positive.")
    if np.any((prior_probabilities <= 0.0) | (prior_probabilities >= 1.0)):
        raise ValueError("prior_probabilities must lie strictly inside (0, 1).")
    finite_mask = np.isfinite(observed)
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != observed.shape:
            raise ValueError("mask must have the same shape as observed.")
        finite_mask = finite_mask & mask
    observed = np.where(finite_mask, observed, 0.0)
    return observed, prior_probabilities, design_matrix, finite_mask


def _log_bernoulli_prior(
    configurations: np.ndarray,
    prior_probabilities: np.ndarray,
) -> np.ndarray:
    prior = np.clip(prior_probabilities, 1e-12, 1.0 - 1e-12)
    return configurations @ np.log(prior) + (1.0 - configurations) @ np.log1p(-prior)


def _gaussian_log_likelihood(
    configurations: np.ndarray,
    observed: np.ndarray,
    design_matrix: np.ndarray,
    noise_std: float,
    finite_mask: np.ndarray,
) -> np.ndarray:
    if not np.any(finite_mask):
        return np.zeros(configurations.shape[0], dtype=float)
    means = configurations @ design_matrix[finite_mask].T
    residuals = observed[finite_mask][None, :] - means
    return -0.5 * np.sum(
        np.log(2.0 * np.pi * noise_std**2) + residuals * residuals / noise_std**2,
        axis=1,
    )
