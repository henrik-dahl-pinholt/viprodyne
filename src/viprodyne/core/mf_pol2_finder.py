"""Mean-field Bernoulli inference for Pol2 loading probabilities."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

from viprodyne.core.bernoulli_transfer_pol2 import mean_field_bernoulli_elbo_jax


@dataclass(frozen=True)
class MeanFieldBernoulliResult:
    """Result of mean-field Bernoulli Pol2 loading optimization."""

    load_probabilities: np.ndarray
    logits: np.ndarray
    elbo: float
    predicted_signal: np.ndarray
    success: bool
    message: str
    n_iterations: int


def mean_field_bernoulli_elbo_and_gradient(
    logits: np.ndarray,
    observed: np.ndarray,
    prior_probabilities: np.ndarray,
    design_matrix: np.ndarray,
    noise_std: float,
    mask: np.ndarray | None = None,
) -> tuple[float, np.ndarray]:
    """Return JAX-computed ELBO and gradient with respect to Bernoulli logits."""
    logits, observed, prior, design_matrix, finite_mask = _prepare_inputs(
        logits,
        observed,
        prior_probabilities,
        design_matrix,
        noise_std,
        mask,
    )
    elbo, gradient = mean_field_bernoulli_elbo_and_gradient_jax(
        jnp.asarray(logits),
        jnp.asarray(observed),
        jnp.asarray(prior),
        jnp.asarray(design_matrix),
        jnp.asarray(float(noise_std)),
        jnp.asarray(finite_mask),
    )
    return float(elbo), np.asarray(gradient, dtype=float)


@jax.jit
def mean_field_bernoulli_elbo_from_logits_jax(
    logits: jnp.ndarray,
    observed: jnp.ndarray,
    prior_probabilities: jnp.ndarray,
    design_matrix: jnp.ndarray,
    noise_std: jnp.ndarray,
    finite_mask: jnp.ndarray,
) -> jnp.ndarray:
    """JAX mean-field Bernoulli ELBO parameterized by logits."""
    return mean_field_bernoulli_elbo_jax(
        jax.nn.sigmoid(logits),
        observed,
        prior_probabilities,
        design_matrix,
        noise_std,
        finite_mask,
    )


mean_field_bernoulli_elbo_and_gradient_jax = jax.jit(
    jax.value_and_grad(mean_field_bernoulli_elbo_from_logits_jax)
)


def fit_mean_field_bernoulli(
    observed: np.ndarray,
    prior_probabilities: np.ndarray,
    design_matrix: np.ndarray,
    noise_std: float,
    mask: np.ndarray | None = None,
    initial_logits: np.ndarray | None = None,
    maxiter: int = 1000,
    gtol: float = 1e-5,
) -> MeanFieldBernoulliResult:
    """Optimize independent Bernoulli loading probabilities."""
    prior_probabilities = np.asarray(prior_probabilities, dtype=np.float32)
    finite_mask = np.isfinite(np.asarray(observed, dtype=np.float32))
    if mask is not None:
        finite_mask = finite_mask & np.asarray(mask, dtype=bool)
    if initial_logits is None:
        x0 = _logit(np.clip(prior_probabilities, 1e-6, 1.0 - 1e-6))
    else:
        x0 = np.asarray(initial_logits, dtype=np.float32)
        if x0.shape != prior_probabilities.shape:
            raise ValueError("initial_logits must match prior_probabilities.")
    if not np.any(finite_mask):
        logits = _logit(np.clip(prior_probabilities, 1e-6, 1.0 - 1e-6))
        return MeanFieldBernoulliResult(
            load_probabilities=prior_probabilities.astype(np.float32),
            logits=logits.astype(np.float32),
            elbo=0.0,
            predicted_signal=np.asarray(design_matrix, dtype=np.float32)
            @ prior_probabilities.astype(np.float32),
            success=True,
            message="No finite observations; returned prior optimum.",
            n_iterations=0,
        )

    def objective(x: np.ndarray) -> tuple[float, np.ndarray]:
        elbo, gradient = mean_field_bernoulli_elbo_and_gradient(
            x,
            observed,
            prior_probabilities,
            design_matrix,
            noise_std,
            mask,
        )
        return -elbo, -gradient

    result = minimize(
        objective,
        x0,
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": int(maxiter), "gtol": float(gtol), "ftol": 1e-7, "maxls": 50},
    )
    logits = np.asarray(result.x, dtype=np.float32)
    load_probabilities = np.asarray(jax.nn.sigmoid(jnp.asarray(logits)), dtype=float)
    elbo, _ = mean_field_bernoulli_elbo_and_gradient(
        logits,
        observed,
        prior_probabilities,
        design_matrix,
        noise_std,
        mask,
    )
    success = bool(result.success)
    if not success:
        _, final_gradient = mean_field_bernoulli_elbo_and_gradient(
            logits,
            observed,
            prior_probabilities,
            design_matrix,
            noise_std,
            mask,
        )
        success = bool(np.linalg.norm(final_gradient, ord=np.inf) <= 10.0 * gtol)
    return MeanFieldBernoulliResult(
        load_probabilities=load_probabilities.astype(np.float32),
        logits=logits.astype(np.float32),
        elbo=elbo,
        predicted_signal=(
            np.asarray(design_matrix, dtype=np.float32) @ load_probabilities.astype(np.float32)
        ),
        success=success,
        message=str(result.message),
        n_iterations=int(result.nit),
    )


def _prepare_inputs(
    logits: np.ndarray,
    observed: np.ndarray,
    prior_probabilities: np.ndarray,
    design_matrix: np.ndarray,
    noise_std: float,
    mask: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    logits = np.asarray(logits, dtype=np.float32)
    observed = np.asarray(observed, dtype=np.float32)
    prior_probabilities = np.asarray(prior_probabilities, dtype=np.float32)
    design_matrix = np.asarray(design_matrix, dtype=np.float32)
    if observed.ndim != 1:
        raise ValueError("observed must be one-dimensional.")
    if prior_probabilities.ndim != 1:
        raise ValueError("prior_probabilities must be one-dimensional.")
    if logits.shape != prior_probabilities.shape:
        raise ValueError("logits must match prior_probabilities.")
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
    return logits, observed, prior_probabilities, design_matrix, finite_mask


def _logit(probabilities: np.ndarray) -> np.ndarray:
    probabilities = np.asarray(probabilities, dtype=float)
    return np.log(probabilities) - np.log1p(-probabilities)
