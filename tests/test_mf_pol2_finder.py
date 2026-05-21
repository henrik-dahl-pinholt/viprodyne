import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import expit, logit

from viprodyne.core.bernoulli_transfer_pol2 import (
    enumerate_binary_configurations,
    exact_bernoulli_posterior,
)
from viprodyne.core.mf_pol2_finder import (
    fit_mean_field_bernoulli,
    mean_field_bernoulli_elbo_and_gradient,
)


def finite_difference_gradient(fn, x, eps=1e-2):
    grad = np.zeros_like(x, dtype=np.float32)
    for i in range(x.size):
        step = np.zeros_like(x, dtype=np.float32)
        step[i] = eps
        grad[i] = (fn(x + step) - fn(x - step)) / (2.0 * eps)
    return grad


def exact_marginals(observed, prior, design, noise):
    _, marginal, _, predicted, _ = exact_bernoulli_posterior(
        jnp.asarray(observed),
        jnp.asarray(prior),
        jnp.asarray(design),
        jnp.asarray(noise),
        jnp.asarray(np.isfinite(observed)),
        enumerate_binary_configurations(len(prior)),
    )
    return np.asarray(marginal), np.asarray(predicted)


def test_mean_field_gradient_matches_finite_difference():
    observed = np.array([0.4, 1.2], dtype=np.float32)
    prior = np.array([0.25, 0.6, 0.4], dtype=np.float32)
    design = np.array([[1.0, 0.0, 0.5], [0.0, 2.0, 1.0]], dtype=np.float32)
    logits = np.array([-0.2, 0.5, 1.1], dtype=np.float32)
    noise = np.float32(0.8)

    elbo, gradient = mean_field_bernoulli_elbo_and_gradient(
        jnp.asarray(logits),
        jnp.asarray(observed),
        jnp.asarray(prior),
        jnp.asarray(design),
        jnp.asarray(noise),
        jnp.asarray(np.isfinite(observed)),
    )
    expected = finite_difference_gradient(
        lambda x: float(
            mean_field_bernoulli_elbo_and_gradient(
                jnp.asarray(x),
                jnp.asarray(observed),
                jnp.asarray(prior),
                jnp.asarray(design),
                jnp.asarray(noise),
                jnp.asarray(np.isfinite(observed)),
            )[0]
        ),
        logits,
    )

    assert isinstance(elbo, jax.Array)
    assert elbo.dtype == jnp.float32
    np.testing.assert_allclose(np.asarray(gradient), expected, rtol=2e-3, atol=2e-3)


def test_single_loading_fit_matches_exact_posterior():
    observed = np.array([0.8], dtype=np.float32)
    prior = np.array([0.3], dtype=np.float32)
    design = np.array([[2.0]], dtype=np.float32)
    noise = np.float32(0.5)
    marginal, predicted = exact_marginals(observed, prior, design, noise)

    result = fit_mean_field_bernoulli(observed, prior, design, noise)

    assert result.success
    assert result.load_probabilities.dtype == np.float32
    np.testing.assert_allclose(result.load_probabilities, marginal, rtol=1e-4)
    np.testing.assert_allclose(result.predicted_signal, predicted, rtol=1e-4)


def test_factorized_observations_fit_matches_exact_marginals():
    observed = np.array([0.2, 1.7], dtype=np.float32)
    prior = np.array([0.4, 0.65], dtype=np.float32)
    design = np.array([[1.0, 0.0], [0.0, 2.0]], dtype=np.float32)
    noise = np.float32(0.7)
    marginal, _ = exact_marginals(observed, prior, design, noise)

    result = fit_mean_field_bernoulli(observed, prior, design, noise)

    assert result.success
    np.testing.assert_allclose(result.load_probabilities, marginal, rtol=1e-4)


def test_missing_observations_return_prior_optimum():
    observed = np.array([np.nan, np.nan], dtype=np.float32)
    prior = np.array([0.2, 0.6, 0.8], dtype=np.float32)
    design = np.ones((2, 3), dtype=np.float32)

    result = fit_mean_field_bernoulli(
        observed,
        prior,
        design,
        noise_std=1.0,
        initial_logits=np.zeros_like(prior),
    )

    assert result.success
    np.testing.assert_allclose(result.load_probabilities, prior, rtol=2e-6, atol=1e-6)
    assert result.elbo == pytest.approx(0.0, abs=1e-7)


def test_gradient_is_zero_at_prior_when_all_observations_missing():
    prior = np.array([0.2, 0.6, 0.8], dtype=np.float32)
    elbo, gradient = mean_field_bernoulli_elbo_and_gradient(
        jnp.asarray(logit(prior).astype(np.float32)),
        jnp.asarray([0.0], dtype=jnp.float32),
        jnp.asarray(prior),
        jnp.ones((1, 3), dtype=jnp.float32),
        jnp.asarray(1.0, dtype=jnp.float32),
        jnp.asarray([False]),
    )

    assert float(elbo) == pytest.approx(0.0, abs=1e-7)
    np.testing.assert_allclose(np.asarray(gradient), np.zeros_like(prior), atol=1e-7)


def test_initial_logits_are_respected_and_validated():
    prior = np.array([0.4], dtype=np.float32)
    result = fit_mean_field_bernoulli(
        observed=np.array([np.nan], dtype=np.float32),
        prior_probabilities=prior,
        design_matrix=np.ones((1, 1), dtype=np.float32),
        noise_std=1.0,
        initial_logits=np.array([2.0], dtype=np.float32),
        maxiter=1,
    )

    assert result.load_probabilities.shape == prior.shape
    assert np.isfinite(result.elbo)
    with pytest.raises(ValueError, match="initial_logits"):
        fit_mean_field_bernoulli(
            observed=np.array([0.0], dtype=np.float32),
            prior_probabilities=prior,
            design_matrix=np.ones((1, 1), dtype=np.float32),
            noise_std=1.0,
            initial_logits=np.array([1.0, 2.0], dtype=np.float32),
        )


def test_logit_gradient_points_toward_posterior_for_single_variable():
    observed = np.array([1.0], dtype=np.float32)
    prior = np.array([0.2], dtype=np.float32)
    design = np.array([[2.0]], dtype=np.float32)
    noise = np.float32(0.5)
    marginal, _ = exact_marginals(observed, prior, design, noise)
    q_low = 0.5 * marginal
    _, gradient = mean_field_bernoulli_elbo_and_gradient(
        jnp.asarray(logit(q_low).astype(np.float32)),
        jnp.asarray(observed),
        jnp.asarray(prior),
        jnp.asarray(design),
        jnp.asarray(noise),
        jnp.asarray(np.isfinite(observed)),
    )

    assert gradient[0] > 0.0
    assert expit(logit(q_low))[0] < marginal[0]
