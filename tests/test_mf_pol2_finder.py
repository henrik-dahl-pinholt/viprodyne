import numpy as np
import pytest
import jax
import jax.numpy as jnp
from scipy.special import expit, logit

from viprodyne.core.bernoulli_transfer_pol2 import exact_bernoulli_posterior
from viprodyne.core.mf_pol2_finder import (
    fit_mean_field_bernoulli,
    mean_field_bernoulli_elbo_and_gradient,
    mean_field_bernoulli_elbo_and_gradient_jax,
)


def finite_difference_gradient(fn, x, eps=1e-2):
    grad = np.zeros_like(x, dtype=float)
    for i in range(x.size):
        step = np.zeros_like(x, dtype=float)
        step[i] = eps
        grad[i] = (fn(x + step) - fn(x - step)) / (2.0 * eps)
    return grad


def test_mean_field_gradient_matches_finite_difference():
    observed = np.array([0.4, 1.2])
    prior = np.array([0.25, 0.6, 0.4])
    design = np.array([[1.0, 0.0, 0.5], [0.0, 2.0, 1.0]])
    logits = np.array([-0.2, 0.5, 1.1])
    noise = 0.8

    elbo, gradient = mean_field_bernoulli_elbo_and_gradient(
        logits,
        observed,
        prior,
        design,
        noise,
    )
    expected = finite_difference_gradient(
        lambda x: mean_field_bernoulli_elbo_and_gradient(x, observed, prior, design, noise)[0],
        logits,
    )

    assert np.isfinite(elbo)
    np.testing.assert_allclose(gradient, expected, rtol=2e-3, atol=2e-3)


def test_mean_field_jax_value_and_grad_kernel_matches_wrapper():
    observed = np.array([0.4, 1.2])
    prior = np.array([0.25, 0.6, 0.4])
    design = np.array([[1.0, 0.0, 0.5], [0.0, 2.0, 1.0]])
    logits = np.array([-0.2, 0.5, 1.1])
    noise = 0.8
    wrapper_elbo, wrapper_gradient = mean_field_bernoulli_elbo_and_gradient(
        logits,
        observed,
        prior,
        design,
        noise,
    )

    elbo, gradient = mean_field_bernoulli_elbo_and_gradient_jax(
        jnp.asarray(logits),
        jnp.asarray(observed),
        jnp.asarray(prior),
        jnp.asarray(design),
        jnp.asarray(noise),
        jnp.asarray(np.isfinite(observed)),
    )

    assert isinstance(elbo, jax.Array)
    assert elbo.dtype == jnp.float32
    assert float(elbo) == pytest.approx(wrapper_elbo)
    np.testing.assert_allclose(np.asarray(gradient), wrapper_gradient)


def test_single_loading_fit_matches_exact_posterior():
    observed = np.array([0.8])
    prior = np.array([0.3])
    design = np.array([[2.0]])
    noise = 0.5
    exact = exact_bernoulli_posterior(observed, prior, design, noise)

    result = fit_mean_field_bernoulli(observed, prior, design, noise)

    assert result.success
    assert result.load_probabilities.dtype == np.float32
    np.testing.assert_allclose(result.load_probabilities, exact.marginal_probabilities, rtol=1e-4)
    np.testing.assert_allclose(result.predicted_signal, exact.predicted_signal, rtol=1e-4)
    assert result.elbo == pytest.approx(exact.log_evidence, rel=1e-5)


def test_factorized_observations_fit_matches_exact_marginals():
    observed = np.array([0.2, 1.7])
    prior = np.array([0.4, 0.65])
    design = np.array([[1.0, 0.0], [0.0, 2.0]])
    noise = 0.7
    exact = exact_bernoulli_posterior(observed, prior, design, noise)

    result = fit_mean_field_bernoulli(observed, prior, design, noise)

    assert result.success
    np.testing.assert_allclose(result.load_probabilities, exact.marginal_probabilities, rtol=5e-5)
    assert result.elbo == pytest.approx(exact.log_evidence, rel=1e-5)


def test_missing_observations_return_prior_optimum():
    observed = np.array([np.nan, np.nan])
    prior = np.array([0.2, 0.6, 0.8])
    design = np.ones((2, 3))

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
    prior = np.array([0.2, 0.6, 0.8])
    elbo, gradient = mean_field_bernoulli_elbo_and_gradient(
        logits=logit(prior),
        observed=np.array([np.nan]),
        prior_probabilities=prior,
        design_matrix=np.ones((1, 3)),
        noise_std=1.0,
    )

    assert elbo == pytest.approx(0.0, abs=1e-7)
    np.testing.assert_allclose(gradient, np.zeros_like(prior), atol=1e-7)


def test_initial_logits_are_respected_and_validated():
    prior = np.array([0.4])
    result = fit_mean_field_bernoulli(
        observed=np.array([np.nan]),
        prior_probabilities=prior,
        design_matrix=np.ones((1, 1)),
        noise_std=1.0,
        initial_logits=np.array([2.0]),
        maxiter=1,
    )

    assert result.load_probabilities.shape == prior.shape
    assert np.isfinite(result.elbo)
    with pytest.raises(ValueError, match="initial_logits"):
        fit_mean_field_bernoulli(
            observed=np.array([0.0]),
            prior_probabilities=prior,
            design_matrix=np.ones((1, 1)),
            noise_std=1.0,
            initial_logits=np.array([1.0, 2.0]),
        )


def test_logit_gradient_points_toward_posterior_for_single_variable():
    observed = np.array([1.0])
    prior = np.array([0.2])
    design = np.array([[2.0]])
    noise = 0.5
    exact = exact_bernoulli_posterior(observed, prior, design, noise)
    q_low = 0.5 * exact.marginal_probabilities
    _, gradient = mean_field_bernoulli_elbo_and_gradient(logit(q_low), observed, prior, design, noise)

    assert gradient[0] > 0.0
    assert expit(logit(q_low))[0] < exact.marginal_probabilities[0]
