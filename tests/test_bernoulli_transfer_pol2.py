import numpy as np
import pytest
from scipy.special import logsumexp

from viprodyne.core.bernoulli_transfer_pol2 import (
    build_ms2_design_matrix,
    enumerate_binary_configurations,
    exact_bernoulli_posterior,
    mean_field_bernoulli_elbo,
)


def gaussian_logpdf(value, mean, noise):
    return -0.5 * (np.log(2.0 * np.pi * noise**2) + (value - mean) ** 2 / noise**2)


def test_binary_configuration_order():
    configs = enumerate_binary_configurations(3)

    np.testing.assert_array_equal(
        configs,
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 1.0],
        ],
    )


def test_design_matrix_uses_sampling_minus_loading_times():
    design = build_ms2_design_matrix(
        sampling_times=np.array([0.0, 1.0, 2.0]),
        loading_grid=np.array([0.0, 1.0]),
        kernel=lambda t: ((t >= 0.0) & (t <= 1.0)).astype(float),
    )

    np.testing.assert_array_equal(design, [[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])


def test_single_loading_posterior_matches_bayes_rule_and_elbo_identity():
    prior = np.array([0.3])
    design = np.array([[2.0]])
    observed = np.array([0.8])
    noise = 0.5

    posterior = exact_bernoulli_posterior(observed, prior, design, noise)

    logp0 = np.log1p(-prior[0]) + gaussian_logpdf(observed[0], 0.0, noise)
    logp1 = np.log(prior[0]) + gaussian_logpdf(observed[0], 2.0, noise)
    logz = logsumexp([logp0, logp1])
    expected_p = np.exp(logp1 - logz)

    assert posterior.log_evidence == pytest.approx(logz)
    np.testing.assert_allclose(posterior.marginal_probabilities, [expected_p])
    np.testing.assert_allclose(posterior.pairwise_probabilities, [[expected_p]])
    np.testing.assert_allclose(posterior.predicted_signal, [2.0 * expected_p])
    assert mean_field_bernoulli_elbo([expected_p], observed, prior, design, noise) == pytest.approx(
        logz
    )


def test_two_loading_posterior_matches_manual_enumeration():
    prior = np.array([0.25, 0.6])
    design = np.array([[1.0, 2.0]])
    observed = np.array([1.3])
    noise = 0.7
    configs = enumerate_binary_configurations(2)

    posterior = exact_bernoulli_posterior(observed, prior, design, noise)

    means = configs @ design[0]
    log_prior = configs @ np.log(prior) + (1.0 - configs) @ np.log1p(-prior)
    log_joint = log_prior + gaussian_logpdf(observed[0], means, noise)
    weights = np.exp(log_joint - logsumexp(log_joint))
    expected_pairwise = np.einsum("c,ci,cj->ij", weights, configs, configs)

    assert posterior.log_evidence == pytest.approx(logsumexp(log_joint))
    np.testing.assert_allclose(posterior.posterior_probabilities, weights)
    np.testing.assert_allclose(posterior.marginal_probabilities, weights @ configs)
    np.testing.assert_allclose(posterior.pairwise_probabilities, expected_pairwise)
    assert mean_field_bernoulli_elbo(
        posterior.marginal_probabilities,
        observed,
        prior,
        design,
        noise,
    ) <= posterior.log_evidence + 1e-12


def test_missing_observations_leave_independent_prior_unchanged():
    prior = np.array([0.2, 0.5, 0.8])
    posterior = exact_bernoulli_posterior(
        observed=np.array([np.nan]),
        prior_probabilities=prior,
        design_matrix=np.array([[1.0, 2.0, 3.0]]),
        noise_std=1.0,
    )

    expected_pairwise = np.outer(prior, prior)
    np.fill_diagonal(expected_pairwise, prior)
    assert posterior.log_evidence == pytest.approx(0.0)
    np.testing.assert_allclose(posterior.marginal_probabilities, prior)
    np.testing.assert_allclose(posterior.pairwise_probabilities, expected_pairwise)


def test_exact_enumeration_guard_is_explicit():
    with pytest.raises(ValueError, match="max_loadings"):
        exact_bernoulli_posterior(
            observed=np.zeros(1),
            prior_probabilities=np.full(3, 0.5),
            design_matrix=np.zeros((1, 3)),
            noise_std=1.0,
            max_loadings=2,
        )
