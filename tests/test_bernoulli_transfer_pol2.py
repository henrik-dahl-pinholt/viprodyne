import jax
import jax.numpy as jnp
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


def run_exact(observed, prior, design, noise):
    configs = enumerate_binary_configurations(len(prior))
    return exact_bernoulli_posterior(
        jnp.asarray(observed),
        jnp.asarray(prior),
        jnp.asarray(design),
        jnp.asarray(noise),
        jnp.asarray(np.isfinite(observed)),
        configs,
    )


def test_binary_configuration_order():
    configs = enumerate_binary_configurations(3)

    assert isinstance(configs, jax.Array)
    assert configs.dtype == jnp.float32
    np.testing.assert_array_equal(
        np.asarray(configs),
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
        sampling_times=jnp.asarray([0.0, 1.0, 2.0]),
        loading_grid=jnp.asarray([0.0, 1.0]),
        kernel=lambda t: ((t >= 0.0) & (t <= 1.0)).astype(jnp.float32),
    )

    assert isinstance(design, jax.Array)
    assert design.dtype == jnp.float32
    np.testing.assert_array_equal(np.asarray(design), [[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])


def test_single_loading_posterior_matches_bayes_rule_and_elbo_identity():
    prior = np.array([0.3], dtype=np.float32)
    design = np.array([[2.0]], dtype=np.float32)
    observed = np.array([0.8], dtype=np.float32)
    noise = np.float32(0.5)

    logz_j, marginal, pairwise, predicted, _ = run_exact(observed, prior, design, noise)

    logp0 = np.log1p(-prior[0]) + gaussian_logpdf(observed[0], 0.0, noise)
    logp1 = np.log(prior[0]) + gaussian_logpdf(observed[0], 2.0, noise)
    logz = logsumexp([logp0, logp1])
    expected_p = np.exp(logp1 - logz)

    assert logz_j.dtype == jnp.float32
    assert float(logz_j) == pytest.approx(logz)
    np.testing.assert_allclose(np.asarray(marginal), [expected_p], rtol=1e-6)
    np.testing.assert_allclose(np.asarray(pairwise), [[expected_p]], rtol=1e-6)
    np.testing.assert_allclose(np.asarray(predicted), [2.0 * expected_p], rtol=1e-6)
    elbo = mean_field_bernoulli_elbo(
        jnp.asarray([expected_p], dtype=jnp.float32),
        jnp.asarray(observed),
        jnp.asarray(prior),
        jnp.asarray(design),
        jnp.asarray(noise),
        jnp.asarray(np.isfinite(observed)),
    )
    assert float(elbo) == pytest.approx(logz, rel=1e-6)


def test_two_loading_posterior_matches_manual_enumeration():
    prior = np.array([0.25, 0.6], dtype=np.float32)
    design = np.array([[1.0, 2.0]], dtype=np.float32)
    observed = np.array([1.3], dtype=np.float32)
    noise = np.float32(0.7)
    configs = np.asarray(enumerate_binary_configurations(2))

    logz_j, marginal, pairwise, _, weights_j = run_exact(observed, prior, design, noise)

    means = configs @ design[0]
    log_prior = configs @ np.log(prior) + (1.0 - configs) @ np.log1p(-prior)
    log_joint = log_prior + gaussian_logpdf(observed[0], means, noise)
    weights = np.exp(log_joint - logsumexp(log_joint))
    expected_pairwise = np.einsum("c,ci,cj->ij", weights, configs, configs)

    assert float(logz_j) == pytest.approx(logsumexp(log_joint))
    np.testing.assert_allclose(np.asarray(weights_j), weights, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(np.asarray(marginal), weights @ configs, rtol=1e-6)
    np.testing.assert_allclose(np.asarray(pairwise), expected_pairwise, rtol=1e-6)
    elbo = mean_field_bernoulli_elbo(
        marginal,
        jnp.asarray(observed),
        jnp.asarray(prior),
        jnp.asarray(design),
        jnp.asarray(noise),
        jnp.asarray(np.isfinite(observed)),
    )
    assert float(elbo) <= float(logz_j) + 2e-6


def test_missing_observations_leave_independent_prior_unchanged():
    prior = np.array([0.2, 0.5, 0.8], dtype=np.float32)
    observed = np.array([np.nan], dtype=np.float32)
    design = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    logz, marginal, pairwise, _, _ = run_exact(observed, prior, design, np.float32(1.0))

    expected_pairwise = np.outer(prior, prior)
    np.fill_diagonal(expected_pairwise, prior)
    assert float(logz) == pytest.approx(0.0, abs=2e-7)
    np.testing.assert_allclose(np.asarray(marginal), prior, rtol=1e-6)
    np.testing.assert_allclose(np.asarray(pairwise), expected_pairwise, rtol=1e-6)
