import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import logsumexp

from viprodyne.core.bernoulli_transfer_pol2 import (
    bernoulli_transfer_log_likelihood,
    bernoulli_transfer_log_likelihood_batch,
    build_ms2_design_matrix,
    enumerate_binary_configurations,
    exact_bernoulli_posterior,
    mean_field_bernoulli_elbo,
)
from viprodyne.core.mf_pol2_finder import fit_mean_field_bernoulli


def gaussian_logpdf(value, mean, noise):
    return -0.5 * (np.log(2.0 * np.pi * noise**2) + (value - mean) ** 2 / noise**2)


def independent_loading_theory(observed, prior, loading_weight, noise):
    observed = np.asarray(observed, dtype=np.float32)
    prior = np.asarray(prior, dtype=np.float32)
    loading_weight = np.broadcast_to(np.asarray(loading_weight, dtype=np.float32), prior.shape)
    noise = np.broadcast_to(np.asarray(noise, dtype=np.float32), observed.shape)
    finite_mask = np.isfinite(observed)
    safe_observed = np.where(finite_mask, observed, 0.0)

    logp0 = np.log1p(-prior) + gaussian_logpdf(safe_observed, 0.0, noise)
    logp1 = np.log(prior) + gaussian_logpdf(safe_observed, loading_weight, noise)
    local_logz = np.logaddexp(logp0, logp1)
    posterior = np.exp(logp1 - local_logz)
    posterior = np.where(finite_mask, posterior, prior).astype(np.float32)
    logz = np.sum(np.where(finite_mask, local_logz, 0.0), axis=-1)
    return np.asarray(logz, dtype=np.float32), posterior


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


def design_from_windows(n_loadings, window_weights, starts):
    design = np.zeros((len(starts), n_loadings), dtype=np.float32)
    for row, start in enumerate(starts):
        design[row, start : start + len(window_weights)] = np.asarray(
            window_weights,
            dtype=np.float32,
        )
    return design


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


def test_transfer_log_likelihood_matches_exact_contiguous_windows():
    prior = np.array([0.2, 0.55, 0.3, 0.75, 0.4], dtype=np.float32)
    window_weights = np.array([0.25, 1.0, 0.5], dtype=np.float32)
    starts = np.array([0, 1, 2], dtype=np.int32)
    observed = np.array([0.4, 1.2, 0.7], dtype=np.float32)
    noise = np.float32(0.8)
    finite_mask = np.isfinite(observed)
    design = design_from_windows(len(prior), window_weights, starts)

    configs = enumerate_binary_configurations(len(prior))
    exact_logz, *_ = exact_bernoulli_posterior(
        jnp.asarray(observed),
        jnp.asarray(prior),
        jnp.asarray(design),
        jnp.asarray(noise),
        jnp.asarray(finite_mask),
        configs,
    )
    transfer_logz = bernoulli_transfer_log_likelihood(
        jnp.asarray(observed),
        jnp.asarray(prior),
        jnp.asarray(window_weights),
        jnp.asarray(starts),
        jnp.asarray(noise),
        jnp.asarray(finite_mask),
    )

    assert isinstance(transfer_logz, jax.Array)
    assert transfer_logz.dtype == jnp.float32
    assert float(transfer_logz) == pytest.approx(float(exact_logz), rel=1e-6, abs=1e-6)


def test_transfer_log_likelihood_matches_exact_with_gaps_and_missing_data():
    prior = np.array([0.35, 0.2, 0.6, 0.45, 0.7, 0.25, 0.8], dtype=np.float32)
    window_weights = np.array([1.2, 0.4], dtype=np.float32)
    starts = np.array([1, 3, 5], dtype=np.int32)
    observed = np.array([0.9, np.nan, 0.55], dtype=np.float32)
    noise = np.array([0.7, 0.9, 0.6], dtype=np.float32)
    finite_mask = np.isfinite(observed)
    design = design_from_windows(len(prior), window_weights, starts)

    configs = enumerate_binary_configurations(len(prior))
    exact_logz, *_ = exact_bernoulli_posterior(
        jnp.asarray(observed),
        jnp.asarray(prior),
        jnp.asarray(design),
        jnp.asarray(noise),
        jnp.asarray(finite_mask),
        configs,
    )
    transfer_logz = bernoulli_transfer_log_likelihood(
        jnp.asarray(observed),
        jnp.asarray(prior),
        jnp.asarray(window_weights),
        jnp.asarray(starts),
        jnp.asarray(noise),
        jnp.asarray(finite_mask),
    )

    assert float(transfer_logz) == pytest.approx(float(exact_logz), rel=1e-6, abs=1e-6)


def test_transfer_log_likelihood_batch_matches_single_trajectory_calls():
    window_weights = np.array([0.6, 1.1, 0.2], dtype=np.float32)
    starts = np.array([0, 2], dtype=np.int32)
    observed = np.array(
        [
            [0.5, 1.4],
            [1.1, np.nan],
        ],
        dtype=np.float32,
    )
    priors = np.array(
        [
            [0.2, 0.5, 0.7, 0.4, 0.6],
            [0.8, 0.3, 0.4, 0.65, 0.25],
        ],
        dtype=np.float32,
    )
    noise = np.array([0.5, 0.9], dtype=np.float32)
    finite_mask = np.isfinite(observed)

    batch_logz = bernoulli_transfer_log_likelihood_batch(
        jnp.asarray(observed),
        jnp.asarray(priors),
        jnp.asarray(window_weights),
        jnp.asarray(starts),
        jnp.asarray(noise),
        jnp.asarray(finite_mask),
    )
    single_logz = jnp.stack(
        [
            bernoulli_transfer_log_likelihood(
                jnp.asarray(observed[0]),
                jnp.asarray(priors[0]),
                jnp.asarray(window_weights),
                jnp.asarray(starts),
                jnp.asarray(noise[0]),
                jnp.asarray(finite_mask[0]),
            ),
            bernoulli_transfer_log_likelihood(
                jnp.asarray(observed[1]),
                jnp.asarray(priors[1]),
                jnp.asarray(window_weights),
                jnp.asarray(starts),
                jnp.asarray(noise[1]),
                jnp.asarray(finite_mask[1]),
            ),
        ]
    )

    assert batch_logz.dtype == jnp.float32
    np.testing.assert_allclose(np.asarray(batch_logz), np.asarray(single_logz), rtol=1e-6)


def test_noninteracting_kernel_matches_analytic_theory_for_posteriors_and_elbo():
    sampling_times = jnp.asarray([0.0, 1.0, 2.0, 3.0], dtype=jnp.float32)
    loading_grid = sampling_times
    loading_weight = np.float32(1.7)
    design = build_ms2_design_matrix(
        sampling_times=sampling_times,
        loading_grid=loading_grid,
        kernel=lambda t: jnp.where((t >= 0.0) & (t < 0.4), loading_weight, 0.0),
    )
    observed = np.array([0.15, 1.9, 0.7, 1.35], dtype=np.float32)
    prior = np.array([0.2, 0.45, 0.7, 0.35], dtype=np.float32)
    noise = np.float32(0.55)
    finite_mask = np.isfinite(observed)
    configs = enumerate_binary_configurations(len(prior))

    np.testing.assert_allclose(
        np.asarray(design),
        loading_weight * np.eye(len(prior), dtype=np.float32),
    )
    expected_logz, expected_posterior = independent_loading_theory(
        observed,
        prior,
        loading_weight,
        noise,
    )
    expected_pairwise = np.outer(expected_posterior, expected_posterior).astype(np.float32)
    np.fill_diagonal(expected_pairwise, expected_posterior)

    exact_logz, marginals, pairwise, predicted, _ = exact_bernoulli_posterior(
        jnp.asarray(observed),
        jnp.asarray(prior),
        design,
        jnp.asarray(noise),
        jnp.asarray(finite_mask),
        configs,
    )
    marginals = np.asarray(marginals).copy()
    pairwise = np.asarray(pairwise).copy()
    predicted = np.asarray(predicted).copy()
    elbo = mean_field_bernoulli_elbo(
        jnp.asarray(expected_posterior),
        jnp.asarray(observed),
        jnp.asarray(prior),
        design,
        jnp.asarray(noise),
        jnp.asarray(finite_mask),
    )
    transfer_logz = bernoulli_transfer_log_likelihood(
        jnp.asarray(observed),
        jnp.asarray(prior),
        jnp.asarray([loading_weight], dtype=jnp.float32),
        jnp.arange(len(prior), dtype=jnp.int32),
        jnp.asarray(noise),
        jnp.asarray(finite_mask),
    )

    assert float(exact_logz) == pytest.approx(float(expected_logz), rel=1e-6, abs=1e-6)
    assert float(elbo) == pytest.approx(float(expected_logz), rel=1e-6, abs=1e-6)
    assert float(transfer_logz) == pytest.approx(float(expected_logz), rel=1e-6, abs=1e-6)
    np.testing.assert_allclose(marginals, expected_posterior, rtol=1e-6)
    np.testing.assert_allclose(pairwise, expected_pairwise, rtol=1e-6)
    np.testing.assert_allclose(
        predicted,
        loading_weight * expected_posterior,
        rtol=1e-6,
    )

    mean_field_result = fit_mean_field_bernoulli(observed, prior, np.asarray(design), noise)
    assert mean_field_result.elbo == pytest.approx(float(expected_logz), rel=5e-6, abs=5e-6)
    np.testing.assert_allclose(
        mean_field_result.load_probabilities,
        expected_posterior,
        rtol=5e-5,
        atol=5e-6,
    )


def test_large_noninteracting_batch_runs_without_materializing_state_space():
    n_tracks = 200
    n_timepoints = 1000
    rng = np.random.default_rng(20260520)
    loading_weight = np.float32(1.25)
    prior = rng.uniform(0.05, 0.95, size=(n_tracks, n_timepoints)).astype(np.float32)
    observed = rng.normal(
        loc=loading_weight * (prior > 0.5),
        scale=0.35,
        size=(n_tracks, n_timepoints),
    ).astype(np.float32)
    observed[:, ::137] = np.nan
    noise = np.linspace(0.45, 0.75, n_tracks, dtype=np.float32)
    finite_mask = np.isfinite(observed)
    starts = jnp.arange(n_timepoints, dtype=jnp.int32)

    expected_logz, posterior = independent_loading_theory(
        observed,
        prior,
        loading_weight,
        noise[:, None],
    )
    transfer_logz = bernoulli_transfer_log_likelihood_batch(
        jnp.asarray(observed),
        jnp.asarray(prior),
        jnp.asarray([loading_weight], dtype=jnp.float32),
        starts,
        jnp.asarray(noise),
        jnp.asarray(finite_mask),
    )
    shared_design = jnp.eye(n_timepoints, dtype=jnp.float32) * loading_weight
    batch_elbo = jax.jit(
        jax.vmap(mean_field_bernoulli_elbo, in_axes=(0, 0, 0, None, 0, 0))
    )
    elbo = batch_elbo(
        jnp.asarray(posterior),
        jnp.asarray(observed),
        jnp.asarray(prior),
        shared_design,
        jnp.asarray(noise),
        jnp.asarray(finite_mask),
    )

    assert transfer_logz.shape == (n_tracks,)
    assert transfer_logz.dtype == jnp.float32
    assert elbo.shape == (n_tracks,)
    assert elbo.dtype == jnp.float32
    assert np.all(np.isfinite(np.asarray(transfer_logz)))
    assert np.all(np.isfinite(np.asarray(elbo)))
    np.testing.assert_allclose(np.asarray(transfer_logz), expected_logz, rtol=2e-5, atol=2e-3)
    np.testing.assert_allclose(np.asarray(elbo), expected_logz, rtol=2e-5, atol=2e-3)
