import numpy as np
import pytest
from scipy.special import digamma, gammaln

from viprodyne.variational.distributions import DeltaNode, DirichletNode, GammaNode


def test_gamma_node_moments_entropy_and_prior_term():
    node = GammaNode(
        name="loading_rate",
        prior_shape=2.0,
        prior_rate=3.0,
        shape=5.0,
        rate=7.0,
    )

    moments = node.moments()
    assert moments["mean"] == pytest.approx(5.0 / 7.0)
    assert moments["expected_log"] == pytest.approx(float(digamma(5.0) - np.log(7.0)))

    entropy = 5.0 - np.log(7.0) + gammaln(5.0) + (1.0 - 5.0) * digamma(5.0)
    expected_log_prior = (
        2.0 * np.log(3.0)
        - gammaln(2.0)
        + (2.0 - 1.0) * moments["expected_log"]
        - 3.0 * moments["mean"]
    )
    assert node.entropy() == pytest.approx(float(entropy))
    assert node.expected_log_prior() == pytest.approx(float(expected_log_prior))
    assert node.elbo_contribution() == pytest.approx(float(entropy + expected_log_prior))


def test_gamma_node_conjugate_update_and_pinning():
    node = GammaNode(name="R_01", prior_shape=np.array([1.0, 2.0]), prior_rate=np.array([3.0, 4.0]))
    node.set_posterior_from_sufficient_statistics(counts=np.array([5.0, 6.0]), exposure=2.0)

    np.testing.assert_allclose(node.shape, [6.0, 8.0])
    np.testing.assert_allclose(node.rate, [5.0, 6.0])

    node.pin([0.1, 0.2])
    node.set_posterior_from_sufficient_statistics(counts=np.array([10.0, 10.0]), exposure=10.0)
    moments = node.moments()
    np.testing.assert_allclose(moments["mean"], [0.1, 0.2])
    np.testing.assert_allclose(moments["expected_log"], np.log([0.1, 0.2]))
    assert node.entropy() == 0.0
    assert node.elbo_contribution() == 0.0


def test_dirichlet_node_moments_entropy_and_update():
    prior = np.array([1.0, 2.0, 3.0])
    node = DirichletNode(name="pi", prior_concentration=prior)
    node.set_posterior_from_counts(np.array([2.0, 0.0, 1.0]))

    concentration = np.array([3.0, 2.0, 4.0])
    total = np.sum(concentration)
    moments = node.moments()
    np.testing.assert_allclose(moments["mean"], concentration / total)
    np.testing.assert_allclose(moments["expected_log"], digamma(concentration) - digamma(total))

    log_beta = np.sum(gammaln(concentration)) - gammaln(total)
    entropy = log_beta + (total - len(concentration)) * digamma(total)
    entropy -= np.sum((concentration - 1.0) * digamma(concentration))
    assert node.entropy() == pytest.approx(float(entropy))


def test_dirichlet_pin_normalizes_and_skips_updates():
    node = DirichletNode(name="pi", prior_concentration=np.ones(3))
    node.pin(np.array([2.0, 2.0, 4.0]))
    node.set_posterior_from_counts(np.array([100.0, 0.0, 0.0]))

    np.testing.assert_allclose(node.moments()["mean"], [0.25, 0.25, 0.5])
    assert node.entropy() == 0.0
    assert node.elbo_contribution() == 0.0


def test_plated_dirichlet_sampling_returns_one_simplex_per_plate():
    rng = np.random.default_rng(123)
    node = DirichletNode(name="pi", prior_concentration=np.ones((2, 3)))

    sample = node.sample(rng)
    prior_sample = node.sample_prior(rng, size=4)

    assert sample.shape == (2, 3)
    np.testing.assert_allclose(np.sum(sample, axis=-1), np.ones(2))
    assert prior_sample.shape == (4, 2, 3)
    np.testing.assert_allclose(np.sum(prior_sample, axis=-1), np.ones((4, 2)))


def test_delta_node_emits_deterministic_moments():
    node = DeltaNode(name="rc", value=np.array([2.0, 3.0]))

    np.testing.assert_allclose(node.moments()["mean"], [2.0, 3.0])
    np.testing.assert_allclose(node.moments()["expected_log"], np.log([2.0, 3.0]))
    assert node.entropy() == 0.0
    np.testing.assert_allclose(node.sample(size=2), [[2.0, 3.0], [2.0, 3.0]])
