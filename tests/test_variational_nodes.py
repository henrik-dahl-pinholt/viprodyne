import numpy as np
import pytest

from viprodyne.core.contact_survival import ContactSurvivalStats
from viprodyne.variational import (
    DrivenRateMap,
    InitialStateProb,
    LoadingRate,
    ObservedIntensity,
    PolymeraseLoadings,
    PromoterState,
    RcNode,
    TransitionRate,
    VariationalGraph,
    VariationalNode,
)


class StatsNode(VariationalNode):
    def __init__(self, name, stats):
        super().__init__(name)
        self.stats = stats

    def moments(self):
        return self.stats

    def entropy(self):
        return 0.0

    def sample(self, rng=None, size=None):
        return self.stats


def analytic_independent_posterior(observed, prior, weight, noise):
    logp0 = np.log1p(-prior) - 0.5 * (
        np.log(np.float32(2.0 * np.pi) * noise**2) + observed**2 / noise**2
    )
    logp1 = np.log(prior) - 0.5 * (
        np.log(np.float32(2.0 * np.pi) * noise**2) + (observed - weight) ** 2 / noise**2
    )
    logz = np.logaddexp(logp0, logp1)
    return np.exp(logp1 - logz).astype(np.float32), np.sum(logz, dtype=np.float32)


def test_observed_intensity_emits_float32_data_and_mask():
    node = ObservedIntensity(
        name="I",
        observed=np.array([0.2, np.nan, 1.0], dtype=np.float32),
        noise_std=np.float32(0.5),
    )

    moments = node.moments()
    assert moments["observed"].dtype == np.float32
    assert moments["noise_std"].dtype == np.float32
    np.testing.assert_array_equal(moments["finite_mask"], [True, False, True])
    np.testing.assert_allclose(node.sample(size=2)[:, [0, 2]], [[0.2, 1.0], [0.2, 1.0]])


def test_parameter_nodes_update_from_graph_child_statistics():
    graph = VariationalGraph()
    initial = InitialStateProb(name="pi", prior_concentration=np.ones(3, dtype=np.float32))
    loading = LoadingRate(name="r_on", prior_shape=1.0, prior_rate=2.0)
    transition = TransitionRate(
        name="R_10",
        prior_shape=2.0,
        prior_rate=3.0,
        n_states=2,
        to_state=1,
        from_state=0,
    )
    stats = StatsNode(
        "stats",
        {
            "initial_state_counts": np.array([2.0, 0.0, 1.0], dtype=np.float32),
            "loading_counts": np.float32(4.0),
            "loading_exposure": np.float32(2.0),
            "transition_counts": np.array([[0.0, 0.0], [5.0, 0.0]], dtype=np.float32),
            "transition_exposure": np.array([10.0, 3.0], dtype=np.float32),
        },
    )
    for node in [initial, loading, transition, stats]:
        graph.add_node(node)
    graph.add_edge("pi", "stats")
    graph.add_edge("r_on", "stats")
    graph.add_edge("R_10", "stats")

    graph.run_schedule(["pi", "r_on", "R_10"])

    np.testing.assert_allclose(initial.concentration, [3.0, 1.0, 2.0])
    assert initial.concentration.dtype == np.float32
    assert loading.shape == pytest.approx(np.float32(5.0))
    assert loading.rate == pytest.approx(np.float32(4.0))
    assert transition.shape == pytest.approx(np.float32(7.0))
    assert transition.rate == pytest.approx(np.float32(13.0))


def test_driven_rate_map_updates_from_contact_survival_stats():
    gamma_from = np.array([0.5, 1.0, 0.5], dtype=np.float32)
    stats = ContactSurvivalStats(
        expected_jumps=3.0,
        gamma_from=gamma_from,
        p_contact=np.ones_like(gamma_from),
        dt=0.25,
    )
    driven = DrivenRateMap(
        name="R_contact",
        initial_rate=np.float32(1.0),
        rate_bounds=(1e-4, 100.0),
        prior_shape=2.0,
        prior_rate=1.5,
    )
    child = StatsNode("contact_stats", {"contact_survival_stats": stats})
    graph = VariationalGraph()
    graph.add_node(driven)
    graph.add_node(child)
    graph.add_edge("R_contact", "contact_stats")

    graph.run_schedule(["R_contact"])

    expected_map = (3.0 + 2.0 - 1.0) / (1.5 + np.sum(gamma_from) * 0.25)
    moments = driven.moments()
    assert moments["mean"].dtype == np.float32
    assert moments["expected_log"].dtype == np.float32
    assert moments["is_driven"] is True
    assert driven.sample().dtype == np.float32
    assert moments["mean"] == pytest.approx(expected_map, rel=5e-4)


def test_rc_node_emits_contact_probability_and_can_optimize_map_value():
    time_grid = np.linspace(0.0, 3.0, 4, dtype=np.float32)

    def contact_probability(times, rc):
        return np.exp(-times / rc).astype(np.float32)

    def objective(rc, context):
        del context
        return -float((rc - np.float32(2.0)) ** 2)

    node = RcNode(
        name="rc",
        value=np.float32(1.0),
        time_grid=time_grid,
        contact_probability_fn=contact_probability,
        bounds=(0.25, 5.0),
        objective_fn=objective,
    )
    graph = VariationalGraph()
    graph.add_node(node)

    graph.run_schedule(["rc"])
    moments = node.moments()

    assert moments["mean"].dtype == np.float32
    assert moments["p_contact"].dtype == np.float32
    assert moments["mean"] == pytest.approx(2.0, rel=1e-4)
    np.testing.assert_allclose(moments["p_contact"], np.exp(-time_grid / np.float32(2.0)))


def test_promoter_state_uses_parent_rates_and_emits_sufficient_statistics():
    graph = VariationalGraph()
    pi = InitialStateProb(
        name="pi",
        prior_concentration=np.ones(2, dtype=np.float32),
        pinned_value=np.array([1.0, 0.0], dtype=np.float32),
    )
    kon = TransitionRate(
        name="kon",
        prior_shape=1.0,
        prior_rate=1.0,
        pinned_value=np.float32(0.7),
        n_states=2,
        to_state=1,
        from_state=0,
    )
    koff = TransitionRate(
        name="koff",
        prior_shape=1.0,
        prior_rate=1.0,
        pinned_value=np.float32(0.2),
        n_states=2,
        to_state=0,
        from_state=1,
    )
    promoter = PromoterState(
        name="s",
        time_grid=np.linspace(0.0, 2.0, 5, dtype=np.float32),
        n_states=2,
        rate_edges=(kon.edge, koff.edge),
        initial_probability_node="pi",
    )
    for node in [pi, kon, koff, promoter]:
        graph.add_node(node)
    graph.add_edge("pi", "s")
    graph.add_edge("kon", "s")
    graph.add_edge("koff", "s")

    graph.run_schedule(["s"])
    moments = graph.moments.get("s")

    total = np.float32(0.9)
    expected_on = np.float32(0.7) / total * (1.0 - np.exp(-total * np.float32(2.0)))
    assert moments["posterior"].dtype == np.float32
    assert moments["expected_occupancy"].dtype == np.float32
    assert moments["expected_jumps"].dtype == np.float32
    assert moments["posterior"].shape == (1, 5, 2)
    assert moments["posterior"][0, -1, 1] == pytest.approx(expected_on, rel=3e-6)
    assert moments["transition_counts"].shape == (1, 2, 2)
    assert moments["transition_exposure"].shape == (1, 2)


def test_polymerase_loadings_exact_and_mean_field_match_independent_theory():
    observed = np.array([0.1, 1.4, 0.6], dtype=np.float32)
    prior = np.array([0.2, 0.55, 0.7], dtype=np.float32)
    weight = np.float32(1.3)
    noise = np.float32(0.4)
    design = np.eye(3, dtype=np.float32) * weight
    expected_posterior, expected_logz = analytic_independent_posterior(
        observed,
        prior,
        weight,
        noise,
    )

    exact = PolymeraseLoadings(
        name="tau_exact",
        observed=observed,
        prior_probabilities=prior,
        design_matrix=design,
        noise_std=noise,
        mode="exact",
    )
    mean_field = PolymeraseLoadings(
        name="tau_mf",
        observed=observed,
        prior_probabilities=prior,
        design_matrix=design,
        noise_std=noise,
        mode="mean_field",
    )
    transfer = PolymeraseLoadings(
        name="tau_transfer",
        observed=observed,
        prior_probabilities=prior,
        noise_std=noise,
        mode="transfer",
        window_weights=np.array([weight], dtype=np.float32),
        observation_starts=np.arange(3, dtype=np.int32),
    )

    np.testing.assert_allclose(exact.moments()["load_probabilities"], expected_posterior, rtol=1e-6)
    np.testing.assert_allclose(
        mean_field.moments()["load_probabilities"],
        expected_posterior,
        rtol=5e-5,
        atol=2e-5,
    )
    assert exact.elbo_contribution() == pytest.approx(float(expected_logz), rel=1e-6)
    assert mean_field.elbo_contribution() == pytest.approx(float(expected_logz), rel=5e-6)
    assert transfer.elbo_contribution() == pytest.approx(float(expected_logz), rel=1e-6)
