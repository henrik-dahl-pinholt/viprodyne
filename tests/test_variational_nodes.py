import numpy as np
import pytest

from viprodyne.core.contact_survival import ContactSurvivalStats
from viprodyne.core.rate_edges import RateEdge
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


def bernoulli_entropy(probabilities):
    q = np.clip(np.asarray(probabilities, dtype=np.float32), 1e-7, 1.0 - 1e-7)
    return -np.sum(q * np.log(q) + (1.0 - q) * np.log1p(-q))


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


def test_driven_rate_map_reads_contact_survival_stats_keyed_by_rate_name():
    stats = ContactSurvivalStats(
        expected_jumps=3.0,
        gamma_from=np.array([0.5, 1.0, 0.5], dtype=np.float32),
        p_contact=np.ones(3, dtype=np.float32),
        dt=0.25,
    )
    driven = DrivenRateMap(
        name="R_contact",
        initial_rate=np.float32(1.0),
        rate_bounds=(1e-4, 100.0),
        prior_shape=2.0,
        prior_rate=1.5,
    )
    child = StatsNode("contact_stats", {"contact_survival_stats_by_rate": {"other": stats}})
    graph = VariationalGraph()
    graph.add_node(driven)
    graph.add_node(child)
    graph.add_edge("R_contact", "contact_stats")

    graph.run_schedule(["R_contact"])

    assert driven.moments()["mean"] == pytest.approx(1.0)

    child.stats = {"contact_survival_stats_by_rate": {"R_contact": stats}}
    graph.moments.publish("contact_stats", child.moments())
    graph.run_schedule(["R_contact"])

    expected_map = (3.0 + 2.0 - 1.0) / (1.5 + np.sum(stats.gamma_from) * 0.25)
    assert driven.moments()["mean"] == pytest.approx(expected_map, rel=5e-4)


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


def test_promoter_state_uses_expected_log_rates_and_exit_potentials():
    graph = VariationalGraph()
    pi = InitialStateProb(
        name="pi",
        prior_concentration=np.ones(2, dtype=np.float32),
        pinned_value=np.array([1.0, 0.0], dtype=np.float32),
    )
    kon = TransitionRate(
        name="kon",
        prior_shape=2.0,
        prior_rate=4.0,
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
        time_grid=np.array([0.0, 0.5, 1.0], dtype=np.float32),
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

    kon_tilde = np.exp(graph.moments.get("kon")["expected_log"])
    kon_mean = graph.moments.get("kon")["mean"]
    generator = promoter.tilted_generator
    potentials = promoter.tilt_potentials
    assert generator.dtype == np.float32
    assert potentials.dtype == np.float32
    np.testing.assert_allclose(generator[:, 1, 0], kon_tilde, rtol=2e-6)
    np.testing.assert_allclose(generator[:, 0, 1], 0.2, rtol=2e-6)
    np.testing.assert_allclose(np.sum(generator, axis=-2), np.zeros((2, 2)), atol=2e-7)
    np.testing.assert_allclose(potentials[:, 0], kon_tilde - kon_mean, rtol=2e-6)
    np.testing.assert_allclose(potentials[:, 1], 0.0, atol=2e-7)


def test_promoter_state_uses_pol2_loading_child_message():
    graph = VariationalGraph()
    pi = InitialStateProb(
        name="pi",
        prior_concentration=np.ones(2, dtype=np.float32),
        pinned_value=np.array([0.5, 0.5], dtype=np.float32),
    )
    kon = TransitionRate(
        name="kon",
        prior_shape=1.0,
        prior_rate=1.0,
        pinned_value=np.float32(0.1),
        n_states=2,
        to_state=1,
        from_state=0,
    )
    koff = TransitionRate(
        name="koff",
        prior_shape=1.0,
        prior_rate=1.0,
        pinned_value=np.float32(0.1),
        n_states=2,
        to_state=0,
        from_state=1,
    )
    r0 = LoadingRate(
        name="r0",
        prior_shape=1.0,
        prior_rate=1.0,
        pinned_value=np.float32(0.4),
        state_index=0,
    )
    r1 = LoadingRate(
        name="r1",
        prior_shape=1.0,
        prior_rate=1.0,
        pinned_value=np.float32(2.0),
        state_index=1,
    )
    tau = StatsNode("tau", {"load_probabilities": np.array([0.25, 0.75], dtype=np.float32)})
    promoter = PromoterState(
        name="s",
        time_grid=np.array([0.0, 0.5, 1.0], dtype=np.float32),
        n_states=2,
        rate_edges=(kon.edge, koff.edge),
        initial_probability_node="pi",
    )
    for node in [pi, kon, koff, r0, r1, tau, promoter]:
        graph.add_node(node)
    graph.add_edge("pi", "s")
    graph.add_edge("kon", "s")
    graph.add_edge("koff", "s")
    graph.add_edge("s", "tau")
    graph.add_edge("r0", "tau")
    graph.add_edge("r1", "tau")

    graph.run_schedule(["s"])

    dt = np.array([0.5, 0.5], dtype=np.float32)
    load_probabilities = np.array([0.25, 0.75], dtype=np.float32)
    rates = np.array([0.4, 2.0], dtype=np.float32)
    log_load = np.log(-np.expm1(-dt[:, None] * rates[None, :]))
    log_no_load = -dt[:, None] * rates[None, :]
    expected_potentials = (
        load_probabilities[:, None] * log_load
        + (1.0 - load_probabilities[:, None]) * log_no_load
    ) / dt[:, None]

    np.testing.assert_allclose(promoter.tilt_potentials, expected_potentials, rtol=2e-6)


def test_promoter_state_uses_sampler_expected_count_child_message():
    graph = VariationalGraph()
    pi = InitialStateProb(
        name="pi",
        prior_concentration=np.ones(2, dtype=np.float32),
        pinned_value=np.array([0.5, 0.5], dtype=np.float32),
    )
    kon = TransitionRate(
        name="kon",
        prior_shape=1.0,
        prior_rate=1.0,
        pinned_value=np.float32(0.1),
        n_states=2,
        to_state=1,
        from_state=0,
    )
    koff = TransitionRate(
        name="koff",
        prior_shape=1.0,
        prior_rate=1.0,
        pinned_value=np.float32(0.1),
        n_states=2,
        to_state=0,
        from_state=1,
    )
    r0 = LoadingRate(
        name="r0",
        prior_shape=1.0,
        prior_rate=1.0,
        pinned_value=np.float32(0.4),
        state_index=0,
    )
    r1 = LoadingRate(
        name="r1",
        prior_shape=1.0,
        prior_rate=1.0,
        pinned_value=np.float32(2.0),
        state_index=1,
    )
    tau = StatsNode("tau", {"expected_loading_counts": np.array([0.2, 1.1], dtype=np.float32)})
    promoter = PromoterState(
        name="s",
        time_grid=np.array([0.0, 0.5, 1.0], dtype=np.float32),
        n_states=2,
        rate_edges=(kon.edge, koff.edge),
        initial_probability_node="pi",
    )
    for node in [pi, kon, koff, r0, r1, tau, promoter]:
        graph.add_node(node)
    graph.add_edge("pi", "s")
    graph.add_edge("kon", "s")
    graph.add_edge("koff", "s")
    graph.add_edge("s", "tau")
    graph.add_edge("r0", "tau")
    graph.add_edge("r1", "tau")

    graph.run_schedule(["s"])

    dt = np.array([0.5, 0.5], dtype=np.float32)
    counts = np.array([0.2, 1.1], dtype=np.float32)
    rates = np.array([0.4, 2.0], dtype=np.float32)
    expected_potentials = (counts[:, None] * np.log(rates[None, :]) - dt[:, None] * rates) / dt[
        :, None
    ]

    np.testing.assert_allclose(promoter.tilt_potentials, expected_potentials, rtol=2e-6)


def test_polymerase_loadings_sampler_mode_smoke_runs_and_emits_counts():
    node = PolymeraseLoadings(
        name="tau_sampler",
        observed=np.array([0.2, np.nan, 0.8], dtype=np.float32),
        noise_std=np.float32(0.5),
        mode="sampler",
        sampling_times=np.array([0.0, 0.5, 1.0], dtype=np.float32),
        fine_grid=np.array([0.0, 0.5, 1.0], dtype=np.float32),
        sampler_rates_on_grid=np.array([0.7, 0.8, 0.6], dtype=np.float32),
        rise_time=np.float32(0.5),
        plateau_time=np.float32(0.0),
        rna_intensity=np.float32(1.0),
        sampler_seed=1,
        sampler_iterations=16,
        sampler_repeats=1,
        sampler_compute_elbo=True,
        sampler_elbo_iterations=12,
        sampler_elbo_steps=2,
        sampler_elbo_repeats=1,
    )

    moments = node.moments()

    assert moments["posterior_rate"].dtype == np.float32
    assert moments["expected_loading_counts"].dtype == np.float32
    assert moments["load_probabilities"].dtype == np.float32
    assert moments["posterior_rate"].shape == (1, 3)
    assert np.all(np.isfinite(moments["posterior_rate"]))
    assert np.isfinite(moments["log_partition"])


def test_promoter_state_applies_contact_drive_and_emits_survival_stats():
    graph = VariationalGraph()
    pi = InitialStateProb(
        name="pi",
        prior_concentration=np.ones(2, dtype=np.float32),
        pinned_value=np.array([1.0, 0.0], dtype=np.float32),
    )
    kon = DrivenRateMap(
        name="kon_contact",
        initial_rate=np.float32(0.8),
        rate_bounds=(1e-4, 10.0),
        pinned_value=np.float32(0.8),
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
    drive = StatsNode("rc", {"p_contact": np.array([0.25, 0.75], dtype=np.float32)})
    promoter = PromoterState(
        name="s",
        time_grid=np.array([0.0, 0.5, 1.0], dtype=np.float32),
        n_states=2,
        rate_edges=(
            RateEdge(
                n_states=2,
                to_state=1,
                from_state=0,
                rate_node="kon_contact",
                drive_node="rc",
            ),
            koff.edge,
        ),
        initial_probability_node="pi",
    )
    for node in [pi, kon, koff, drive, promoter]:
        graph.add_node(node)
    graph.add_edge("pi", "s")
    graph.add_edge("kon_contact", "s")
    graph.add_edge("koff", "s")
    graph.add_edge("rc", "s")

    graph.run_schedule(["s"])

    p_contact = np.array([0.25, 0.75], dtype=np.float32)
    dt = np.float32(0.5)
    q_rate = p_contact * np.float32(0.8)
    effective_rate = -np.log1p(-p_contact * (-np.expm1(-np.float32(0.8) * dt))) / dt
    np.testing.assert_allclose(promoter.tilted_generator[:, 1, 0], q_rate, rtol=2e-6)
    np.testing.assert_allclose(
        promoter.tilt_potentials[:, 0],
        q_rate - effective_rate,
        rtol=2e-6,
        atol=2e-7,
    )
    np.testing.assert_allclose(
        np.sum(promoter.tilted_generator, axis=-2),
        np.zeros((2, 2), dtype=np.float32),
        atol=2e-7,
    )
    stats = graph.moments.get("s")["contact_survival_stats_by_rate"]["kon_contact"]
    assert stats.p_contact.dtype == np.float32
    assert stats.p_contact.shape == (1, 2)
    np.testing.assert_allclose(stats.p_contact[0], p_contact)
    assert np.isfinite(stats.expected_jumps)


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
    expected_entropy = bernoulli_entropy(expected_posterior)
    assert exact.entropy() == pytest.approx(float(expected_entropy), rel=1e-6)
    assert mean_field.entropy() == pytest.approx(float(expected_entropy), rel=5e-5)
    assert transfer.entropy() == pytest.approx(float(expected_entropy), rel=2e-4)
    assert exact.moments()["entropy"].dtype == np.float32
    assert mean_field.moments()["entropy"].dtype == np.float32
    assert transfer.moments()["entropy"].dtype == np.float32
    np.testing.assert_allclose(
        transfer.moments()["load_probabilities"],
        expected_posterior,
        rtol=2e-4,
        atol=2e-5,
    )
    with pytest.raises(NotImplementedError, match="joint samples"):
        transfer.sample()
    assert exact.elbo_contribution() == pytest.approx(float(expected_logz), rel=1e-6)
    assert mean_field.elbo_contribution() == pytest.approx(float(expected_logz), rel=5e-6)
    assert transfer.elbo_contribution() == pytest.approx(float(expected_logz), rel=1e-6)


def test_exact_polymerase_entropy_matches_log_partition_identity():
    observed = np.array([1.1], dtype=np.float32)
    prior = np.array([0.35, 0.65], dtype=np.float32)
    design = np.array([[1.0, 0.7]], dtype=np.float32)
    noise = np.float32(0.5)
    node = PolymeraseLoadings(
        name="tau_exact_interacting",
        observed=observed,
        prior_probabilities=prior,
        design_matrix=design,
        noise_std=noise,
        mode="exact",
    )
    configs = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    means = configs @ design[0]
    log_prior = configs @ np.log(prior) + (1.0 - configs) @ np.log1p(-prior)
    log_likelihood = -0.5 * (
        np.log(np.float32(2.0 * np.pi) * noise**2) + (observed[0] - means) ** 2 / noise**2
    )
    log_psi = log_prior + log_likelihood
    weights = np.asarray(node.posterior_probabilities, dtype=np.float32)
    entropy_from_logz = float(node.elbo_contribution() - np.sum(weights * log_psi))

    assert node.entropy() == pytest.approx(entropy_from_logz, rel=2e-6)
