import numpy as np
import pytest

from viprodyne import (
    CAVIConfig,
    ContactDrive,
    ContactThresholdProfileResult,
    MS2Dataset,
    ModelConfig,
    ProximalKernel,
    ViprodyneModel,
    profile_contact_threshold,
)


def test_profile_contact_threshold_runs_candidates_and_returns_best_fit():
    dataset = MS2Dataset(
        name="toy",
        observed=np.array([[0.1, 0.4, 0.8]], dtype=np.float32),
        noise_std=np.float32(0.5),
        time_grid=np.array([0.0, 0.5, 1.0, 1.5], dtype=np.float32),
    )
    config = ModelConfig(
        n_states=2,
        pol2_mode="transfer",
        t_rise=np.float32(0.5),
        t_plateau=np.float32(0.0),
        driven_transition_indices=(1,),
        driven_rate_initial=np.float32(0.5),
        driven_rate_bounds=(1e-3, 5.0),
    )

    profile = profile_contact_threshold(
        datasets=(dataset,),
        config=config,
        contact_scores=np.array([0.2, 0.6, 0.9], dtype=np.float32),
        candidate_values=np.array([0.3, 0.7], dtype=np.float32),
        fit_config=CAVIConfig(max_iterations=2, min_iterations=2, compute_elbo=False),
    )

    assert isinstance(profile, ContactThresholdProfileResult)
    assert profile.candidate_values.dtype == np.float32
    assert profile.elbos.dtype == np.float32
    assert profile.elbos.shape == (2,)
    assert len(profile.fits) == 2
    assert profile.best_index == int(np.argmax(profile.elbos))
    assert profile.best_fit is profile.fits[profile.best_index]
    assert profile.best_value == profile.candidate_values[profile.best_index]
    assert np.all(np.isfinite(profile.elbos))
    assert profile.fits[0].cavi.elbo is not None
    assert "toy" in profile.best_fit.datasets


def test_profile_contact_threshold_allows_candidates_with_no_contact():
    dataset = MS2Dataset(
        name="toy",
        observed=np.array([[0.1, 0.4]], dtype=np.float32),
        noise_std=np.float32(0.5),
        time_grid=np.array([0.0, 0.5, 1.0], dtype=np.float32),
    )
    config = ModelConfig(
        n_states=2,
        pol2_mode="transfer",
        t_rise=np.float32(0.5),
        t_plateau=np.float32(0.0),
        driven_transition_indices=(1,),
    )

    profile = profile_contact_threshold(
        datasets=(dataset,),
        config=config,
        contact_scores=np.array([0.2, 0.6], dtype=np.float32),
        candidate_values=np.array([0.1, 0.5], dtype=np.float32),
        fit_config=CAVIConfig(max_iterations=1, min_iterations=1),
    )

    assert len(profile.fits) == 2
    assert np.all(np.isfinite(profile.elbos))
    assert profile.fits[0].datasets["toy"].contact_probability is not None
    np.testing.assert_allclose(profile.fits[0].datasets["toy"].contact_probability, [0.0, 0.0])


def test_profile_contact_threshold_uses_dataset_score_mapping():
    dataset = MS2Dataset(
        name="toy",
        observed=np.array([[0.1, 0.4]], dtype=np.float32),
        noise_std=np.float32(0.5),
        time_grid=np.array([0.0, 0.5, 1.0], dtype=np.float32),
    )
    config = ModelConfig(
        n_states=2,
        pol2_mode="transfer",
        t_rise=np.float32(0.5),
        t_plateau=np.float32(0.0),
        driven_transition_indices=(1,),
    )

    profile = profile_contact_threshold(
        datasets=(dataset,),
        config=config,
        contact_scores={"toy": np.array([0.2, 0.6], dtype=np.float32)},
        candidate_values=np.array([0.5], dtype=np.float32),
        fit_config=CAVIConfig(max_iterations=1, min_iterations=1),
        less_than=False,
    )

    assert profile.elbos.shape == (1,)
    assert profile.best_fit.datasets["toy"].state_posterior.shape == (1, 3, 2)


def test_profile_contact_threshold_accepts_sampling_times_without_time_grid():
    dataset = MS2Dataset(
        name="toy",
        observed=np.array([[0.1, 0.4, 0.8]], dtype=np.float32),
        noise_std=np.float32(0.5),
        sampling_times=np.array([0.0, 0.5, 1.0], dtype=np.float32),
    )
    config = ModelConfig(
        n_states=2,
        pol2_mode="transfer",
        t_rise=np.float32(0.5),
        t_plateau=np.float32(0.0),
        driven_transition_indices=(1,),
    )

    profile = profile_contact_threshold(
        datasets=(dataset,),
        config=config,
        contact_scores=np.array([0.2, 0.6, 0.9], dtype=np.float32),
        candidate_values=np.array([0.5], dtype=np.float32),
        fit_config=CAVIConfig(max_iterations=1, min_iterations=1, compute_elbo=False),
    )

    fit = profile.best_fit.datasets["toy"]
    np.testing.assert_allclose(
        fit.time_grid,
        np.array([-0.25, 0.25, 0.75, 1.25], dtype=np.float32),
    )
    assert fit.state_posterior.shape == (1, 4, 2)
    assert fit.loading_posterior.shape == (1, 3)


def test_profile_contact_threshold_names_unnamed_datasets_deterministically():
    dataset = MS2Dataset(
        observed=np.array([[0.1, 0.4]], dtype=np.float32),
        noise_std=np.float32(0.5),
        time_grid=np.array([0.0, 0.5, 1.0], dtype=np.float32),
    )
    config = ModelConfig(
        n_states=2,
        pol2_mode="transfer",
        t_rise=np.float32(0.5),
        t_plateau=np.float32(0.0),
        driven_transition_indices=(1,),
    )

    profile = profile_contact_threshold(
        datasets=(dataset,),
        config=config,
        contact_scores=np.array([0.2, 0.6], dtype=np.float32),
        candidate_values=np.array([0.5], dtype=np.float32),
        fit_config=CAVIConfig(max_iterations=1, min_iterations=1, compute_elbo=False),
    )

    assert dataset.name is None
    assert set(profile.best_fit.datasets) == {"dataset_0"}


def test_profile_contact_threshold_recovers_latent_statistics_from_synthetic_data():
    rng = np.random.default_rng(1)
    n_observations = 24
    n_traces = 32
    dt = np.float32(0.5)
    time_grid = np.arange(n_observations + 1, dtype=np.float32) * dt
    observation_times = time_grid[1:]
    loading_times = time_grid[:-1]
    contact_score = (
        0.5 + 0.5 * np.sin(np.linspace(0.0, 2.0 * np.pi, n_observations, dtype=np.float32))
    ).astype(np.float32)
    true_threshold = np.float32(0.3)
    contact_probability = (contact_score < true_threshold).astype(np.float32)
    truth = _contact_threshold_truth(contact_probability, dt)
    kernel = ProximalKernel(
        t_rise=np.float32(0.5),
        t_plateau=np.float32(0.5),
        rna_intensity=np.float32(1.0),
    )
    design = np.asarray(
        kernel(observation_times[:, None] - loading_times[None, :]),
        dtype=np.float32,
    )
    latent_loadings = rng.binomial(
        1,
        truth["loading_probability"],
        size=(n_traces, n_observations),
    ).astype(np.float32)
    clean_signal = latent_loadings @ design.T
    observed = clean_signal + rng.normal(
        0.0,
        0.08,
        size=clean_signal.shape,
    ).astype(np.float32)
    dataset = MS2Dataset(
        name="toy",
        observed=observed,
        noise_std=np.float32(0.08),
        time_grid=time_grid,
    )
    config = ModelConfig(
        n_states=2,
        pol2_mode="transfer",
        ms2_kernel="proximal",
        t_rise=np.float32(0.5),
        t_plateau=np.float32(0.5),
        rna_intensity=np.float32(1.0),
        driven_transition_indices=(1,),
        driven_rate_initial=np.float32(1.0),
        driven_rate_bounds=(1e-3, 5.0),
    )

    profile = profile_contact_threshold(
        datasets=(dataset,),
        config=config,
        contact_scores=contact_score,
        candidate_values=np.array([0.2, 0.3, 0.4, 0.5, 0.6], dtype=np.float32),
        fit_config=CAVIConfig(max_iterations=20, min_iterations=20, tolerance=0.0),
    )
    direct_dataset = MS2Dataset(
        name="toy",
        observed=observed,
        noise_std=np.float32(0.08),
        time_grid=time_grid,
    )
    direct_model = ViprodyneModel(
        datasets=(direct_dataset,),
        config=ModelConfig(
            n_states=2,
            pol2_mode="transfer",
            ms2_kernel="proximal",
            t_rise=np.float32(0.5),
            t_plateau=np.float32(0.5),
            rna_intensity=np.float32(1.0),
            driven_transition_indices=(1,),
            driven_rate_initial=np.float32(1.0),
            driven_rate_bounds=(1e-3, 5.0),
            rc_initial=true_threshold,
            rc_bounds=(0.2, 0.6),
            rc_candidate_values=np.array([0.2, 0.3, 0.4, 0.5, 0.6], dtype=np.float32),
        ),
        contact_drive=ContactDrive.threshold(contact_score),
    )
    direct_fit = direct_model.run_inference(
        CAVIConfig(max_iterations=20, min_iterations=20, tolerance=0.0),
    )
    fit = profile.best_fit.datasets["toy"]
    direct = direct_fit.datasets["toy"]
    loading_posterior = np.asarray(fit.loading_posterior, dtype=np.float32)
    direct_loading_posterior = np.asarray(direct.loading_posterior, dtype=np.float32)
    posterior_state_mean = np.asarray(fit.state_posterior, dtype=np.float32).mean(axis=0)
    direct_state_mean = np.asarray(direct.state_posterior, dtype=np.float32).mean(axis=0)
    posterior_loading_mean = loading_posterior.mean(axis=0)
    direct_loading_mean = direct_loading_posterior.mean(axis=0)

    # This finite marginal-loading toy is used for posterior-statistic recovery;
    # the full profiled ELBO can prefer the stricter neighboring threshold.
    assert profile.best_value <= true_threshold
    assert direct.contact_rc == pytest.approx(true_threshold)
    np.testing.assert_allclose(
        direct.contact_probability,
        contact_probability,
        atol=0.0,
    )
    assert np.all(np.isfinite(profile.elbos))
    assert profile.elbos[profile.best_index] > profile.elbos[-1]
    assert np.mean((loading_posterior > 0.5) == latent_loadings) > 0.98
    assert np.mean((direct_loading_posterior > 0.5) == latent_loadings) > 0.98
    assert _rmse(np.asarray(fit.predicted_signal, dtype=np.float32), clean_signal) < 0.03
    assert _rmse(np.asarray(direct.predicted_signal, dtype=np.float32), clean_signal) < 0.03
    assert _correlation(posterior_loading_mean, truth["loading_probability"]) > 0.93
    assert _correlation(direct_loading_mean, truth["loading_probability"]) > 0.93
    assert np.mean(np.abs(posterior_loading_mean - truth["loading_probability"])) < 0.05
    assert np.mean(np.abs(direct_loading_mean - truth["loading_probability"])) < 0.05
    assert _correlation(posterior_state_mean[:, 1], truth["state_probability"][:, 1]) > 0.95
    assert _correlation(direct_state_mean[:, 1], truth["state_probability"][:, 1]) > 0.95
    assert np.mean(np.abs(posterior_state_mean - truth["state_probability"])) < 0.18
    assert np.mean(np.abs(direct_state_mean - truth["state_probability"])) < 0.18


def _contact_threshold_truth(contact_probability: np.ndarray, dt: np.float32) -> dict[str, np.ndarray]:
    kon = np.float32(1.0)
    koff = np.float32(0.25)
    loading_rates = np.array([0.02, 1.5], dtype=np.float32)
    state = np.array([1.0, 0.0], dtype=np.float32)
    loading_probability = []
    state_probability = [state.copy()]
    for contact in np.asarray(contact_probability, dtype=np.float32):
        loading_probability.append(np.sum(state * (1.0 - np.exp(-loading_rates * dt))))
        p_on = contact * (1.0 - np.exp(-kon * dt))
        p_off = 1.0 - np.exp(-koff * dt)
        state = np.array(
            [
                state[0] * (1.0 - p_on) + state[1] * p_off,
                state[1] * (1.0 - p_off) + state[0] * p_on,
            ],
            dtype=np.float32,
        )
        state = state / np.sum(state)
        state_probability.append(state.copy())
    return {
        "loading_probability": np.asarray(loading_probability, dtype=np.float32),
        "state_probability": np.asarray(state_probability, dtype=np.float32),
    }


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    left = np.ravel(np.asarray(left, dtype=np.float32))
    right = np.ravel(np.asarray(right, dtype=np.float32))
    return float(np.corrcoef(left, right)[0, 1])


def _rmse(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.sqrt(np.mean((left - right) ** 2)))
