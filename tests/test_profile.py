import numpy as np

from viprodyne import (
    CAVIConfig,
    ContactThresholdProfileResult,
    MS2Dataset,
    ModelConfig,
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
