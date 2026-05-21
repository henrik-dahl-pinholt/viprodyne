"""Small contact-threshold profile example.

This mirrors the notebook workflow where an external time-varying score is
thresholded into a driven-transition contact probability and profiled by ELBO.
"""

from __future__ import annotations

import numpy as np

from viprodyne import CAVIConfig, MS2Dataset, ModelConfig, profile_contact_threshold
from viprodyne.core.ms2_kernels import ProximalKernel


def main() -> None:
    rng = np.random.default_rng(2026)
    n_observations = 24
    n_traces = 8
    dt = np.float32(0.5)
    observation_times = (np.arange(n_observations, dtype=np.float32) + 0.5) * dt
    loading_times = np.arange(n_observations, dtype=np.float32) * dt
    contact_score = (
        0.5 + 0.5 * np.sin(np.linspace(0.0, 2.0 * np.pi, n_observations, dtype=np.float32))
    ).astype(np.float32)
    true_threshold = np.float32(0.3)
    contact_probability = (contact_score < true_threshold).astype(np.float32)

    loading_probability = _expected_loading_probability(
        contact_probability=contact_probability,
        dt=dt,
    )
    kernel = ProximalKernel(
        t_rise=np.float32(0.5),
        t_plateau=np.float32(0.5),
        rna_intensity=np.float32(1.0),
    )
    clean_signal = _expected_ms2_signal(
        sampling_times=observation_times,
        loading_times=loading_times,
        loading_probability=loading_probability,
        kernel=kernel,
    )
    observed = clean_signal[None, :] + rng.normal(
        0.0,
        0.08,
        size=(n_traces, n_observations),
    ).astype(np.float32)

    dataset = MS2Dataset(
        observed=observed,
        noise_std=np.float32(0.1),
        dt=dt,
    )
    config = ModelConfig(
        n_states=2,
        pol2_mode="transfer",
        ms2_kernel="proximal",
        t_rise=np.float32(0.5),
        t_plateau=np.float32(0.5),
        rna_intensity=np.float32(1.0),
        driven_transition_indices=(1,),
        contact_drives=(contact_score,),
        driven_rate_initial=np.float32(0.8),
        driven_rate_bounds=(1e-3, 5.0),
    )
    fit_config = CAVIConfig(max_iterations=10, min_iterations=10, tolerance=0.0)
    profile = profile_contact_threshold(
        datasets=(dataset,),
        config=config,
        candidate_values=np.linspace(0.2, 0.8, 7, dtype=np.float32),
        fit_config=fit_config,
    )

    print("candidate thresholds:", profile.candidate_values)
    print("profile ELBOs:", profile.elbos)
    print("best threshold:", profile.best_value)
    print("true threshold:", true_threshold)


def _expected_loading_probability(contact_probability: np.ndarray, dt: np.float32) -> np.ndarray:
    kon = np.float32(1.0)
    koff = np.float32(0.25)
    loading_rates = np.array([0.02, 1.5], dtype=np.float32)
    state = np.array([1.0, 0.0], dtype=np.float32)
    loading_probability = []
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
    return np.asarray(loading_probability, dtype=np.float32)


def _expected_ms2_signal(
    *,
    sampling_times: np.ndarray,
    loading_times: np.ndarray,
    loading_probability: np.ndarray,
    kernel: ProximalKernel,
) -> np.ndarray:
    weights = np.asarray(
        kernel(
            np.asarray(sampling_times, dtype=np.float32)[:, None]
            - np.asarray(loading_times, dtype=np.float32)[None, :]
        ),
        dtype=np.float32,
    )
    return np.asarray(weights @ loading_probability.astype(np.float32), dtype=np.float32)


if __name__ == "__main__":
    main()
