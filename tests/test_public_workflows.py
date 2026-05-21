import numpy as np

from viprodyne import MS2Dataset, ModelConfig, ViprodyneModel


def test_getting_started_public_workflow_smoke():
    dataset = MS2Dataset(
        name="condition_0",
        observed=np.array([[0.1, np.nan, 0.8]], dtype=np.float32),
        noise_std=np.float32(0.5),
        time_grid=np.array([0.0, 0.5, 1.0, 1.5], dtype=np.float32),
    )
    model = ViprodyneModel(
        datasets=(dataset,),
        config=ModelConfig(
            n_states=2,
            ms2_kernel="proximal",
            t_rise=np.float32(0.25),
            t_plateau=np.float32(0.75),
            rna_intensity=np.float32(1.0),
        ),
    )

    fit = model.run_inference(max_iterations=2, min_iterations=2, compute_elbo=True)
    posterior = fit.datasets["condition_0"]

    assert fit.cavi.n_iterations == 2
    assert fit.cavi.elbo.dtype == np.float32
    assert np.isfinite(fit.cavi.elbo)
    assert posterior.observed.shape == (1, 3)
    assert posterior.finite_mask.tolist() == [[True, False, True]]
    assert posterior.time_grid.dtype == np.float32
    assert posterior.state_posterior.shape == (1, 4, 2)
    assert posterior.loading_posterior.shape == (1, 3)
    assert posterior.predicted_signal.shape == (1, 3)
    assert posterior.initial_probabilities.shape == (2,)
    assert set(posterior.transition_rates) == {0, 1}
    assert set(posterior.loading_rates) == {0, 1}
    assert all(rate.dtype == np.float32 for rate in posterior.transition_rates.values())
    assert all(rate.dtype == np.float32 for rate in posterior.loading_rates.values())
