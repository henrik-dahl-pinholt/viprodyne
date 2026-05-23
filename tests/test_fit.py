import numpy as np

from viprodyne import (
    CAVIConfig,
    DatasetInferenceResult,
    MS2Dataset,
    ModelConfig,
    ModelInferenceResult,
    ViprodyneModel,
    run_cavi,
)


def make_model():
    dataset = MS2Dataset(
        name="track_0",
        observed=np.array([[0.2, 0.9]], dtype=np.float32),
        noise_std=np.float32(0.5),
    )
    return ViprodyneModel(
        datasets=(dataset,),
        config=ModelConfig(
            n_states=2,
            time_grid=np.array([0.0, 0.5, 1.0], dtype=np.float32),
            t_rise=np.float32(0.25),
            t_plateau=np.float32(0.5),
        ),
    )


def test_cavi_runs_schedule_and_computes_elbo_once():
    model = make_model()
    calls = {"elbo": 0}
    original_compute_elbo = model.compute_elbo

    def counting_compute_elbo():
        calls["elbo"] += 1
        return original_compute_elbo()

    model.compute_elbo = counting_compute_elbo

    result = run_cavi(
        model,
        CAVIConfig(
            max_iterations=3,
            min_iterations=3,
            tolerance=0.0,
            compute_elbo=True,
        ),
    )

    assert result.n_iterations == 3
    assert len(result.history) == 3
    assert calls["elbo"] == 1
    assert result.elbo.dtype == np.float32
    assert np.isfinite(result.elbo)
    assert result.max_parameter_change.dtype == np.float32
    assert "track_0:s" in result.schedule
    assert "track_0:r0" in result.parameter_nodes
    assert "CAVIResult(" in str(result)


def test_model_fit_cavi_updates_loading_rates_from_pol2_blanket_stats():
    model = make_model()
    before = np.asarray(model.graph.nodes["track_0:r0"].moments()["mean"], dtype=np.float32)

    result = model.fit_cavi(max_iterations=2, min_iterations=2, compute_elbo=False)
    after = np.asarray(model.graph.nodes["track_0:r0"].moments()["mean"], dtype=np.float32)
    pol2_moments = model.graph.moments.get("track_0:tau")

    assert result.elbo is None
    assert "loading_counts_by_rate" in pol2_moments
    assert "track_0:r0" in pol2_moments["loading_counts_by_rate"]
    assert not np.allclose(after, before)


def test_model_run_inference_returns_structured_outputs():
    model = make_model()

    result = model.run_inference(max_iterations=2, min_iterations=2, compute_elbo=True)
    dataset = result.datasets["track_0"]

    assert isinstance(result, ModelInferenceResult)
    assert isinstance(dataset, DatasetInferenceResult)
    assert result.cavi.n_iterations == 2
    assert result.cavi.elbo.dtype == np.float32
    assert result.elbo_terms is not None
    assert dataset.name == "track_0"
    assert dataset.observed.shape == (1, 2)
    assert dataset.finite_mask.shape == (1, 2)
    assert dataset.state_posterior.shape == (1, 2, 2)
    assert dataset.loading_posterior.shape == (1, 2)
    assert dataset.predicted_signal.shape == (1, 2)
    assert dataset.loading_mask.shape == (1, 2)
    assert dataset.initial_probabilities.shape == (2,)
    assert set(dataset.transition_rates) == {0, 1}
    assert set(dataset.loading_rates) == {0, 1}
    assert dataset.transition_rate_nodes[0] == "track_0:R0"
    assert dataset.loading_rate_nodes[1] == "track_0:r1"
    assert "ModelInferenceResult" in str(result)
    assert "DatasetInferenceResult" in str(dataset)


def test_cavi_progress_reports_pending_nodes(capsys):
    model = make_model()

    _ = model.fit_cavi(
        max_iterations=2,
        min_iterations=2,
        tolerance=0.0,
        compute_elbo=False,
        progress=True,
    )

    captured = capsys.readouterr()
    assert "CAVI [" in captured.out
    assert "pending" in captured.out or "all parameter nodes converged" in captured.out


def test_model_fit_alias_runs_inference():
    model = make_model()

    result = model.fit(max_iterations=1, min_iterations=1, compute_elbo=False)

    assert isinstance(result, ModelInferenceResult)
    assert result.cavi.n_iterations == 1
    assert result.cavi.elbo is None
    assert result.elbo_terms is None
    assert "track_0" in result.datasets
