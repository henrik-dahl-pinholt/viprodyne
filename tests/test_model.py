import numpy as np

from viprodyne import MS2Dataset, ModelConfig, ViprodyneModel


def make_dataset(name, offset=0.0):
    return MS2Dataset(
        name=name,
        observed=np.array([0.1 + offset, 0.8 + offset], dtype=np.float32),
        noise_std=np.float32(0.5),
        design_matrix=np.eye(2, dtype=np.float32),
        prior_load_probabilities=np.array([0.25, 0.6], dtype=np.float32),
    )


def test_model_builds_dataset_plates_with_shared_parameter_nodes():
    model = ViprodyneModel(
        datasets=(make_dataset("d0"), make_dataset("d1", offset=0.2)),
        config=ModelConfig(
            n_states=2,
            time_grid=np.array([0.0, 1.0], dtype=np.float32),
            shared_transition_rates=True,
            shared_loading_rates=True,
        ),
    )

    assert "shared:R0" in model.graph.nodes
    assert "shared:R1" in model.graph.nodes
    assert "shared:r0" in model.graph.nodes
    assert "shared:r1" in model.graph.nodes
    assert "d0:R0" not in model.graph.nodes
    assert "d1:R0" not in model.graph.nodes

    assert set(model.graph.parents_of("d0:s")) == {"d0:pi", "shared:R0", "shared:R1"}
    assert set(model.graph.parents_of("d0:tau")) == {"d0:s", "shared:r0", "shared:r1"}
    assert model.graph.children_of("d0:tau") == ("d0:I",)
    assert model.dataset_nodes["d0"]["transition_rates"] == ["shared:R0", "shared:R1"]
    assert model.dataset_nodes["d1"]["loading_rates"] == ["shared:r0", "shared:r1"]


def test_model_schedule_runs_promoter_and_pol2_nodes():
    model = ViprodyneModel(
        datasets=(make_dataset("d0"),),
        config=ModelConfig(
            n_states=2,
            time_grid=np.array([0.0, 0.5, 1.0], dtype=np.float32),
            shared_transition_rates=False,
            shared_loading_rates=False,
        ),
    )

    model.run_schedule(["d0:s", "d0:tau"])
    promoter_moments = model.graph.moments.get("d0:s")
    pol2_moments = model.graph.moments.get("d0:tau")

    assert promoter_moments["posterior"].dtype == np.float32
    assert promoter_moments["posterior"].shape == (1, 3, 2)
    assert pol2_moments["load_probabilities"].dtype == np.float32
    assert np.all(np.isfinite(pol2_moments["load_probabilities"]))
    assert "d0:s" in model.default_schedule()
    assert "d0:tau" in model.default_schedule()


def test_model_uses_transfer_pol2_mode_when_window_inputs_are_available():
    dataset = MS2Dataset(
        name="d0",
        observed=np.array([0.1, np.nan, 0.9], dtype=np.float32),
        noise_std=np.float32(0.5),
        prior_load_probabilities=np.array([0.25, 0.6, 0.4], dtype=np.float32),
        window_weights=np.array([1.0], dtype=np.float32),
        observation_starts=np.arange(3, dtype=np.int32),
    )
    model = ViprodyneModel(
        datasets=(dataset,),
        config=ModelConfig(
            n_states=2,
            time_grid=np.array([0.0, 1.0], dtype=np.float32),
            pol2_mode="transfer",
        ),
    )

    moments = model.graph.moments.get("d0:tau")
    assert moments["elbo"].dtype == np.float32
    assert np.isfinite(moments["elbo"])
