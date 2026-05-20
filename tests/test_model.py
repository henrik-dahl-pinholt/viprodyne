import jax.numpy as jnp
import numpy as np
import pytest

from viprodyne import DrivenRateMap, MS2Dataset, ModelConfig, ProximalKernel, RcNode, ViprodyneModel


def make_dataset(name, offset=0.0):
    return MS2Dataset(
        name=name,
        observed=np.array([0.1 + offset, 0.8 + offset], dtype=np.float32),
        noise_std=np.float32(0.5),
    )


def test_model_builds_dataset_plates_with_shared_parameter_nodes():
    model = ViprodyneModel(
        datasets=(make_dataset("d0"), make_dataset("d1", offset=0.2)),
        config=ModelConfig(
            n_states=2,
            time_grid=np.array([0.0, 0.5, 1.0], dtype=np.float32),
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


def test_model_can_share_selected_rates_only():
    model = ViprodyneModel(
        datasets=(make_dataset("d0"), make_dataset("d1", offset=0.2)),
        config=ModelConfig(
            n_states=2,
            time_grid=np.array([0.0, 0.5, 1.0], dtype=np.float32),
            shared_transition_rate_indices=(1,),
            shared_loading_rate_states=(0,),
        ),
    )

    assert "shared:R1" in model.graph.nodes
    assert "d0:R0" in model.graph.nodes
    assert "d1:R0" in model.graph.nodes
    assert "shared:R0" not in model.graph.nodes
    assert "d0:R1" not in model.graph.nodes
    assert "d1:R1" not in model.graph.nodes
    assert "shared:r0" in model.graph.nodes
    assert "d0:r1" in model.graph.nodes
    assert "d1:r1" in model.graph.nodes
    assert "shared:r1" not in model.graph.nodes

    assert set(model.graph.parents_of("d0:s")) == {"d0:pi", "d0:R0", "shared:R1"}
    assert set(model.graph.parents_of("d1:s")) == {"d1:pi", "d1:R0", "shared:R1"}
    assert set(model.graph.parents_of("d0:tau")) == {"d0:s", "shared:r0", "d0:r1"}
    assert set(model.graph.parents_of("d1:tau")) == {"d1:s", "shared:r0", "d1:r1"}


def test_model_rate_scopes_cover_track_dataset_and_global():
    datasets = (
        MS2Dataset(
            name="a_track0",
            rate_group="condition_a",
            observed=np.array([0.1, 0.8], dtype=np.float32),
            noise_std=np.float32(0.5),
        ),
        MS2Dataset(
            name="a_track1",
            rate_group="condition_a",
            observed=np.array([0.2, 0.7], dtype=np.float32),
            noise_std=np.float32(0.5),
        ),
        MS2Dataset(
            name="b_track0",
            rate_group="condition_b",
            observed=np.array([0.3, 0.6], dtype=np.float32),
            noise_std=np.float32(0.5),
        ),
    )
    model = ViprodyneModel(
        datasets=datasets,
        config=ModelConfig(
            n_states=2,
            time_grid=np.array([0.0, 0.5, 1.0], dtype=np.float32),
            transition_rate_scope="global",
            loading_rate_scope="dataset",
        ),
    )

    assert set(model.dataset_nodes["a_track0"]["transition_rates"]) == {"shared:R0", "shared:R1"}
    assert set(model.dataset_nodes["a_track1"]["transition_rates"]) == {"shared:R0", "shared:R1"}
    assert set(model.dataset_nodes["a_track0"]["loading_rates"]) == {
        "condition_a:r0",
        "condition_a:r1",
    }
    assert set(model.dataset_nodes["a_track1"]["loading_rates"]) == {
        "condition_a:r0",
        "condition_a:r1",
    }
    assert set(model.dataset_nodes["b_track0"]["loading_rates"]) == {
        "condition_b:r0",
        "condition_b:r1",
    }
    assert "a_track0:r0" not in model.graph.nodes
    assert "a_track1:r0" not in model.graph.nodes

    track_model = ViprodyneModel(
        datasets=datasets[:2],
        config=ModelConfig(
            n_states=2,
            time_grid=np.array([0.0, 0.5, 1.0], dtype=np.float32),
            transition_rate_scope="track",
            loading_rate_scope="track",
        ),
    )

    assert set(track_model.dataset_nodes["a_track0"]["transition_rates"]) == {
        "a_track0:R0",
        "a_track0:R1",
    }
    assert set(track_model.dataset_nodes["a_track1"]["transition_rates"]) == {
        "a_track1:R0",
        "a_track1:R1",
    }
    assert set(track_model.dataset_nodes["a_track0"]["loading_rates"]) == {
        "a_track0:r0",
        "a_track0:r1",
    }
    assert set(track_model.dataset_nodes["a_track1"]["loading_rates"]) == {
        "a_track1:r0",
        "a_track1:r1",
    }


def test_model_supports_per_rate_scope_overrides():
    datasets = (
        MS2Dataset(
            name="track0",
            rate_group="condition",
            observed=np.array([0.1, 0.8], dtype=np.float32),
            noise_std=np.float32(0.5),
        ),
        MS2Dataset(
            name="track1",
            rate_group="condition",
            observed=np.array([0.2, 0.7], dtype=np.float32),
            noise_std=np.float32(0.5),
        ),
    )
    model = ViprodyneModel(
        datasets=datasets,
        config=ModelConfig(
            n_states=2,
            time_grid=np.array([0.0, 0.5, 1.0], dtype=np.float32),
            transition_rate_scope="dataset",
            transition_rate_scopes={1: "global"},
            loading_rate_scope="dataset",
            loading_rate_scopes={0: "track"},
        ),
    )

    assert set(model.graph.parents_of("track0:s")) == {"track0:pi", "condition:R0", "shared:R1"}
    assert set(model.graph.parents_of("track1:s")) == {"track1:pi", "condition:R0", "shared:R1"}
    assert set(model.graph.parents_of("track0:tau")) == {"track0:s", "track0:r0", "condition:r1"}
    assert set(model.graph.parents_of("track1:tau")) == {"track1:s", "track1:r0", "condition:r1"}
    assert "condition:R1" not in model.graph.nodes
    assert "condition:r0" not in model.graph.nodes


def test_model_rejects_reserved_rate_prefix_labels():
    with pytest.raises(ValueError, match="reserved"):
        ViprodyneModel(
            datasets=(
                MS2Dataset(
                    name="shared",
                    observed=np.array([0.1, 0.8], dtype=np.float32),
                    noise_std=np.float32(0.5),
                ),
            ),
            config=ModelConfig(n_states=2, time_grid=np.array([0.0, 0.5, 1.0], dtype=np.float32)),
        )

    with pytest.raises(ValueError, match="must not contain ':'"):
        ViprodyneModel(
            datasets=(
                MS2Dataset(
                    name="track0",
                    rate_group="condition:0",
                    observed=np.array([0.1, 0.8], dtype=np.float32),
                    noise_std=np.float32(0.5),
                ),
            ),
            config=ModelConfig(n_states=2, time_grid=np.array([0.0, 0.5, 1.0], dtype=np.float32)),
        )


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


def test_model_uses_transfer_pol2_mode_from_kernel_config():
    dataset = MS2Dataset(
        name="d0",
        observed=np.array([0.1, np.nan, 0.9], dtype=np.float32),
        noise_std=np.float32(0.5),
    )
    model = ViprodyneModel(
        datasets=(dataset,),
        config=ModelConfig(
            n_states=2,
            time_grid=np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32),
            pol2_mode="transfer",
            t_rise=np.float32(0.5),
            t_plateau=np.float32(1.0),
            rna_intensity=np.float32(1.7),
        ),
    )

    polymerase = model.graph.nodes["d0:tau"]
    assert polymerase.mode == "transfer"
    assert polymerase.design_matrix is None
    assert polymerase.window_weights.dtype == np.float32
    assert polymerase.observation_starts.dtype == np.int32
    moments = model.graph.moments.get("d0:tau")
    assert moments["elbo"].dtype == np.float32
    assert np.isfinite(moments["elbo"])


def test_model_accepts_explicit_kernel_function():
    def rectangular_kernel(offsets):
        return jnp.where((offsets >= 0.0) & (offsets < 0.75), 2.0, 0.0)

    dataset = MS2Dataset(
        name="d0",
        observed=np.array([0.2, 0.4], dtype=np.float32),
        noise_std=np.float32(0.5),
    )
    model = ViprodyneModel(
        datasets=(dataset,),
        config=ModelConfig(
            n_states=2,
            time_grid=np.array([0.0, 0.5, 1.0], dtype=np.float32),
            ms2_kernel=rectangular_kernel,
        ),
    )

    polymerase = model.graph.nodes["d0:tau"]
    np.testing.assert_allclose(
        polymerase.window_weights,
        np.array([[2.0, 2.0], [0.0, 2.0]], dtype=np.float32),
    )


def test_model_accepts_kernel_dataclass():
    dataset = MS2Dataset(
        name="d0",
        observed=np.array([0.2, 0.4], dtype=np.float32),
        noise_std=np.float32(0.5),
    )
    model = ViprodyneModel(
        datasets=(dataset,),
        config=ModelConfig(
            n_states=2,
            time_grid=np.array([0.0, 0.5, 1.0], dtype=np.float32),
            ms2_kernel=ProximalKernel(
                t_rise=np.float32(0.25),
                t_plateau=np.float32(0.5),
                rna_intensity=np.float32(3.0),
            ),
        ),
    )

    polymerase = model.graph.nodes["d0:tau"]
    assert polymerase.mode == "transfer"
    assert polymerase.window_weights.dtype == np.float32


def test_model_derives_pol2_prior_from_promoter_and_loading_rates():
    model = ViprodyneModel(
        datasets=(make_dataset("d0"),),
        config=ModelConfig(
            n_states=2,
            time_grid=np.array([0.0, 0.5, 1.0], dtype=np.float32),
        ),
    )
    model.graph.nodes["d0:r0"].pin(np.float32(0.5))
    model.graph.nodes["d0:r1"].pin(np.float32(1.0))
    model.graph.moments.publish("d0:r0", model.graph.nodes["d0:r0"].moments())
    model.graph.moments.publish("d0:r1", model.graph.nodes["d0:r1"].moments())

    model.run_schedule(["d0:s", "d0:tau"])

    promoter_moments = model.graph.moments.get("d0:s")
    state_probabilities = promoter_moments["interval_state_probabilities"][0]
    dt = promoter_moments["interval_durations"]
    loading_rates = np.array(
        [
            model.graph.moments.get("d0:r0")["mean"],
            model.graph.moments.get("d0:r1")["mean"],
        ],
        dtype=np.float32,
    )
    expected_prior = np.sum(
        state_probabilities * (1.0 - np.exp(-dt[:, None] * loading_rates[None, :])),
        axis=-1,
    )
    polymerase = model.graph.nodes["d0:tau"]

    np.testing.assert_allclose(polymerase.prior_probabilities, expected_prior, rtol=2e-6)


def test_model_accepts_dataset_specific_time_grids():
    d0 = MS2Dataset(
        name="d0",
        observed=np.array([0.1, 0.8], dtype=np.float32),
        noise_std=np.float32(0.5),
        time_grid=np.array([0.0, 0.5, 1.0], dtype=np.float32),
    )
    d1 = MS2Dataset(
        name="d1",
        observed=np.array([0.2, 0.5, 0.9], dtype=np.float32),
        noise_std=np.float32(0.5),
        time_grid=np.array([0.0, 0.25, 0.75, 1.5], dtype=np.float32),
    )
    model = ViprodyneModel(
        datasets=(d0, d1),
        config=ModelConfig(n_states=2),
    )

    model.run_schedule(["d0:s", "d1:s"])

    assert model.graph.nodes["d0:s"].time_grid.shape == (3,)
    assert model.graph.nodes["d1:s"].time_grid.shape == (4,)
    assert model.graph.moments.get("d0:s")["posterior"].shape == (1, 3, 2)
    assert model.graph.moments.get("d1:s")["posterior"].shape == (1, 4, 2)


def test_model_builds_driven_transition_with_dataset_contact_drive():
    dataset = MS2Dataset(
        name="d0",
        observed=np.array([0.1, 0.9], dtype=np.float32),
        noise_std=np.float32(0.5),
        contact_probability=np.array([0.25, 0.75], dtype=np.float32),
    )
    model = ViprodyneModel(
        datasets=(dataset,),
        config=ModelConfig(
            n_states=2,
            time_grid=np.array([0.0, 0.5, 1.0], dtype=np.float32),
            driven_transition_indices=(1,),
            driven_rate_initial=np.float32(0.8),
            driven_rate_bounds=(1e-4, 10.0),
        ),
    )

    assert isinstance(model.graph.nodes["d0:R1"], DrivenRateMap)
    assert isinstance(model.graph.nodes["d0:rc"], RcNode)
    assert model.dataset_nodes["d0"]["contact_drive"] == "d0:rc"
    promoter = model.graph.nodes["d0:s"]
    assert promoter.rate_edges[1].drive_node == "d0:rc"

    model.run_schedule(["d0:s"])
    moments = model.graph.moments.get("d0:s")

    assert "contact_survival_stats_by_rate" in moments
    assert "d0:R1" in moments["contact_survival_stats_by_rate"]
    np.testing.assert_allclose(
        promoter.tilted_generator[:, 1, 0],
        np.array([0.2, 0.6], dtype=np.float32),
        rtol=2e-6,
    )
    np.testing.assert_allclose(
        np.sum(promoter.tilted_generator, axis=-2),
        np.zeros((2, 2), dtype=np.float32),
        atol=2e-7,
    )
    assert "d0:rc" in model.default_schedule()


def test_model_requires_contact_probability_for_driven_transitions():
    with pytest.raises(ValueError, match="contact_probability"):
        ViprodyneModel(
            datasets=(make_dataset("d0"),),
            config=ModelConfig(
                n_states=2,
                time_grid=np.array([0.0, 0.5, 1.0], dtype=np.float32),
                driven_transition_indices=(1,),
            ),
        )
