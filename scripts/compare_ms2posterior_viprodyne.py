#!/usr/bin/env python
"""Parity investigation between MS2Posterior and viprodyne.

This script is intentionally not a pytest test: it imports a local checkout of
MS2Posterior and compares implementation details that are useful when debugging
the model migration.  The default toy problem uses a proximal kernel whose
support is one grid interval, so the Pol2 transfer layer has an analytically
simple, non-interacting structure while still exercising the real code paths.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")

from viprodyne import MS2Dataset, ModelConfig, ProximalKernel, ViprodyneModel
from viprodyne.core.bernoulli_transfer_pol2 import bernoulli_transfer_posterior


FLOAT = np.float32
DEFAULT_MS2POSTERIOR_PATH = Path("/net/levsha/share/hdp/git_reps/MS2Posterior")


@dataclass(frozen=True)
class ToyCase:
    dt: np.float32
    fine_grid: np.ndarray
    time_grid: np.ndarray
    sampling_times: np.ndarray
    observed: np.ndarray
    noise: np.ndarray
    t_rise: np.float32
    t_plateau: np.float32
    rna_intensity: np.float32
    initial_probabilities: np.ndarray
    fixed_state_posterior_old: np.ndarray
    fixed_pol2_rate_density: np.ndarray
    transition_shape: np.ndarray
    transition_rate: np.ndarray
    loading_shape: np.ndarray
    loading_rate: np.ndarray
    prior_shape: np.float32
    prior_rate: np.float32


def make_toy_case() -> ToyCase:
    dt = np.asarray(0.5, dtype=FLOAT)
    n_traces = 2
    n_loadings = 8
    n_states = 2
    fine_grid = (np.arange(n_loadings, dtype=FLOAT) * dt).astype(FLOAT)
    time_grid = (np.arange(n_loadings + 1, dtype=FLOAT) * dt).astype(FLOAT)
    sampling_times = ((np.arange(n_loadings, dtype=FLOAT) + np.float32(1.0)) * dt).astype(
        FLOAT
    )
    t_rise = np.asarray(dt, dtype=FLOAT)
    t_plateau = np.asarray(0.0, dtype=FLOAT)
    rna_intensity = np.asarray(1.3, dtype=FLOAT)
    true_loads = np.asarray(
        [
            [0, 1, 0, 1, 0, 0, 1, 0],
            [1, 0, 0, 1, 1, 0, 0, 1],
        ],
        dtype=FLOAT,
    )
    # With support equal to one interval, observation t_i reads loading i.
    observed = true_loads * rna_intensity
    observed += np.asarray(
        [
            [0.05, -0.04, 0.03, 0.02, -0.01, 0.04, -0.03, 0.01],
            [-0.02, 0.04, 0.01, -0.05, 0.02, -0.03, 0.03, -0.01],
        ],
        dtype=FLOAT,
    )
    noise = np.asarray([0.45, 0.5], dtype=FLOAT)

    grid_index = np.arange(n_loadings + 1, dtype=FLOAT)
    fixed_state_posterior = np.zeros((n_loadings + 1, n_traces, n_states), dtype=FLOAT)
    for trace in range(n_traces):
        logits = -0.4 + 0.35 * trace + 0.25 * np.sin(0.8 * grid_index)
        p_on = 1.0 / (1.0 + np.exp(-logits))
        fixed_state_posterior[:, trace, 1] = p_on.astype(FLOAT)
        fixed_state_posterior[:, trace, 0] = (1.0 - p_on).astype(FLOAT)
    fixed_pol2_rate_density = np.asarray(
        [
            [0.03, 0.82, 0.08, 0.76, 0.12, 0.05, 0.68, 0.09],
            [0.79, 0.07, 0.06, 0.73, 0.71, 0.08, 0.10, 0.69],
        ],
        dtype=FLOAT,
    )

    return ToyCase(
        dt=dt,
        fine_grid=fine_grid,
        time_grid=time_grid,
        sampling_times=sampling_times,
        observed=observed.astype(FLOAT),
        noise=noise,
        t_rise=t_rise,
        t_plateau=t_plateau,
        rna_intensity=rna_intensity,
        initial_probabilities=np.asarray([0.62, 0.38], dtype=FLOAT),
        fixed_state_posterior_old=fixed_state_posterior,
        fixed_pol2_rate_density=fixed_pol2_rate_density,
        transition_shape=np.asarray([2.4, 3.1], dtype=FLOAT),
        transition_rate=np.asarray([5.0, 4.2], dtype=FLOAT),
        loading_shape=np.asarray([3.2, 4.6], dtype=FLOAT),
        loading_rate=np.asarray([7.0, 5.5], dtype=FLOAT),
        prior_shape=np.asarray(1.2, dtype=FLOAT),
        prior_rate=np.asarray(2.5, dtype=FLOAT),
    )


def import_ms2posterior(path: Path):
    sys.path.insert(0, str(path))
    import Bernoulli_Transfer_Pol2 as old_transfer  # noqa: PLC0415
    import Variational_State_Finder as old_vsf  # noqa: PLC0415

    return old_vsf, old_transfer


def build_old_model(old_vsf: Any, case: ToyCase):
    model = old_vsf.MS2Posterior(
        fine_grid=jnp.asarray(case.fine_grid),
        nstates=2,
        sampling_times=jnp.asarray(case.sampling_times),
        arr_dat=np.asarray(case.observed, dtype=FLOAT),
        T_rise=float(case.t_rise),
        T_plateau=float(case.t_plateau),
        MS2_intensity=float(case.rna_intensity),
        noise=np.asarray(case.noise, dtype=FLOAT),
        per_track_rates=False,
        argdict={
            "Loading Rates": {
                "prior_shape": np.full(2, case.prior_shape, dtype=FLOAT),
                "prior_rate": np.full(2, case.prior_rate, dtype=FLOAT),
            },
            "Transition Rates": {
                "prior_shape": np.full(2, case.prior_shape, dtype=FLOAT),
                "prior_rate": np.full(2, case.prior_rate, dtype=FLOAT),
            },
            "Initial State Probabilities": {
                "prior_concentration": np.broadcast_to(
                    case.initial_probabilities,
                    (case.observed.shape[0], 2),
                ).astype(FLOAT),
            },
            "Polymerase Loadings": {
                "posterior_method": "transfer",
                "transfer_batch_size": case.observed.shape[0],
                "transfer_use_x64": False,
            },
        },
    )
    set_old_gamma(model, "Transition Rates", case.transition_shape, case.transition_rate)
    set_old_gamma(model, "Loading Rates", case.loading_shape, case.loading_rate)
    model.M.publish(
        "Initial State Probabilities",
        {
            "Initial State Probabilities <log pi>": np.broadcast_to(
                np.log(case.initial_probabilities),
                (case.observed.shape[0], 2),
            ).astype(FLOAT)
        },
    )
    return model


def build_viprodyne_model(case: ToyCase) -> ViprodyneModel:
    dataset = MS2Dataset(
        observed=case.observed,
        noise_std=case.noise,
        time_grid=case.time_grid,
        sampling_times=case.sampling_times,
    )
    config = ModelConfig(
        n_states=2,
        pol2_mode="transfer",
        ms2_kernel=ProximalKernel(
            t_rise=case.t_rise,
            t_plateau=case.t_plateau,
            rna_intensity=case.rna_intensity,
        ),
        transition_prior_shape=case.prior_shape,
        transition_prior_rate=case.prior_rate,
        loading_prior_shape=case.prior_shape,
        loading_prior_rate=case.prior_rate,
    )
    model = ViprodyneModel((dataset,), config)
    sync_viprodyne_parameters(model, case)
    return model


def set_old_gamma(model: Any, name: str, shape: np.ndarray, rate: np.ndarray) -> None:
    node = model.G[name]
    node.shape = np.asarray(shape, dtype=FLOAT)
    node.rate = np.asarray(rate, dtype=FLOAT)
    model.M.publish(name, node.moments())


def sync_viprodyne_parameters(model: ViprodyneModel, case: ToyCase) -> None:
    nodes = model.dataset_nodes["dataset_0"]
    model.graph.nodes[str(nodes["initial"])].pin(case.initial_probabilities)
    model.graph.moments.publish(
        str(nodes["initial"]),
        model.graph.nodes[str(nodes["initial"])].moments(),
    )
    for index, node_name in enumerate(nodes["transition_rates"]):
        node = model.graph.nodes[str(node_name)]
        node.shape = np.asarray(case.transition_shape[index], dtype=FLOAT)
        node.rate = np.asarray(case.transition_rate[index], dtype=FLOAT)
        model.graph.moments.publish(str(node_name), node.moments())
    for index, node_name in enumerate(nodes["loading_rates"]):
        node = model.graph.nodes[str(node_name)]
        node.shape = np.asarray(case.loading_shape[index], dtype=FLOAT)
        node.rate = np.asarray(case.loading_rate[index], dtype=FLOAT)
        model.graph.moments.publish(str(node_name), node.moments())


def publish_fixed_state_messages(old_model: Any, vip_model: ViprodyneModel, case: ToyCase) -> None:
    old_state = np.asarray(case.fixed_state_posterior_old, dtype=FLOAT)
    old_model.M.publish(
        "Promoter State",
        {
            "State_posterior": old_state,
            "masked_posterior": old_state,
            "masked_joint": np.zeros(
                (old_state.shape[0] - 1, old_state.shape[1], 2, 2),
                dtype=FLOAT,
            ),
        },
    )
    nodes = vip_model.dataset_nodes["dataset_0"]
    interval_probs = np.swapaxes(old_state[1:], 0, 1).astype(FLOAT)
    vip_model.graph.moments.publish(
        str(nodes["promoter"]),
        {
            "posterior": np.swapaxes(old_state, 0, 1).astype(FLOAT),
            "initial_state_counts": interval_probs[:, 0],
            "interval_state_probabilities": interval_probs,
            "interval_durations": np.diff(case.time_grid).astype(FLOAT),
            "expected_occupancy": interval_probs
            * np.diff(case.time_grid)[None, :, None].astype(FLOAT),
            "expected_jumps": np.zeros(
                (case.observed.shape[0], case.fine_grid.size, 2, 2),
                dtype=FLOAT,
            ),
            "transition_counts": np.zeros((case.observed.shape[0], 2, 2), dtype=FLOAT),
            "transition_exposure": np.sum(
                interval_probs * np.diff(case.time_grid)[None, :, None],
                axis=1,
                dtype=FLOAT,
            ),
        },
    )


def publish_fixed_pol2_messages(old_model: Any, vip_model: ViprodyneModel, case: ToyCase) -> None:
    old_model.M.publish(
        "Polymerase Loadings",
        {"Pol2_posterior": np.asarray(case.fixed_pol2_rate_density, dtype=FLOAT)},
    )
    nodes = vip_model.dataset_nodes["dataset_0"]
    # To exactly emulate the legacy promoter tilt, viprodyne must receive
    # interval counts equal to the old rate-density-like message times dt.
    vip_model.graph.moments.publish(
        str(nodes["polymerase"]),
        {
            "expected_loading_counts": (
                np.asarray(case.fixed_pol2_rate_density, dtype=FLOAT) * case.dt
            ).astype(FLOAT),
            "loading_mask": np.ones_like(case.fixed_pol2_rate_density, dtype=bool),
        },
    )


def run_pol2_comparison(old_model: Any, vip_model: ViprodyneModel, case: ToyCase) -> dict[str, Any]:
    publish_fixed_state_messages(old_model, vip_model, case)
    old_model.G["Polymerase Loadings"].update(
        {
            "parents": {
                **old_model.M.get("Loading Rates"),
                **old_model.M.get("Promoter State"),
            },
            "children": old_model.M.get("MS2 Data"),
            "co_parents": {},
        },
        rho=1.0,
    )
    old_messages = old_model.G["Polymerase Loadings"].moments()
    old_model.M.publish("Polymerase Loadings", old_messages)
    old_transfer_result = old_model.G["Polymerase Loadings"]._run_transfer_matrix(
        with_posterior=True
    )

    nodes = vip_model.dataset_nodes["dataset_0"]
    vip_model.graph.run_schedule([str(nodes["polymerase"])], rho=1.0)
    vip_messages = vip_model.graph.moments.get(str(nodes["polymerase"]))
    vip_poly = vip_model.graph.nodes[str(nodes["polymerase"])]

    legacy_prior = np.asarray(old_transfer_result["prior_load_prob"], dtype=FLOAT)
    saved_prior = np.asarray(vip_poly.prior_probabilities, dtype=FLOAT).copy()
    vip_poly.prior_probabilities = legacy_prior
    vip_poly.update_from_current_inputs()
    forced_legacy = vip_poly.moments()
    vip_poly.prior_probabilities = saved_prior
    vip_poly.update_from_current_inputs()
    vip_model.graph.moments.publish(str(nodes["polymerase"]), vip_poly.moments())

    return {
        "old_prior": legacy_prior,
        "vip_native_prior": np.asarray(vip_poly.prior_probabilities, dtype=FLOAT),
        "old_posterior": np.asarray(old_messages["Pol2_posterior"], dtype=FLOAT),
        "vip_native_posterior": np.asarray(vip_messages["load_probabilities"], dtype=FLOAT),
        "vip_forced_legacy_posterior": np.asarray(
            forced_legacy["load_probabilities"],
            dtype=FLOAT,
        ),
        "old_logz": np.asarray(old_transfer_result["loglik_sum"], dtype=FLOAT),
        "vip_native_logz": np.asarray(vip_messages["elbo"], dtype=FLOAT),
        "vip_forced_legacy_logz": np.asarray(forced_legacy["elbo"], dtype=FLOAT),
        "old_predicted": np.asarray(old_messages["Predicted MS2"], dtype=FLOAT),
        "vip_native_predicted": np.asarray(vip_messages["predicted_signal"], dtype=FLOAT),
        "vip_forced_legacy_predicted": np.asarray(
            forced_legacy["predicted_signal"],
            dtype=FLOAT,
        ),
    }


def run_promoter_comparison(old_model: Any, vip_model: ViprodyneModel, case: ToyCase) -> dict[str, Any]:
    publish_fixed_pol2_messages(old_model, vip_model, case)
    old_model.G["Promoter State"].update(
        {
            "parents": {
                **old_model.M.get("Transition Rates"),
                **old_model.M.get("Initial State Probabilities"),
            },
            "children": old_model.M.get("Polymerase Loadings"),
            "co_parents": old_model.M.get("Loading Rates"),
        },
        rho=1.0,
    )
    old_messages = old_model.G["Promoter State"].moments()
    old_model.M.publish("Promoter State", old_messages)

    nodes = vip_model.dataset_nodes["dataset_0"]
    vip_model.graph.run_schedule([str(nodes["promoter"])], rho=1.0)
    vip_messages = vip_model.graph.moments.get(str(nodes["promoter"]))

    old_elbo = old_model.G["Promoter State"].e_log_p_local(
        {
            **old_model.M.get("Transition Rates"),
            **old_model.M.get("Initial State Probabilities"),
        }
    ) + old_model.G["Promoter State"].entropy()

    return {
        "old_state_posterior": np.swapaxes(
            np.asarray(old_messages["State_posterior"], dtype=FLOAT),
            0,
            1,
        ),
        "vip_state_posterior": np.asarray(vip_messages["posterior"], dtype=FLOAT),
        "old_jump_counts": old_transition_sufficient_statistics(old_messages, case)[0],
        "vip_jump_counts": np.sum(
            np.asarray(vip_messages["transition_counts"], dtype=FLOAT),
            axis=0,
            dtype=FLOAT,
        ),
        "old_exposure": old_transition_sufficient_statistics(old_messages, case)[1],
        "vip_exposure": np.sum(
            np.asarray(vip_messages["transition_exposure"], dtype=FLOAT),
            axis=0,
            dtype=FLOAT,
        ),
        "old_elbo": np.asarray(old_elbo, dtype=FLOAT),
        "vip_elbo": np.asarray(vip_messages["elbo"], dtype=FLOAT),
    }


def old_transition_sufficient_statistics(
    old_promoter_messages: Mapping[str, Any],
    case: ToyCase,
    *,
    include_initial_exposure: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    joint = np.asarray(old_promoter_messages["masked_joint"], dtype=FLOAT)
    posterior = np.asarray(old_promoter_messages["masked_posterior"], dtype=FLOAT)
    counts = np.sum(joint * case.dt, axis=(0, 1), dtype=FLOAT)
    exposure_source = posterior if include_initial_exposure else posterior[1:]
    exposure = np.sum(exposure_source * case.dt, axis=(0, 1), dtype=FLOAT)
    return counts, exposure


def run_rate_update_comparison(
    old_model: Any,
    vip_model: ViprodyneModel,
    case: ToyCase,
    promoter_comparison: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    nodes = vip_model.dataset_nodes["dataset_0"]

    old_loading = clone_old_gamma_state(old_model, "Loading Rates")
    old_model.G["Loading Rates"].update(
        {
            "parents": {},
            "children": {"Pol2_posterior": case.fixed_pol2_rate_density},
            "co_parents": {
                "masked_posterior": np.asarray(case.fixed_state_posterior_old, dtype=FLOAT)
            },
        },
        rho=1.0,
    )
    old_loading_shape = np.asarray(old_model.G["Loading Rates"].shape, dtype=FLOAT)
    old_loading_rate = np.asarray(old_model.G["Loading Rates"].rate, dtype=FLOAT)
    restore_old_gamma_state(old_model, "Loading Rates", old_loading)

    interval_probs = np.swapaxes(case.fixed_state_posterior_old[1:], 0, 1).astype(FLOAT)
    old_loading_counts = np.sum(
        case.fixed_pol2_rate_density[:, :, None] * interval_probs * case.dt,
        axis=(0, 1),
        dtype=FLOAT,
    )
    old_loading_exposure = np.sum(interval_probs * case.dt, axis=(0, 1), dtype=FLOAT)
    vip_model.graph.moments.publish(
        str(nodes["polymerase"]),
        {
            "loading_counts_by_rate": {
                str(nodes["loading_rates"][0]): old_loading_counts[0],
                str(nodes["loading_rates"][1]): old_loading_counts[1],
            },
            "loading_exposure_by_rate": {
                str(nodes["loading_rates"][0]): old_loading_exposure[0],
                str(nodes["loading_rates"][1]): old_loading_exposure[1],
            },
        },
    )
    reset_vip_gamma_nodes(vip_model, nodes["loading_rates"], case.loading_shape, case.loading_rate)
    for node_name in nodes["loading_rates"]:
        vip_model.graph.run_schedule([str(node_name)], rho=1.0)
    vip_loading_shape = np.asarray(
        [vip_model.graph.nodes[str(name)].shape for name in nodes["loading_rates"]],
        dtype=FLOAT,
    )
    vip_loading_rate = np.asarray(
        [vip_model.graph.nodes[str(name)].rate for name in nodes["loading_rates"]],
        dtype=FLOAT,
    )

    old_transition = clone_old_gamma_state(old_model, "Transition Rates")
    old_model.G["Transition Rates"].update(
        {
            "parents": {},
            "children": {
                "masked_posterior": old_model.M.get("Promoter State")["masked_posterior"],
                "masked_joint": old_model.M.get("Promoter State")["masked_joint"],
            },
            "co_parents": {},
        },
        rho=1.0,
    )
    old_transition_shape = np.asarray(old_model.G["Transition Rates"].shape, dtype=FLOAT)
    old_transition_rate = np.asarray(old_model.G["Transition Rates"].rate, dtype=FLOAT)
    restore_old_gamma_state(old_model, "Transition Rates", old_transition)

    old_counts, old_exposure = old_transition_sufficient_statistics(
        old_model.M.get("Promoter State"),
        case,
        include_initial_exposure=True,
    )
    old_counts = np.asarray(old_counts, dtype=FLOAT)
    old_exposure = np.asarray(old_exposure, dtype=FLOAT)
    vip_model.graph.moments.publish(
        str(nodes["promoter"]),
        {
            "transition_counts": old_counts,
            "transition_exposure": old_exposure,
        },
    )
    reset_vip_gamma_nodes(
        vip_model,
        nodes["transition_rates"],
        case.transition_shape,
        case.transition_rate,
    )
    for node_name in nodes["transition_rates"]:
        vip_model.graph.run_schedule([str(node_name)], rho=1.0)
    vip_transition_shape = np.asarray(
        [vip_model.graph.nodes[str(name)].shape for name in nodes["transition_rates"]],
        dtype=FLOAT,
    )
    vip_transition_rate = np.asarray(
        [vip_model.graph.nodes[str(name)].rate for name in nodes["transition_rates"]],
        dtype=FLOAT,
    )

    # Native viprodyne loading counts use Bernoulli expected counts, not the
    # legacy dt-scaled rate-density message.  This isolates that difference.
    correct_loading_counts = np.sum(
        case.fixed_pol2_rate_density[:, :, None] * interval_probs,
        axis=(0, 1),
        dtype=FLOAT,
    )

    return {
        "old_loading_shape": old_loading_shape,
        "vip_forced_loading_shape": vip_loading_shape,
        "old_loading_rate": old_loading_rate,
        "vip_forced_loading_rate": vip_loading_rate,
        "legacy_loading_counts": old_loading_counts,
        "bernoulli_loading_counts": correct_loading_counts,
        "old_transition_shape": old_transition_shape,
        "vip_forced_transition_shape": vip_transition_shape,
        "old_transition_rate": old_transition_rate,
        "vip_forced_transition_rate": vip_transition_rate,
    }


def clone_old_gamma_state(model: Any, node_name: str) -> tuple[np.ndarray, np.ndarray]:
    node = model.G[node_name]
    return np.asarray(node.shape, dtype=FLOAT).copy(), np.asarray(node.rate, dtype=FLOAT).copy()


def restore_old_gamma_state(model: Any, node_name: str, state: tuple[np.ndarray, np.ndarray]) -> None:
    shape, rate = state
    set_old_gamma(model, node_name, shape, rate)


def reset_vip_gamma_nodes(
    model: ViprodyneModel,
    node_names: list[str],
    shapes: np.ndarray,
    rates: np.ndarray,
) -> None:
    for index, node_name in enumerate(node_names):
        node = model.graph.nodes[str(node_name)]
        node.shape = np.asarray(shapes[index], dtype=FLOAT)
        node.rate = np.asarray(rates[index], dtype=FLOAT)
        model.graph.moments.publish(str(node_name), node.moments())


def run_transfer_core_comparison(old_transfer: Any, case: ToyCase) -> dict[str, Any]:
    interval_probs = np.swapaxes(case.fixed_state_posterior_old[1:], 0, 1).astype(FLOAT)
    loading_expected_log = digamma32(case.loading_shape) - np.log(case.loading_rate)
    old_rate_density = np.exp(
        np.sum(interval_probs * loading_expected_log[None, None, :], axis=-1)
    ).astype(FLOAT)
    old_result = old_transfer.exact_bernoulli_transfer_posterior(
        case.observed,
        old_rate_density,
        case.noise,
        case.sampling_times,
        case.fine_grid,
        float(case.t_rise),
        float(case.t_plateau),
        float(case.rna_intensity),
        batch_size=case.observed.shape[0],
        use_x64=False,
    )
    prepared = old_transfer._prepare_inputs(
        case.observed,
        old_rate_density,
        case.noise,
        case.sampling_times,
        case.fine_grid,
        float(case.t_rise),
        float(case.t_plateau),
        float(case.rna_intensity),
    )
    vip_logz = []
    vip_posterior = []
    vip_predicted = []
    for trace in range(case.observed.shape[0]):
        logz, posterior, predicted, *_ = bernoulli_transfer_posterior(
            jnp.asarray(case.observed[trace], dtype=jnp.float32),
            jnp.asarray(prepared["prior_probs"][trace], dtype=jnp.float32),
            jnp.asarray(prepared["weights"], dtype=jnp.float32),
            jnp.asarray(prepared["starts"], dtype=jnp.int32),
            jnp.full(case.observed.shape[1], case.noise[trace], dtype=jnp.float32),
            jnp.isfinite(jnp.asarray(case.observed[trace])),
        )
        grid_slice = slice(
            prepared["pad_left"],
            prepared["pad_left"] + prepared["original_grid_size"],
        )
        vip_logz.append(np.asarray(logz, dtype=FLOAT))
        vip_posterior.append(np.asarray(posterior, dtype=FLOAT)[grid_slice])
        vip_predicted.append(np.asarray(predicted, dtype=FLOAT))
    return {
        "old_logz": np.asarray(old_result["loglik_sum"], dtype=FLOAT),
        "vip_logz": np.asarray(np.sum(vip_logz), dtype=FLOAT),
        "old_posterior": np.asarray(old_result["posterior_load_prob"], dtype=FLOAT),
        "vip_posterior": np.asarray(vip_posterior, dtype=FLOAT),
        "old_predicted": np.asarray(old_result["predicted_ms2"], dtype=FLOAT),
        "vip_predicted": np.asarray(vip_predicted, dtype=FLOAT),
        "old_weights": np.asarray(old_result["weights"], dtype=FLOAT),
        "old_starts": np.asarray(old_result["starts"], dtype=np.int32),
    }


def run_elbo_comparison(old_model: Any, vip_model: ViprodyneModel) -> dict[str, Any]:
    old_total, old_terms = old_model.ELBO()
    vip_terms = vip_model.compute_elbo_terms()
    nodes = vip_model.dataset_nodes["dataset_0"]
    grouped_vip = {
        "Loading Rates": sum(float(vip_terms[str(name)]) for name in nodes["loading_rates"]),
        "Transition Rates": sum(float(vip_terms[str(name)]) for name in nodes["transition_rates"]),
        "Initial State Probabilities": float(vip_terms[str(nodes["initial"])]),
        "Promoter State": float(vip_terms[str(nodes["promoter"])]),
        "Polymerase Loadings": float(vip_terms[str(nodes["polymerase"])]),
        "MS2 Data": float(vip_terms[str(nodes["observed"])]),
    }
    return {
        "old_total": float(old_total),
        "vip_total": float(sum(grouped_vip.values())),
        "old_terms": {name: float(value) for name, value in old_terms.items()},
        "vip_terms": grouped_vip,
    }


def digamma32(value: np.ndarray) -> np.ndarray:
    return np.asarray(jax.scipy.special.digamma(jnp.asarray(value, dtype=jnp.float32)))


def max_abs(left: np.ndarray | float, right: np.ndarray | float) -> float:
    left_arr = np.asarray(left, dtype=np.float64)
    right_arr = np.asarray(right, dtype=np.float64)
    if left_arr.shape != right_arr.shape:
        return float("nan")
    return float(np.nanmax(np.abs(left_arr - right_arr)))


def print_metric(label: str, left: np.ndarray | float, right: np.ndarray | float) -> None:
    print(f"{label:<58} max_abs={max_abs(left, right):.6g}")


def print_report(
    transfer_core: Mapping[str, Any],
    pol2: Mapping[str, Any],
    promoter: Mapping[str, Any],
    rates: Mapping[str, Any],
    elbo: Mapping[str, Any],
) -> None:
    print("# MS2Posterior / viprodyne parity investigation")
    print()
    print("## Direct Pol2 transfer core with identical prior probabilities/windows")
    print_metric("log Z", transfer_core["old_logz"], transfer_core["vip_logz"])
    print_metric("posterior loading probabilities", transfer_core["old_posterior"], transfer_core["vip_posterior"])
    print_metric("predicted MS2", transfer_core["old_predicted"], transfer_core["vip_predicted"])
    print(f"old transfer weights: {transfer_core['old_weights']}")
    print(f"old transfer starts: {transfer_core['old_starts']}")
    print()
    print("## Pol2 node update")
    print_metric("old prior vs vip native prior", pol2["old_prior"], pol2["vip_native_prior"])
    print_metric(
        "old posterior vs vip native posterior",
        pol2["old_posterior"],
        pol2["vip_native_posterior"],
    )
    print_metric(
        "old posterior vs vip forced legacy-prior posterior",
        pol2["old_posterior"],
        pol2["vip_forced_legacy_posterior"],
    )
    print_metric("old log Z vs vip native log Z", pol2["old_logz"], pol2["vip_native_logz"])
    print_metric(
        "old log Z vs vip forced legacy-prior log Z",
        pol2["old_logz"],
        pol2["vip_forced_legacy_logz"],
    )
    print()
    print("## Promoter update with legacy-scaled Pol2 message")
    print_metric(
        "state posterior",
        promoter["old_state_posterior"],
        promoter["vip_state_posterior"],
    )
    print_metric("transition counts", promoter["old_jump_counts"], promoter["vip_jump_counts"])
    print_metric("state exposure", promoter["old_exposure"], promoter["vip_exposure"])
    print_metric("promoter ELBO contribution", promoter["old_elbo"], promoter["vip_elbo"])
    print()
    print("## Rate update from identical sufficient statistics")
    print_metric(
        "loading shape",
        rates["old_loading_shape"],
        rates["vip_forced_loading_shape"],
    )
    print_metric("loading rate", rates["old_loading_rate"], rates["vip_forced_loading_rate"])
    print_metric(
        "transition shape",
        rates["old_transition_shape"],
        rates["vip_forced_transition_shape"],
    )
    print_metric(
        "transition rate",
        rates["old_transition_rate"],
        rates["vip_forced_transition_rate"],
    )
    print_metric(
        "legacy dt-scaled loading counts vs Bernoulli counts",
        rates["legacy_loading_counts"],
        rates["bernoulli_loading_counts"],
    )
    print()
    print("## ELBO terms after the forced promoter/Pol2 state")
    for key in sorted(elbo["old_terms"]):
        if key in elbo["vip_terms"]:
            print(
                f"{key:<34} old={elbo['old_terms'][key]: .8g} "
                f"vip={elbo['vip_terms'][key]: .8g} "
                f"diff={elbo['vip_terms'][key] - elbo['old_terms'][key]: .8g}"
            )
    print(
        f"{'total':<34} old={elbo['old_total']: .8g} "
        f"vip={elbo['vip_total']: .8g} diff={elbo['vip_total'] - elbo['old_total']: .8g}"
    )
    print()
    print("## Interpretation")
    print(
        "- The direct transfer kernels should match when they receive identical "
        "Bernoulli prior probabilities and windows."
    )
    print(
        "- Native Pol2 node updates intentionally differ when rates are Gamma-distributed: "
        "MS2Posterior plugs exp(E[log k]) into 1-exp(-k dt), while viprodyne uses "
        "E[log(1-exp(-k dt))] and E[-k dt] in the Bernoulli prior log odds."
    )
    print(
        "- MS2Posterior treats transfer posterior loading probabilities like a rate "
        "density in downstream updates, so loading counts are multiplied by dt. "
        "Viprodyne treats them as Bernoulli expected counts."
    )
    print(
        "- Promoter posteriors can be made close by passing viprodyne a legacy-scaled "
        "Pol2 count message. Remaining transition-statistic differences reflect "
        "MS2Posterior's grid-density/Riemann sufficient statistics versus viprodyne's "
        "interval-integrated CTMC sufficient statistics."
    )
    print(
        "- Initial-state ELBO terms differ unless the viprodyne initial node is pinned; "
        "MS2Posterior hard-codes that node's prior and entropy terms to zero."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ms2posterior-path",
        type=Path,
        default=DEFAULT_MS2POSTERIOR_PATH,
        help="Path containing Variational_State_Finder.py.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    old_vsf, old_transfer = import_ms2posterior(args.ms2posterior_path)
    case = make_toy_case()
    old_model = build_old_model(old_vsf, case)
    vip_model = build_viprodyne_model(case)

    transfer_core = run_transfer_core_comparison(old_transfer, case)
    pol2 = run_pol2_comparison(old_model, vip_model, case)

    # Restore the synchronized parameter state because the Pol2 comparison
    # updates the viprodyne Pol2 node natively.
    sync_viprodyne_parameters(vip_model, case)
    promoter = run_promoter_comparison(old_model, vip_model, case)
    rates = run_rate_update_comparison(old_model, vip_model, case, promoter)

    # Put the Pol2 node back on the old legacy prior for an apples-to-apples
    # local ELBO comparison with the old code's Pol2 term.  Recompute that prior
    # from the current old promoter posterior rather than reusing the fixed-state
    # Pol2 comparison prior.
    sync_viprodyne_parameters(vip_model, case)
    nodes = vip_model.dataset_nodes["dataset_0"]
    old_model.G["Polymerase Loadings"].update(
        {
            "parents": {
                **old_model.M.get("Loading Rates"),
                **old_model.M.get("Promoter State"),
            },
            "children": old_model.M.get("MS2 Data"),
            "co_parents": {},
        },
        rho=1.0,
    )
    old_current_transfer = old_model.G["Polymerase Loadings"]._run_transfer_matrix(
        with_posterior=True
    )
    old_model.M.publish("Polymerase Loadings", old_model.G["Polymerase Loadings"].moments())
    vip_poly = vip_model.graph.nodes[str(nodes["polymerase"])]
    vip_poly.prior_probabilities = np.asarray(
        old_current_transfer["prior_load_prob"],
        dtype=FLOAT,
    )
    vip_poly.update_from_current_inputs()
    vip_model.graph.moments.publish(str(nodes["polymerase"]), vip_poly.moments())
    elbo_terms = run_elbo_comparison(old_model, vip_model)

    print_report(transfer_core, pol2, promoter, rates, elbo_terms)


if __name__ == "__main__":
    main()
