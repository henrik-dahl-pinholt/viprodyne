"""Top-level model object for constructing viprodyne variational graphs."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from viprodyne.core.rate_edges import RateEdge, transition_states
from viprodyne.variational import (
    InitialStateProb,
    LoadingRate,
    ObservedIntensity,
    PolymeraseLoadings,
    PromoterState,
    TransitionRate,
    VariationalGraph,
)

FLOAT_DTYPE = np.float32


@dataclass(frozen=True)
class MS2Dataset:
    """Input data needed to instantiate one dataset plate."""

    name: str
    observed: np.ndarray
    noise_std: np.ndarray | float
    design_matrix: np.ndarray | None = None
    prior_load_probabilities: np.ndarray | None = None
    finite_mask: np.ndarray | None = None
    window_weights: np.ndarray | None = None
    observation_starts: np.ndarray | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("dataset name must be non-empty.")
        if self.prior_load_probabilities is not None and self.design_matrix is None:
            if self.window_weights is None or self.observation_starts is None:
                raise ValueError(
                    "Pol2 datasets need design_matrix or transfer window_weights/observation_starts."
                )


@dataclass(frozen=True)
class ModelConfig:
    """Graph-construction options for :class:`ViprodyneModel`."""

    n_states: int
    time_grid: np.ndarray
    initial_concentration: np.ndarray | None = None
    transition_prior_shape: np.ndarray | float = np.float32(1.0)
    transition_prior_rate: np.ndarray | float = np.float32(1.0)
    loading_prior_shape: np.ndarray | float = np.float32(1.0)
    loading_prior_rate: np.ndarray | float = np.float32(1.0)
    shared_transition_rates: bool = False
    shared_loading_rates: bool = False
    pol2_mode: str = "mean_field"

    def __post_init__(self) -> None:
        if self.n_states < 2:
            raise ValueError("n_states must be at least 2.")
        time_grid = np.asarray(self.time_grid, dtype=FLOAT_DTYPE)
        if time_grid.ndim != 1 or time_grid.size < 2:
            raise ValueError("time_grid must be one-dimensional with at least two entries.")
        if np.any(np.diff(time_grid) <= 0):
            raise ValueError("time_grid must be strictly increasing.")


@dataclass
class ViprodyneModel:
    """High-level interface that owns graph construction and update scheduling."""

    datasets: tuple[MS2Dataset, ...]
    config: ModelConfig
    graph: VariationalGraph = field(default_factory=VariationalGraph, init=False)
    dataset_nodes: dict[str, dict[str, str | list[str]]] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        self.datasets = tuple(self.datasets)
        if not self.datasets:
            raise ValueError("at least one dataset is required.")
        if len({dataset.name for dataset in self.datasets}) != len(self.datasets):
            raise ValueError("dataset names must be unique.")
        self._build_graph()

    def run_schedule(self, schedule: list[str] | tuple[str, ...] | None = None, rho: float = 1.0) -> None:
        """Run a graph update schedule."""
        self.graph.run_schedule(schedule=schedule, rho=rho)

    def default_schedule(self) -> tuple[str, ...]:
        """Return a conservative default schedule over non-observed nodes."""
        schedule: list[str] = []
        for nodes in self.dataset_nodes.values():
            schedule.extend(nodes["transition_rates"])
            schedule.extend(nodes["loading_rates"])
            schedule.append(nodes["initial"])
            schedule.append(nodes["promoter"])
            if nodes["polymerase"] is not None:
                schedule.append(nodes["polymerase"])
        return tuple(dict.fromkeys(schedule))

    def _build_graph(self) -> None:
        shared_transition_names = self._add_shared_transition_rates()
        shared_loading_names = self._add_shared_loading_rates()
        for dataset in self.datasets:
            self.dataset_nodes[dataset.name] = self._add_dataset_plate(
                dataset,
                shared_transition_names=shared_transition_names,
                shared_loading_names=shared_loading_names,
            )

    def _add_dataset_plate(
        self,
        dataset: MS2Dataset,
        shared_transition_names: list[str] | None,
        shared_loading_names: list[str] | None,
    ) -> dict[str, str | list[str]]:
        initial_name = f"{dataset.name}:pi"
        initial = InitialStateProb(
            name=initial_name,
            prior_concentration=self._initial_concentration(),
        )
        self.graph.add_node(initial)

        transition_names = shared_transition_names
        if transition_names is None:
            transition_names = self._add_transition_rates(dataset.name)
        rate_edges = tuple(
            RateEdge(
                n_states=self.config.n_states,
                to_state=transition_states(self.config.n_states, index)[0],
                from_state=transition_states(self.config.n_states, index)[1],
                rate_node=transition_name,
                transition_index=index,
            )
            for index, transition_name in enumerate(transition_names)
        )
        promoter_name = f"{dataset.name}:s"
        promoter = PromoterState(
            name=promoter_name,
            time_grid=self.config.time_grid,
            n_states=self.config.n_states,
            rate_edges=rate_edges,
            initial_probability_node=initial_name,
        )
        self.graph.add_node(promoter)
        self.graph.add_edge(initial_name, promoter_name)
        for transition_name in transition_names:
            self.graph.add_edge(transition_name, promoter_name)

        observed_name = f"{dataset.name}:I"
        observed = ObservedIntensity(
            name=observed_name,
            observed=dataset.observed,
            noise_std=dataset.noise_std,
            mask=dataset.finite_mask,
        )
        self.graph.add_node(observed)

        loading_names = shared_loading_names
        if loading_names is None:
            loading_names = self._add_loading_rates(dataset.name)

        polymerase_name = None
        if dataset.prior_load_probabilities is not None:
            polymerase_name = f"{dataset.name}:tau"
            polymerase = PolymeraseLoadings(
                name=polymerase_name,
                observed=dataset.observed,
                prior_probabilities=dataset.prior_load_probabilities,
                design_matrix=dataset.design_matrix,
                noise_std=dataset.noise_std,
                finite_mask=dataset.finite_mask,
                mode=self._pol2_mode(dataset),
                window_weights=dataset.window_weights,
                observation_starts=dataset.observation_starts,
            )
            self.graph.add_node(polymerase)
            self.graph.add_edge(promoter_name, polymerase_name)
            for loading_name in loading_names:
                self.graph.add_edge(loading_name, polymerase_name)
            self.graph.add_edge(polymerase_name, observed_name)

        return {
            "initial": initial_name,
            "promoter": promoter_name,
            "observed": observed_name,
            "polymerase": polymerase_name,
            "transition_rates": transition_names,
            "loading_rates": loading_names,
        }

    def _add_shared_transition_rates(self) -> list[str] | None:
        if not self.config.shared_transition_rates:
            return None
        return self._add_transition_rates("shared")

    def _add_transition_rates(self, prefix: str) -> list[str]:
        names = []
        for index in range(self.config.n_states * (self.config.n_states - 1)):
            to_state, from_state = transition_states(self.config.n_states, index)
            name = f"{prefix}:R{index}"
            if name not in self.graph.nodes:
                self.graph.add_node(
                    TransitionRate(
                        name=name,
                        prior_shape=self.config.transition_prior_shape,
                        prior_rate=self.config.transition_prior_rate,
                        n_states=self.config.n_states,
                        to_state=to_state,
                        from_state=from_state,
                    )
                )
            names.append(name)
        return names

    def _add_shared_loading_rates(self) -> list[str] | None:
        if not self.config.shared_loading_rates:
            return None
        return self._add_loading_rates("shared")

    def _add_loading_rates(self, prefix: str) -> list[str]:
        names = []
        for state in range(self.config.n_states):
            name = f"{prefix}:r{state}"
            if name not in self.graph.nodes:
                self.graph.add_node(
                    LoadingRate(
                        name=name,
                        prior_shape=self.config.loading_prior_shape,
                        prior_rate=self.config.loading_prior_rate,
                        state_index=state,
                    )
                )
            names.append(name)
        return names

    def _initial_concentration(self) -> np.ndarray:
        if self.config.initial_concentration is None:
            return np.ones(self.config.n_states, dtype=FLOAT_DTYPE)
        concentration = np.asarray(self.config.initial_concentration, dtype=FLOAT_DTYPE)
        if concentration.shape != (self.config.n_states,):
            raise ValueError("initial_concentration must have shape (n_states,).")
        return concentration

    def _pol2_mode(self, dataset: MS2Dataset) -> str:
        if self.config.pol2_mode != "transfer":
            return self.config.pol2_mode
        if dataset.window_weights is None or dataset.observation_starts is None:
            return "mean_field"
        return "transfer"
