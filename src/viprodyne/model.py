"""Top-level model object for constructing viprodyne variational graphs."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from viprodyne.core.ms2_kernels import (
    MS2ObservationModel,
    ProximalKernel,
    build_ms2_observation_model,
    resolve_ms2_kernel,
)
from viprodyne.core.rate_edges import RateEdge, transition_states
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
)

FLOAT_DTYPE = np.float32


@dataclass(frozen=True)
class MS2Dataset:
    """Input data needed to instantiate one dataset plate."""

    name: str
    observed: np.ndarray
    noise_std: np.ndarray | float
    time_grid: np.ndarray | None = None
    sampling_times: np.ndarray | None = None
    finite_mask: np.ndarray | None = None
    contact_probability: np.ndarray | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("dataset name must be non-empty.")
        if self.time_grid is not None:
            _validate_time_grid(self.time_grid, "dataset.time_grid")
        if self.sampling_times is not None:
            sampling_times = np.asarray(self.sampling_times, dtype=FLOAT_DTYPE)
            observed = np.asarray(self.observed, dtype=FLOAT_DTYPE)
            if sampling_times.shape != observed.shape:
                raise ValueError("sampling_times must have the same shape as observed.")
            if np.any(np.diff(sampling_times) <= 0):
                raise ValueError("sampling_times must be strictly increasing.")


@dataclass(frozen=True)
class ModelConfig:
    """Graph-construction options for :class:`ViprodyneModel`."""

    n_states: int
    time_grid: np.ndarray | None = None
    initial_concentration: np.ndarray | None = None
    transition_prior_shape: np.ndarray | float = np.float32(1.0)
    transition_prior_rate: np.ndarray | float = np.float32(1.0)
    loading_prior_shape: np.ndarray | float = np.float32(1.0)
    loading_prior_rate: np.ndarray | float = np.float32(1.0)
    shared_transition_rates: bool = False
    shared_loading_rates: bool = False
    shared_transition_rate_indices: tuple[int, ...] = ()
    shared_loading_rate_states: tuple[int, ...] = ()
    pol2_mode: str = "auto"
    ms2_kernel: ProximalKernel | str | Callable | None = "proximal"
    t_rise: np.ndarray | float = np.float32(1.0)
    t_plateau: np.ndarray | float = np.float32(0.0)
    rna_intensity: np.ndarray | float = np.float32(1.0)
    kernel_support_tolerance: float = 1e-7
    driven_transition_indices: tuple[int, ...] = ()
    driven_rate_initial: np.ndarray | float = np.float32(1.0)
    driven_rate_bounds: tuple[float, float] = (1e-6, 1.0)
    driven_prior_shape: float = 1.0
    driven_prior_rate: float = 0.0

    def __post_init__(self) -> None:
        if self.n_states < 2:
            raise ValueError("n_states must be at least 2.")
        if self.time_grid is not None:
            _validate_time_grid(self.time_grid, "time_grid")
        if self.pol2_mode not in {"auto", "transfer", "mean_field", "exact"}:
            raise ValueError("pol2_mode must be 'auto', 'transfer', 'mean_field', or 'exact'.")
        if self.kernel_support_tolerance < 0:
            raise ValueError("kernel_support_tolerance must be non-negative.")
        resolve_ms2_kernel(
            self.ms2_kernel,
            self.t_rise,
            self.t_plateau,
            self.rna_intensity,
        )
        n_edges = self.n_states * (self.n_states - 1)
        object.__setattr__(
            self,
            "shared_transition_rate_indices",
            _validate_index_tuple(
                self.shared_transition_rate_indices,
                n_edges,
                "shared_transition_rate_indices",
            ),
        )
        object.__setattr__(
            self,
            "shared_loading_rate_states",
            _validate_index_tuple(
                self.shared_loading_rate_states,
                self.n_states,
                "shared_loading_rate_states",
            ),
        )
        driven_indices = tuple(int(index) for index in self.driven_transition_indices)
        if any(index < 0 or index >= n_edges for index in driven_indices):
            raise ValueError("driven_transition_indices must be valid transition indices.")
        object.__setattr__(self, "driven_transition_indices", driven_indices)
        lo_rate, hi_rate = self.driven_rate_bounds
        if not 0 < lo_rate < hi_rate:
            raise ValueError("driven_rate_bounds must satisfy 0 < lower < upper.")
        if self.driven_prior_shape <= 0:
            raise ValueError("driven_prior_shape must be positive.")
        if self.driven_prior_rate < 0:
            raise ValueError("driven_prior_rate must be non-negative.")


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
        for dataset in self.datasets:
            self._time_grid(dataset)
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
            if nodes["contact_drive"] is not None:
                schedule.append(nodes["contact_drive"])
            schedule.append(nodes["promoter"])
            if nodes["polymerase"] is not None:
                schedule.append(nodes["polymerase"])
        return tuple(dict.fromkeys(schedule))

    def _build_graph(self) -> None:
        for dataset in self.datasets:
            self.dataset_nodes[dataset.name] = self._add_dataset_plate(
                dataset,
            )

    def _add_dataset_plate(
        self,
        dataset: MS2Dataset,
    ) -> dict[str, str | list[str]]:
        time_grid = self._time_grid(dataset)
        initial_name = f"{dataset.name}:pi"
        initial = InitialStateProb(
            name=initial_name,
            prior_concentration=self._initial_concentration(),
        )
        self.graph.add_node(initial)

        transition_names = self._add_transition_rates(dataset.name)
        contact_drive_name = self._add_contact_drive(dataset)
        rate_edges = tuple(
            self._rate_edge(index, transition_name, contact_drive_name)
            for index, transition_name in enumerate(transition_names)
        )
        promoter_name = f"{dataset.name}:s"
        promoter = PromoterState(
            name=promoter_name,
            time_grid=time_grid,
            n_states=self.config.n_states,
            rate_edges=rate_edges,
            initial_probability_node=initial_name,
        )
        self.graph.add_node(promoter)
        self.graph.add_edge(initial_name, promoter_name)
        for transition_name in transition_names:
            self.graph.add_edge(transition_name, promoter_name)
        if contact_drive_name is not None:
            self.graph.add_edge(contact_drive_name, promoter_name)

        observed_name = f"{dataset.name}:I"
        observed = ObservedIntensity(
            name=observed_name,
            observed=dataset.observed,
            noise_std=dataset.noise_std,
            mask=dataset.finite_mask,
        )
        self.graph.add_node(observed)

        loading_names = self._add_loading_rates(dataset.name)

        pol2_observation = self._pol2_observation_model(dataset)
        polymerase_name = None
        if pol2_observation is not None:
            polymerase_name = f"{dataset.name}:tau"
            polymerase = PolymeraseLoadings(
                name=polymerase_name,
                observed=dataset.observed,
                design_matrix=pol2_observation.design_matrix,
                noise_std=dataset.noise_std,
                finite_mask=dataset.finite_mask,
                mode=pol2_observation.mode,
                window_weights=pol2_observation.window_weights,
                observation_starts=pol2_observation.observation_starts,
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
            "contact_drive": contact_drive_name,
            "transition_rates": transition_names,
            "loading_rates": loading_names,
        }

    def _add_transition_rates(self, dataset_name: str) -> list[str]:
        names = []
        shared_indices = self._shared_transition_indices()
        for index in range(self.config.n_states * (self.config.n_states - 1)):
            prefix = "shared" if index in shared_indices else dataset_name
            to_state, from_state = transition_states(self.config.n_states, index)
            name = f"{prefix}:R{index}"
            if name not in self.graph.nodes:
                if self._is_driven_transition(index):
                    self.graph.add_node(
                        DrivenRateMap(
                            name=name,
                            initial_rate=self.config.driven_rate_initial,
                            rate_bounds=self.config.driven_rate_bounds,
                            prior_shape=self.config.driven_prior_shape,
                            prior_rate=self.config.driven_prior_rate,
                        )
                    )
                else:
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

    def _add_loading_rates(self, dataset_name: str) -> list[str]:
        names = []
        shared_states = self._shared_loading_states()
        for state in range(self.config.n_states):
            prefix = "shared" if state in shared_states else dataset_name
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

    def _shared_transition_indices(self) -> set[int]:
        if self.config.shared_transition_rates:
            return set(range(self.config.n_states * (self.config.n_states - 1)))
        return set(self.config.shared_transition_rate_indices)

    def _shared_loading_states(self) -> set[int]:
        if self.config.shared_loading_rates:
            return set(range(self.config.n_states))
        return set(self.config.shared_loading_rate_states)

    def _initial_concentration(self) -> np.ndarray:
        if self.config.initial_concentration is None:
            return np.ones(self.config.n_states, dtype=FLOAT_DTYPE)
        concentration = np.asarray(self.config.initial_concentration, dtype=FLOAT_DTYPE)
        if concentration.shape != (self.config.n_states,):
            raise ValueError("initial_concentration must have shape (n_states,).")
        return concentration

    def _pol2_mode(self) -> str:
        if self.config.pol2_mode == "auto":
            if self.config.ms2_kernel is not None:
                return "transfer"
            return "mean_field"
        return self.config.pol2_mode

    def _pol2_observation_model(self, dataset: MS2Dataset) -> MS2ObservationModel | None:
        mode = self._pol2_mode()
        kernel = resolve_ms2_kernel(
            self.config.ms2_kernel,
            self.config.t_rise,
            self.config.t_plateau,
            self.config.rna_intensity,
        )
        if kernel is None:
            return None
        return build_ms2_observation_model(
            time_grid=self._time_grid(dataset),
            n_observations=np.asarray(dataset.observed).size,
            kernel=kernel,
            sampling_times=dataset.sampling_times,
            mode=mode,
            tolerance=self.config.kernel_support_tolerance,
        )

    def _is_driven_transition(self, transition_index: int) -> bool:
        return transition_index in self.config.driven_transition_indices

    def _rate_edge(
        self,
        transition_index: int,
        transition_name: str,
        contact_drive_name: str | None,
    ) -> RateEdge:
        to_state, from_state = transition_states(self.config.n_states, transition_index)
        drive_node = contact_drive_name if self._is_driven_transition(transition_index) else None
        return RateEdge(
            n_states=self.config.n_states,
            to_state=to_state,
            from_state=from_state,
            rate_node=transition_name,
            drive_node=drive_node,
            transition_index=transition_index,
        )

    def _add_contact_drive(self, dataset: MS2Dataset) -> str | None:
        if not self.config.driven_transition_indices:
            return None
        if dataset.contact_probability is None:
            raise ValueError("driven transition models require dataset.contact_probability.")
        contact = np.asarray(dataset.contact_probability, dtype=FLOAT_DTYPE)
        time_grid = self._time_grid(dataset)
        n_intervals = time_grid.size - 1
        if contact.ndim == 0 or contact.shape[-1] not in (n_intervals, n_intervals + 1):
            raise ValueError("contact_probability last axis must match intervals or grid points.")
        contact_name = f"{dataset.name}:rc"

        def fixed_contact_probability(times, rc):
            del times, rc
            return contact

        self.graph.add_node(
            RcNode(
                name=contact_name,
                value=np.float32(1.0),
                time_grid=time_grid,
                contact_probability_fn=fixed_contact_probability,
                pinned=True,
            )
        )
        return contact_name

    def _time_grid(self, dataset: MS2Dataset) -> np.ndarray:
        if dataset.time_grid is not None:
            return _validate_time_grid(dataset.time_grid, f"{dataset.name}.time_grid")
        if self.config.time_grid is not None:
            return _validate_time_grid(self.config.time_grid, "time_grid")
        raise ValueError("each dataset needs time_grid, or ModelConfig.time_grid must be set.")

def _validate_time_grid(time_grid: np.ndarray, name: str) -> np.ndarray:
    time_grid = np.asarray(time_grid, dtype=FLOAT_DTYPE)
    if time_grid.ndim != 1 or time_grid.size < 2:
        raise ValueError(f"{name} must be one-dimensional with at least two entries.")
    if np.any(np.diff(time_grid) <= 0):
        raise ValueError(f"{name} must be strictly increasing.")
    return time_grid


def _validate_index_tuple(indices: tuple[int, ...], upper_bound: int, name: str) -> tuple[int, ...]:
    values = tuple(int(index) for index in indices)
    if len(set(values)) != len(values):
        raise ValueError(f"{name} must not contain duplicates.")
    if any(index < 0 or index >= upper_bound for index in values):
        raise ValueError(f"{name} contains an out-of-range index.")
    return values
