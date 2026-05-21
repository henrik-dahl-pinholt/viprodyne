"""Top-level model object for constructing viprodyne variational graphs."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import numpy as np

from viprodyne.core.ms2_kernels import (
    MS2ObservationModel,
    ProximalKernel,
    build_ms2_observation_model,
    resolve_ms2_kernel,
)
from viprodyne.core.contact_survival import (
    ContactSurvivalStats,
    contact_survival_log_profile,
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
RateScope = Literal["track", "dataset", "global"]

if TYPE_CHECKING:
    from viprodyne.fit import CAVIResult


@dataclass(frozen=True)
class DatasetInferenceResult:
    """Posterior outputs for one fitted dataset.

    The arrays in this object are copied from the variational graph after
    fitting, so they can be inspected without touching internal node objects.
    `state_posterior` is indexed by trace, grid point, and promoter state.
    `loading_posterior` is indexed by trace and loading interval when a Pol2
    loading node is present.
    """

    name: str
    time_grid: np.ndarray
    sampling_times: np.ndarray | None
    observed: np.ndarray
    finite_mask: np.ndarray
    state_posterior: np.ndarray
    loading_posterior: np.ndarray | None
    predicted_signal: np.ndarray | None
    loading_mask: np.ndarray | None
    initial_probabilities: np.ndarray
    transition_rates: dict[int, np.ndarray]
    loading_rates: dict[int, np.ndarray]
    transition_rate_nodes: dict[int, str]
    loading_rate_nodes: dict[int, str]
    contact_rc: np.ndarray | None = None
    contact_probability: np.ndarray | None = None


@dataclass(frozen=True)
class ModelInferenceResult:
    """Structured result returned by `ViprodyneModel.run_inference`."""

    cavi: CAVIResult | None
    datasets: dict[str, DatasetInferenceResult]
    elbo_terms: dict[str, np.float32] | None = None


@dataclass(frozen=True)
class MS2Dataset:
    """Observed traces and timing information for one dataset plate.

    Parameters
    ----------
    observed:
        Fluorescence observations with shape `(n_traces, n_timepoints)`.
    noise_std:
        Observation noise standard deviation. Scalars and broadcastable arrays
        are accepted.
    name:
        Unique dataset name used in result dictionaries and graph node labels.
    rate_group:
        Optional label used when dataset-scoped rate nodes should be shared
        across several datasets.
    time_grid:
        Pol2 loading interval boundaries. If `sampling_times` is omitted,
        observations are assumed to occur at `time_grid[1:]`.
    sampling_times:
        Optional observation times with shape `(n_timepoints,)`.
    finite_mask:
        Optional boolean mask with the same shape as `observed`.
    contact_probability:
        Optional known contact probability for driven transitions.
    contact_score:
        Optional score that is thresholded by an `RcNode` to create contact
        probabilities.
    """

    observed: np.ndarray
    noise_std: np.ndarray | float
    name: str | None = None
    rate_group: str | None = None
    time_grid: np.ndarray | None = None
    sampling_times: np.ndarray | None = None
    finite_mask: np.ndarray | None = None
    contact_probability: np.ndarray | None = None
    contact_score: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.name is None:
            object.__setattr__(self, "name", f"dataset{np.random.randint(1_000_000)}")
        if self.rate_group is not None and not self.rate_group:
            raise ValueError("rate_group must be non-empty when provided.")
        observed = np.asarray(self.observed, dtype=FLOAT_DTYPE)
        if observed.ndim != 2:
            raise ValueError("observed must have shape (n_traces, n_timepoints).")
        if observed.shape[0] < 1 or observed.shape[1] < 1:
            raise ValueError("observed must have at least one trace and one timepoint.")
        if self.time_grid is not None:
            _validate_time_grid(self.time_grid, "dataset.time_grid")
        if self.sampling_times is not None:
            sampling_times = np.asarray(self.sampling_times, dtype=FLOAT_DTYPE)
            if sampling_times.shape != (observed.shape[1],):
                raise ValueError("sampling_times must have shape (n_timepoints,).")
            if np.any(np.diff(sampling_times) <= 0):
                raise ValueError("sampling_times must be strictly increasing.")
            if self.time_grid is None:
                dts = np.diff(sampling_times)
                left_edges = sampling_times[:-1] - 0.5 * dts
                final_edge = left_edges[-1] + dts[-1]
                inferred_grid = np.concatenate([left_edges, [final_edge]])
                object.__setattr__(self, "time_grid", inferred_grid)
                _validate_time_grid(
                    self.time_grid, "dataset.time_grid (inferred from sampling_times)"
                )
        if self.finite_mask is not None:
            mask = np.asarray(self.finite_mask, dtype=bool)
            if mask.shape != observed.shape:
                raise ValueError("finite_mask must have the same shape as observed.")
        if self.contact_probability is not None and self.contact_score is not None:
            raise ValueError(
                "pass either contact_probability or contact_score, not both."
            )

    @property
    def n_traces(self) -> int:
        """Number of traces in this dataset plate."""
        return int(np.asarray(self.observed).shape[0])

    @property
    def n_timepoints(self) -> int:
        """Number of observed timepoints per trace."""
        return int(np.asarray(self.observed).shape[1])


@dataclass(frozen=True)
class ModelConfig:
    """Options controlling model construction and fitting structure.

    This configuration specifies the number of promoter states, the MS2 kernel,
    Pol2 backend, rate-sharing scopes, priors, and optional driven-transition
    settings. Most users only need to set `n_states`, kernel parameters, and
    any rate-sharing or contact-drive options relevant to the experiment.
    """

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
    transition_rate_scope: RateScope = "dataset"
    loading_rate_scope: RateScope = "dataset"
    transition_rate_scopes: dict[int, RateScope] = field(default_factory=dict)
    loading_rate_scopes: dict[int, RateScope] = field(default_factory=dict)
    pol2_mode: str = "auto"
    ms2_kernel: ProximalKernel | str | Callable | None = "proximal"
    t_rise: np.ndarray | float = np.float32(1.0)
    t_plateau: np.ndarray | float = np.float32(0.0)
    rna_intensity: np.ndarray | float = np.float32(1.0)
    kernel_support_tolerance: float = 1e-7
    sampler_fine_grid: np.ndarray | None = None
    sampler_seed: int = 0
    sampler_iterations: int = 15_000
    sampler_repeats: int = 100
    sampler_compute_elbo: bool = False
    sampler_elbo_iterations: int = 10_000
    sampler_elbo_steps: int = 10
    sampler_elbo_repeats: int = 20
    driven_transition_indices: tuple[int, ...] = ()
    driven_rate_initial: np.ndarray | float = np.float32(1.0)
    driven_rate_bounds: tuple[float, float] = (1e-6, 1.0)
    driven_prior_shape: float = 1.0
    driven_prior_rate: float = 0.0
    rc_initial: np.ndarray | float = np.float32(0.5)
    rc_bounds: tuple[float, float] = (1e-6, 1.0)
    rc_candidate_values: np.ndarray | None = None
    contact_score_less_than: bool = True

    def __post_init__(self) -> None:
        if self.n_states < 2:
            raise ValueError("n_states must be at least 2.")
        if self.time_grid is not None:
            _validate_time_grid(self.time_grid, "time_grid")
        if self.pol2_mode not in {"auto", "transfer", "mean_field", "exact", "sampler"}:
            raise ValueError(
                "pol2_mode must be 'auto', 'transfer', 'mean_field', 'exact', or 'sampler'."
            )
        if self.kernel_support_tolerance < 0:
            raise ValueError("kernel_support_tolerance must be non-negative.")
        if self.sampler_fine_grid is not None:
            _validate_time_grid(self.sampler_fine_grid, "sampler_fine_grid")
        for name in (
            "sampler_iterations",
            "sampler_repeats",
            "sampler_elbo_iterations",
            "sampler_elbo_steps",
            "sampler_elbo_repeats",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive.")
        object.__setattr__(
            self,
            "transition_rate_scope",
            _validate_rate_scope(self.transition_rate_scope, "transition_rate_scope"),
        )
        object.__setattr__(
            self,
            "loading_rate_scope",
            _validate_rate_scope(self.loading_rate_scope, "loading_rate_scope"),
        )
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
        object.__setattr__(
            self,
            "transition_rate_scopes",
            _validate_scope_mapping(
                self.transition_rate_scopes,
                n_edges,
                "transition_rate_scopes",
            ),
        )
        object.__setattr__(
            self,
            "loading_rate_scopes",
            _validate_scope_mapping(
                self.loading_rate_scopes,
                self.n_states,
                "loading_rate_scopes",
            ),
        )
        driven_indices = tuple(int(index) for index in self.driven_transition_indices)
        if any(index < 0 or index >= n_edges for index in driven_indices):
            raise ValueError(
                "driven_transition_indices must be valid transition indices."
            )
        object.__setattr__(self, "driven_transition_indices", driven_indices)
        lo_rate, hi_rate = self.driven_rate_bounds
        if not 0 < lo_rate < hi_rate:
            raise ValueError("driven_rate_bounds must satisfy 0 < lower < upper.")
        if self.driven_prior_shape <= 0:
            raise ValueError("driven_prior_shape must be positive.")
        if self.driven_prior_rate < 0:
            raise ValueError("driven_prior_rate must be non-negative.")
        lo_rc, hi_rc = self.rc_bounds
        if not 0 < lo_rc < hi_rc:
            raise ValueError("rc_bounds must satisfy 0 < lower < upper.")
        if np.any(np.asarray(self.rc_initial, dtype=FLOAT_DTYPE) <= 0.0):
            raise ValueError("rc_initial must be positive.")
        if self.rc_candidate_values is not None:
            candidates = np.asarray(self.rc_candidate_values, dtype=FLOAT_DTYPE)
            if candidates.ndim != 1 or candidates.size == 0:
                raise ValueError(
                    "rc_candidate_values must be a non-empty one-dimensional array."
                )
            if np.any(candidates <= 0.0):
                raise ValueError("rc_candidate_values must be positive.")
            object.__setattr__(self, "rc_candidate_values", candidates)


@dataclass
class ViprodyneModel:
    """Top-level model object.

    Construct this class from one or more `MS2Dataset` objects and a
    `ModelConfig`, then call `run_inference` or `fit` to run coordinate-ascent
    variational inference.
    """

    datasets: tuple[MS2Dataset, ...]
    config: ModelConfig
    graph: VariationalGraph = field(default_factory=VariationalGraph, init=False)
    dataset_nodes: dict[str, dict[str, str | list[str]]] = field(
        default_factory=dict, init=False
    )

    def __post_init__(self) -> None:
        try:
            self.datasets = tuple(self.datasets)
        except TypeError as exc:
            raise ValueError(
                "datasets must be a tuple or list of MS2Dataset objects."
            ) from exc
        if not self.datasets:
            raise ValueError("at least one dataset is required.")
        if len({dataset.name for dataset in self.datasets}) != len(self.datasets):
            raise ValueError("dataset names must be unique.")
        _validate_rate_prefix_labels(self.datasets)
        for dataset in self.datasets:
            self._time_grid(dataset)
        self._build_graph()

    def run_schedule(
        self, schedule: list[str] | tuple[str, ...] | None = None, rho: float = 1.0
    ) -> None:
        """Run a graph update schedule."""
        self.graph.run_schedule(schedule=schedule, rho=rho)

    def fit_cavi(self, config=None, **kwargs):
        """Run coordinate-ascent variational inference for this model."""
        from viprodyne.fit import run_cavi

        config = self._cavi_config(config, kwargs)
        return run_cavi(self, config=config)

    def run_inference(self, config=None, **kwargs) -> ModelInferenceResult:
        """Run CAVI and return structured dataset-level posterior outputs."""
        config = self._cavi_config(config, kwargs)
        from viprodyne.fit import run_cavi

        cavi_result = run_cavi(self, config=config)
        return self.inference_result(
            cavi=cavi_result,
            include_elbo_terms=bool(config.compute_elbo),
        )

    def fit(self, config=None, **kwargs) -> ModelInferenceResult:
        """Alias for :meth:`run_inference` for the standard public workflow."""
        return self.run_inference(config=config, **kwargs)

    def inference_result(
        self,
        cavi=None,
        include_elbo_terms: bool = True,
    ) -> ModelInferenceResult:
        """Collect current graph moments into structured inference outputs."""
        return ModelInferenceResult(
            cavi=cavi,
            datasets={
                dataset.name: self.dataset_result(dataset.name)
                for dataset in self.datasets
            },
            elbo_terms=self.compute_elbo_terms() if include_elbo_terms else None,
        )

    def dataset_result(self, dataset_name: str) -> DatasetInferenceResult:
        """Return structured posterior outputs for one dataset plate."""
        try:
            dataset = next(item for item in self.datasets if item.name == dataset_name)
            nodes = self.dataset_nodes[dataset_name]
        except (KeyError, StopIteration) as exc:
            raise KeyError(f"unknown dataset {dataset_name!r}.") from exc

        promoter = self.graph.moments.get(str(nodes["promoter"]))
        observed = self.graph.moments.get(str(nodes["observed"]))
        initial = self.graph.moments.get(str(nodes["initial"]))
        polymerase_name = nodes["polymerase"]
        polymerase = (
            self.graph.moments.get(str(polymerase_name))
            if polymerase_name is not None
            else {}
        )
        contact_name = nodes["contact_drive"]
        contact = (
            self.graph.moments.get(str(contact_name))
            if contact_name is not None
            else {}
        )
        sampling_times = None
        if polymerase_name is not None:
            sampling_times = np.asarray(
                self.graph.nodes[str(polymerase_name)].sampling_times,
                dtype=FLOAT_DTYPE,
            ).copy()
        elif dataset.sampling_times is not None:
            sampling_times = np.asarray(
                dataset.sampling_times, dtype=FLOAT_DTYPE
            ).copy()

        transition_rate_nodes = {
            index: name for index, name in enumerate(list(nodes["transition_rates"]))
        }
        loading_rate_nodes = {
            index: name for index, name in enumerate(list(nodes["loading_rates"]))
        }
        return DatasetInferenceResult(
            name=dataset.name,
            time_grid=self._time_grid(dataset).copy(),
            sampling_times=sampling_times,
            observed=np.asarray(observed["observed"], dtype=FLOAT_DTYPE).copy(),
            finite_mask=np.asarray(observed["finite_mask"], dtype=bool).copy(),
            state_posterior=np.asarray(promoter["posterior"], dtype=FLOAT_DTYPE).copy(),
            loading_posterior=_optional_float_moment(polymerase, "load_probabilities"),
            predicted_signal=_optional_float_moment(polymerase, "predicted_signal"),
            loading_mask=_optional_bool_moment(polymerase, "loading_mask"),
            initial_probabilities=np.asarray(initial["mean"], dtype=FLOAT_DTYPE).copy(),
            transition_rates={
                index: np.asarray(
                    self.graph.moments.get(name)["mean"],
                    dtype=FLOAT_DTYPE,
                ).copy()
                for index, name in transition_rate_nodes.items()
            },
            loading_rates={
                index: np.asarray(
                    self.graph.moments.get(name)["mean"],
                    dtype=FLOAT_DTYPE,
                ).copy()
                for index, name in loading_rate_nodes.items()
            },
            transition_rate_nodes=transition_rate_nodes,
            loading_rate_nodes=loading_rate_nodes,
            contact_rc=_optional_float_moment(contact, "rc"),
            contact_probability=_optional_float_moment(contact, "p_contact"),
        )

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

    def cavi_schedule(self) -> tuple[str, ...]:
        """Return a CAVI sweep with hidden nodes before parameter nodes."""
        hidden: list[str] = []
        parameters: list[str] = []
        for nodes in self.dataset_nodes.values():
            hidden.append(nodes["promoter"])
            if nodes["polymerase"] is not None:
                hidden.append(nodes["polymerase"])
            parameters.append(nodes["initial"])
            parameters.extend(nodes["transition_rates"])
            parameters.extend(nodes["loading_rates"])
            if nodes["contact_drive"] is not None:
                parameters.append(nodes["contact_drive"])
        return tuple(dict.fromkeys(hidden + parameters))

    def parameter_node_names(self) -> tuple[str, ...]:
        """Return names of nodes used for CAVI convergence monitoring."""
        names = [
            name
            for name, node in self.graph.nodes.items()
            if isinstance(
                node,
                (
                    InitialStateProb,
                    LoadingRate,
                    TransitionRate,
                    DrivenRateMap,
                    RcNode,
                ),
            )
        ]
        return tuple(names)

    def compute_elbo_terms(self) -> dict[str, np.float32]:
        """Compute available local ELBO contributions for all graph nodes."""
        return {
            name: np.asarray(node.elbo_contribution(), dtype=FLOAT_DTYPE)
            for name, node in self.graph.nodes.items()
        }

    def compute_elbo(self) -> np.float32:
        """Compute the current available model ELBO."""
        terms = self.compute_elbo_terms()
        return np.asarray(
            sum(float(value) for value in terms.values()), dtype=FLOAT_DTYPE
        )

    def _cavi_config(self, config, kwargs):
        from viprodyne.fit import CAVIConfig

        if config is not None and kwargs:
            raise ValueError("pass either config or keyword arguments, not both.")
        if config is None:
            return CAVIConfig(**kwargs)
        return config

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
        contact_drive_name = self._add_contact_drive(dataset, transition_names)
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
                sampling_times=pol2_observation.sampling_times,
                fine_grid=self._sampler_fine_grid(pol2_observation),
                rise_time=self._proximal_kernel().t_rise,
                plateau_time=self._proximal_kernel().t_plateau,
                rna_intensity=self._proximal_kernel().rna_intensity,
                sampler_seed=self.config.sampler_seed,
                sampler_iterations=self.config.sampler_iterations,
                sampler_repeats=self.config.sampler_repeats,
                sampler_compute_elbo=self.config.sampler_compute_elbo,
                sampler_elbo_iterations=self.config.sampler_elbo_iterations,
                sampler_elbo_steps=self.config.sampler_elbo_steps,
                sampler_elbo_repeats=self.config.sampler_elbo_repeats,
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
        for index in range(self.config.n_states * (self.config.n_states - 1)):
            scope = self._transition_rate_scope(index)
            prefix = self._rate_prefix(dataset_name, scope)
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
                            prior_shape=self._rate_prior_parameter(
                                self.config.transition_prior_shape,
                                dataset_name,
                                scope,
                                "transition_prior_shape",
                            ),
                            prior_rate=self._rate_prior_parameter(
                                self.config.transition_prior_rate,
                                dataset_name,
                                scope,
                                "transition_prior_rate",
                            ),
                            n_states=self.config.n_states,
                            to_state=to_state,
                            from_state=from_state,
                        )
                    )
            names.append(name)
        return names

    def _add_loading_rates(self, dataset_name: str) -> list[str]:
        names = []
        for state in range(self.config.n_states):
            scope = self._loading_rate_scope(state)
            prefix = self._rate_prefix(dataset_name, scope)
            name = f"{prefix}:r{state}"
            if name not in self.graph.nodes:
                self.graph.add_node(
                    LoadingRate(
                        name=name,
                        prior_shape=self._rate_prior_parameter(
                            self.config.loading_prior_shape,
                            dataset_name,
                            scope,
                            "loading_prior_shape",
                        ),
                        prior_rate=self._rate_prior_parameter(
                            self.config.loading_prior_rate,
                            dataset_name,
                            scope,
                            "loading_prior_rate",
                        ),
                        state_index=state,
                    )
                )
            names.append(name)
        return names

    def _transition_rate_scope(self, index: int) -> RateScope:
        if self.config.shared_transition_rates:
            return "global"
        if index in self.config.shared_transition_rate_indices:
            return "global"
        return self.config.transition_rate_scopes.get(
            index, self.config.transition_rate_scope
        )

    def _loading_rate_scope(self, state: int) -> RateScope:
        if self.config.shared_loading_rates:
            return "global"
        if state in self.config.shared_loading_rate_states:
            return "global"
        return self.config.loading_rate_scopes.get(
            state, self.config.loading_rate_scope
        )

    def _rate_prefix(self, dataset_name: str, scope: RateScope) -> str:
        if scope == "global":
            return "shared"
        if scope == "track":
            return dataset_name
        if scope == "dataset":
            dataset = next(item for item in self.datasets if item.name == dataset_name)
            return dataset.rate_group or dataset.name
        raise ValueError(f"unknown rate scope {scope!r}.")

    def _rate_prior_parameter(
        self,
        value: np.ndarray | float,
        dataset_name: str,
        scope: RateScope,
        name: str,
    ) -> np.ndarray:
        parameter = np.asarray(value, dtype=FLOAT_DTYPE)
        if scope != "track":
            return parameter
        n_traces = next(
            dataset.n_traces
            for dataset in self.datasets
            if dataset.name == dataset_name
        )
        if parameter.shape == ():
            return np.full((n_traces,), parameter, dtype=FLOAT_DTYPE)
        if parameter.shape == (n_traces,):
            return parameter.astype(FLOAT_DTYPE)
        raise ValueError(
            f"{name} must be scalar or have shape (n_traces,) for track scope."
        )

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

    def _pol2_observation_model(
        self, dataset: MS2Dataset
    ) -> MS2ObservationModel | None:
        mode = self._pol2_mode()
        if mode == "sampler":
            self._validate_sampler_kernel()
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
            n_observations=dataset.n_timepoints,
            kernel=kernel,
            sampling_times=dataset.sampling_times,
            mode=mode,
            tolerance=self.config.kernel_support_tolerance,
        )

    def _sampler_fine_grid(self, observation: MS2ObservationModel) -> np.ndarray | None:
        if observation.mode != "sampler":
            return None
        if self.config.sampler_fine_grid is not None:
            return _validate_time_grid(
                self.config.sampler_fine_grid, "sampler_fine_grid"
            )
        return np.asarray(observation.loading_times, dtype=FLOAT_DTYPE)

    def _proximal_kernel(self) -> ProximalKernel:
        if isinstance(self.config.ms2_kernel, ProximalKernel):
            return self.config.ms2_kernel
        return ProximalKernel(
            t_rise=self.config.t_rise,
            t_plateau=self.config.t_plateau,
            rna_intensity=self.config.rna_intensity,
        )

    def _validate_sampler_kernel(self) -> None:
        if isinstance(self.config.ms2_kernel, ProximalKernel):
            return
        if self.config.ms2_kernel is None:
            raise ValueError("sampler Pol2 mode requires a proximal MS2 kernel.")
        if isinstance(self.config.ms2_kernel, str):
            name = self.config.ms2_kernel.lower()
            if name == "proximal":
                return
        raise ValueError("sampler Pol2 mode currently supports only ProximalKernel.")

    def _is_driven_transition(self, transition_index: int) -> bool:
        return transition_index in self.config.driven_transition_indices

    def _rate_edge(
        self,
        transition_index: int,
        transition_name: str,
        contact_drive_name: str | None,
    ) -> RateEdge:
        to_state, from_state = transition_states(self.config.n_states, transition_index)
        drive_node = (
            contact_drive_name if self._is_driven_transition(transition_index) else None
        )
        return RateEdge(
            n_states=self.config.n_states,
            to_state=to_state,
            from_state=from_state,
            rate_node=transition_name,
            drive_node=drive_node,
            transition_index=transition_index,
        )

    def _add_contact_drive(
        self, dataset: MS2Dataset, transition_names: list[str]
    ) -> str | None:
        if not self.config.driven_transition_indices:
            return None
        time_grid = self._time_grid(dataset)
        n_intervals = time_grid.size - 1
        contact_name = f"{dataset.name}:rc"
        if dataset.contact_probability is not None:
            contact = np.asarray(dataset.contact_probability, dtype=FLOAT_DTYPE)
            if contact.ndim == 0 or contact.shape[-1] not in (
                n_intervals,
                n_intervals + 1,
            ):
                raise ValueError(
                    "contact_probability last axis must match intervals or grid points."
                )

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

        if dataset.contact_score is None:
            raise ValueError(
                "driven transition models require dataset.contact_probability or "
                "dataset.contact_score."
            )
        contact_score = np.asarray(dataset.contact_score, dtype=FLOAT_DTYPE)
        if contact_score.ndim == 0 or contact_score.shape[-1] not in (
            n_intervals,
            n_intervals + 1,
        ):
            raise ValueError(
                "contact_score last axis must match intervals or grid points."
            )

        def threshold_contact_probability(times, rc):
            del times
            return _threshold_contact_score(
                contact_score,
                rc,
                less_than=self.config.contact_score_less_than,
            )

        objective = self._contact_threshold_objective(
            contact_score=contact_score,
            transition_names=transition_names,
            less_than=self.config.contact_score_less_than,
        )
        candidate_values = (
            self.config.rc_candidate_values
            if self.config.rc_candidate_values is not None
            else _default_rc_candidate_values(contact_score, self.config.rc_bounds)
        )
        self.graph.add_node(
            RcNode(
                name=contact_name,
                value=self.config.rc_initial,
                time_grid=time_grid,
                contact_probability_fn=threshold_contact_probability,
                bounds=self.config.rc_bounds,
                objective_fn=objective,
                candidate_values=candidate_values,
            )
        )
        return contact_name

    def _contact_threshold_objective(
        self,
        *,
        contact_score: np.ndarray,
        transition_names: list[str],
        less_than: bool,
    ):
        driven_edges = tuple(
            (
                index,
                transition_states(self.config.n_states, index),
                transition_names[index],
            )
            for index in self.config.driven_transition_indices
        )

        def objective(rc_value, context) -> float:
            child_moments = context.child_moments()
            blanket_moments = context.blanket_moments()
            total = 0.0
            for moments in child_moments.values():
                if (
                    not {"expected_occupancy", "expected_jumps", "interval_durations"}
                    <= moments.keys()
                ):
                    continue
                occupancy = np.asarray(moments["expected_occupancy"], dtype=FLOAT_DTYPE)
                jumps = np.asarray(moments["expected_jumps"], dtype=FLOAT_DTYPE)
                interval_durations = np.asarray(
                    moments["interval_durations"], dtype=FLOAT_DTYPE
                )
                dt = _constant_interval_duration(interval_durations)
                p_contact = _interval_threshold_contact_score(
                    contact_score,
                    rc_value,
                    n_intervals=interval_durations.size,
                    less_than=less_than,
                )
                for _, (to_state, from_state), rate_name in driven_edges:
                    if rate_name not in blanket_moments:
                        continue
                    rate = np.asarray(
                        blanket_moments[rate_name]["mean"],
                        dtype=FLOAT_DTYPE,
                    )
                    gamma_from = occupancy[..., from_state] / np.float32(dt)
                    gamma_jump = jumps[..., to_state, from_state] / np.float32(dt)
                    p_broadcast = np.broadcast_to(p_contact, gamma_from.shape).astype(
                        FLOAT_DTYPE
                    )
                    total += _contact_survival_log_likelihood(
                        log_rate=np.log(np.clip(rate, 1e-20, None)).astype(FLOAT_DTYPE),
                        gamma_jump=gamma_jump,
                        gamma_from=gamma_from,
                        p_contact=p_broadcast,
                        dt=dt,
                    )
            return float(total)

        return objective

    def _time_grid(self, dataset: MS2Dataset) -> np.ndarray:
        if dataset.time_grid is not None:
            return _validate_time_grid(dataset.time_grid, f"{dataset.name}.time_grid")
        if self.config.time_grid is not None:
            return _validate_time_grid(self.config.time_grid, "time_grid")
        raise ValueError(
            "each dataset needs time_grid, or ModelConfig.time_grid must be set."
        )


def _validate_time_grid(time_grid: np.ndarray, name: str) -> np.ndarray:
    time_grid = np.asarray(time_grid, dtype=FLOAT_DTYPE)
    if time_grid.ndim != 1 or time_grid.size < 2:
        raise ValueError(f"{name} must be one-dimensional with at least two entries.")
    if np.any(np.diff(time_grid) <= 0):
        raise ValueError(f"{name} must be strictly increasing.")
    return time_grid


def _threshold_contact_score(
    score: np.ndarray,
    threshold: np.ndarray,
    *,
    less_than: bool,
) -> np.ndarray:
    score = np.asarray(score, dtype=FLOAT_DTYPE)
    threshold = np.asarray(threshold, dtype=FLOAT_DTYPE)
    contact = score < threshold if less_than else score > threshold
    return np.asarray(contact, dtype=FLOAT_DTYPE)


def _interval_threshold_contact_score(
    score: np.ndarray,
    threshold: np.ndarray,
    *,
    n_intervals: int,
    less_than: bool,
) -> np.ndarray:
    contact = _threshold_contact_score(score, threshold, less_than=less_than)
    if contact.shape[-1] == n_intervals + 1:
        contact = contact[..., 1:]
    elif contact.shape[-1] != n_intervals:
        raise ValueError("contact_score last axis must match intervals or grid points.")
    return np.clip(contact, 0.0, 1.0).astype(FLOAT_DTYPE)


def _constant_interval_duration(interval_durations: np.ndarray) -> float:
    durations = np.asarray(interval_durations, dtype=FLOAT_DTYPE)
    if durations.ndim != 1 or durations.size == 0:
        raise ValueError(
            "interval_durations must be a non-empty one-dimensional array."
        )
    if not np.allclose(durations, durations[0], rtol=1e-6, atol=1e-7):
        raise ValueError(
            "RcNode threshold updates currently require a uniform time grid."
        )
    return float(durations[0])


def _default_rc_candidate_values(
    contact_score: np.ndarray,
    bounds: tuple[float, float],
) -> np.ndarray:
    lo, hi = map(np.float32, bounds)
    values = np.unique(np.ravel(np.asarray(contact_score, dtype=FLOAT_DTYPE)))
    values = values[np.isfinite(values)]
    values = values[(values > lo) & (values < hi)]
    breaks = np.unique(
        np.concatenate([np.asarray([lo, hi], dtype=FLOAT_DTYPE), values])
    )
    if breaks.size == 1:
        return breaks.astype(FLOAT_DTYPE)
    midpoints = (breaks[:-1] + breaks[1:]) / np.float32(2.0)
    candidates = np.unique(
        np.concatenate(
            [np.asarray([lo, hi], dtype=FLOAT_DTYPE), midpoints.astype(FLOAT_DTYPE)]
        )
    )
    return candidates[(candidates >= lo) & (candidates <= hi)].astype(FLOAT_DTYPE)


def _contact_survival_log_likelihood(
    *,
    log_rate: np.ndarray,
    gamma_jump: np.ndarray,
    gamma_from: np.ndarray,
    p_contact: np.ndarray,
    dt: float,
) -> float:
    log_rate = np.asarray(log_rate, dtype=FLOAT_DTYPE)
    gamma_jump = np.asarray(gamma_jump, dtype=FLOAT_DTYPE)
    gamma_from = np.asarray(gamma_from, dtype=FLOAT_DTYPE)
    p_contact = np.asarray(p_contact, dtype=FLOAT_DTYPE)
    if gamma_jump.shape != gamma_from.shape or gamma_from.shape != p_contact.shape:
        raise ValueError(
            "gamma_jump, gamma_from, and p_contact must have matching shapes."
        )
    if log_rate.shape == ():
        stats = ContactSurvivalStats.from_posteriors(
            gamma_jump=gamma_jump,
            gamma_from=gamma_from,
            p_contact=p_contact,
            dt=dt,
        )
        return contact_survival_log_profile(float(log_rate), stats)

    batch_shape = np.broadcast_shapes(log_rate.shape, gamma_from.shape[:-1])
    n_intervals = gamma_from.shape[-1]
    log_rate = np.broadcast_to(log_rate, batch_shape).astype(FLOAT_DTYPE)
    gamma_jump = np.broadcast_to(gamma_jump, batch_shape + (n_intervals,)).astype(
        FLOAT_DTYPE
    )
    gamma_from = np.broadcast_to(gamma_from, batch_shape + (n_intervals,)).astype(
        FLOAT_DTYPE
    )
    p_contact = np.broadcast_to(p_contact, batch_shape + (n_intervals,)).astype(
        FLOAT_DTYPE
    )
    total = 0.0
    for index in np.ndindex(batch_shape):
        stats = ContactSurvivalStats.from_posteriors(
            gamma_jump=gamma_jump[index],
            gamma_from=gamma_from[index],
            p_contact=p_contact[index],
            dt=dt,
        )
        total += contact_survival_log_profile(float(log_rate[index]), stats)
    return float(total)


def _optional_float_moment(moments: dict, key: str) -> np.ndarray | None:
    if key not in moments:
        return None
    return np.asarray(moments[key], dtype=FLOAT_DTYPE).copy()


def _optional_bool_moment(moments: dict, key: str) -> np.ndarray | None:
    if key not in moments:
        return None
    return np.asarray(moments[key], dtype=bool).copy()


def _validate_index_tuple(
    indices: tuple[int, ...], upper_bound: int, name: str
) -> tuple[int, ...]:
    values = tuple(int(index) for index in indices)
    if len(set(values)) != len(values):
        raise ValueError(f"{name} must not contain duplicates.")
    if any(index < 0 or index >= upper_bound for index in values):
        raise ValueError(f"{name} contains an out-of-range index.")
    return values


def _validate_rate_prefix_labels(datasets: tuple[MS2Dataset, ...]) -> None:
    labels = []
    for dataset in datasets:
        labels.append(("dataset name", dataset.name))
        if dataset.rate_group is not None:
            labels.append(("rate_group", dataset.rate_group))
    for label_type, label in labels:
        if ":" in label:
            raise ValueError(f"{label_type} {label!r} must not contain ':'.")
        if label == "shared":
            raise ValueError(
                f"{label_type} {label!r} is reserved for global rate nodes."
            )


def _validate_rate_scope(scope: str, name: str) -> RateScope:
    if scope not in {"track", "dataset", "global"}:
        raise ValueError(f"{name} must be 'track', 'dataset', or 'global'.")
    return scope


def _validate_scope_mapping(
    mapping: dict[int, RateScope],
    upper_bound: int,
    name: str,
) -> dict[int, RateScope]:
    normalized: dict[int, RateScope] = {}
    for raw_index, raw_scope in dict(mapping).items():
        index = int(raw_index)
        if index < 0 or index >= upper_bound:
            raise ValueError(f"{name} contains an out-of-range index.")
        normalized[index] = _validate_rate_scope(raw_scope, name)
    return normalized
