"""Top-level model object for constructing viprodyne variational graphs."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
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
    `state_posterior` is indexed by trace, requested state-posterior time,
    and promoter state. `loading_posterior` is indexed by trace and requested
    loading-posterior time when a Pol2 loading node is present.
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
    state_posterior_times: np.ndarray | None = None
    loading_posterior_times: np.ndarray | None = None
    loading_posterior_rate: np.ndarray | None = None

    def __str__(self) -> str:
        parts = [
            f"DatasetInferenceResult(name={self.name!r})",
            f"  traces={self.observed.shape[0]}, timepoints={self.observed.shape[1]}",
            f"  states={self.state_posterior.shape[-1]}",
            f"  transition_rates={len(self.transition_rates)}, loading_rates={len(self.loading_rates)}",
        ]
        if self.contact_rc is not None:
            parts.append(f"  contact_rc={np.asarray(self.contact_rc).item():.6g}")
        if self.loading_posterior is not None:
            finite_loads = np.asarray(self.loading_posterior)[np.isfinite(self.loading_posterior)]
            if finite_loads.size:
                parts.append(f"  mean_loading_posterior={float(np.mean(finite_loads)):.6g}")
        return "\n".join(parts)


@dataclass(frozen=True)
class ModelInferenceResult:
    """Structured result returned by `ViprodyneModel.run_inference`."""

    cavi: CAVIResult | None
    datasets: dict[str, DatasetInferenceResult]
    elbo_terms: dict[str, np.float32] | None = None

    def __str__(self) -> str:
        cavi = "no CAVI diagnostics" if self.cavi is None else str(self.cavi)
        dataset_names = ", ".join(sorted(self.datasets))
        return f"ModelInferenceResult(datasets=[{dataset_names}])\n{cavi}"


ContactInputSpec = np.ndarray | Callable


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
    dt:
        Optional frame spacing for regularly sampled traces. If provided,
        observations are treated as centered in each frame.
    name:
        Optional dataset name used in result dictionaries and graph node labels.
        If omitted, `ViprodyneModel` assigns deterministic names like
        `dataset_0`.
    rate_group:
        Optional label used when dataset-scoped rate nodes should be shared
        across several datasets.
    sampling_times:
        Optional observation times with shape `(n_timepoints,)`.
    time_grid:
        Advanced internal loading interval boundaries. Prefer `dt` or
        `sampling_times` for public workflows.
    finite_mask:
        Optional boolean mask with the same shape as `observed`.
    """

    observed: np.ndarray
    noise_std: np.ndarray | float
    dt: np.ndarray | float | None = None
    name: str | None = None
    rate_group: str | None = None
    time_grid: np.ndarray | None = None
    sampling_times: np.ndarray | None = None
    finite_mask: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.name is not None and not self.name:
            raise ValueError("dataset name must be non-empty when provided.")
        if self.rate_group is not None and not self.rate_group:
            raise ValueError("rate_group must be non-empty when provided.")
        observed = np.asarray(self.observed, dtype=FLOAT_DTYPE)
        if observed.ndim != 2:
            raise ValueError("observed must have shape (n_traces, n_timepoints).")
        if observed.shape[0] < 1 or observed.shape[1] < 1:
            raise ValueError("observed must have at least one trace and one timepoint.")
        if self.dt is not None:
            dt = _validate_positive_scalar(self.dt, "dt")
            sampling_times = (
                np.arange(observed.shape[1], dtype=FLOAT_DTYPE) + np.float32(0.5)
            ) * dt
            time_grid = np.arange(observed.shape[1] + 1, dtype=FLOAT_DTYPE) * dt
            if self.sampling_times is not None:
                existing_sampling_times = np.asarray(self.sampling_times, dtype=FLOAT_DTYPE)
                if existing_sampling_times.shape != sampling_times.shape or not np.allclose(
                    existing_sampling_times,
                    sampling_times,
                ):
                    raise ValueError("dt cannot be combined with different sampling_times.")
            if self.time_grid is not None:
                existing_time_grid = np.asarray(self.time_grid, dtype=FLOAT_DTYPE)
                if existing_time_grid.shape != time_grid.shape or not np.allclose(
                    existing_time_grid,
                    time_grid,
                ):
                    raise ValueError("dt cannot be combined with a different time_grid.")
            object.__setattr__(self, "dt", dt)
            object.__setattr__(self, "sampling_times", sampling_times.astype(FLOAT_DTYPE))
            object.__setattr__(self, "time_grid", time_grid.astype(FLOAT_DTYPE))
        if self.time_grid is not None:
            _validate_time_grid(self.time_grid, "dataset.time_grid")
        if self.sampling_times is not None:
            sampling_times = np.asarray(self.sampling_times, dtype=FLOAT_DTYPE)
            if sampling_times.shape != (observed.shape[1],):
                raise ValueError("sampling_times must have shape (n_timepoints,).")
            if np.any(np.diff(sampling_times) <= 0):
                raise ValueError("sampling_times must be strictly increasing.")
            if self.time_grid is None:
                if sampling_times.size < 2:
                    raise ValueError("dt is required when only one sampling time is provided.")
                dts = np.diff(sampling_times)
                first_edge = sampling_times[0] - 0.5 * dts[0]
                interior_edges = 0.5 * (sampling_times[:-1] + sampling_times[1:])
                final_edge = sampling_times[-1] + 0.5 * dts[-1]
                inferred_grid = np.concatenate(
                    [
                        np.asarray([first_edge], dtype=FLOAT_DTYPE),
                        interior_edges,
                        np.asarray([final_edge], dtype=FLOAT_DTYPE),
                    ]
                )
                object.__setattr__(self, "time_grid", inferred_grid)
                _validate_time_grid(
                    self.time_grid, "dataset.time_grid (inferred from sampling_times)"
                )
        if self.finite_mask is not None:
            mask = np.asarray(self.finite_mask, dtype=bool)
            if mask.shape != observed.shape:
                raise ValueError("finite_mask must have the same shape as observed.")

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
    """Options controlling model structure and numerical fitting choices.

    The most common fields are:

    - `n_states`: number of promoter states.
    - `dt`: optional shared frame spacing for regularly sampled datasets that
      do not provide dataset-specific timing.
    - `ms2_kernel`, `t_rise`, `t_plateau`, `rna_intensity`: MS2 observation model.
    - `pol2_mode`: Pol2 posterior backend, usually `"auto"` or `"transfer"`.
    - `pol2_elbo_mode`: local Pol2 ELBO backend. `"native"` uses the posterior
      backend's own contribution; `"mean_field"` keeps the posterior backend but
      computes a mean-field diagnostic contribution.
    - `transition_rate_scope` and `loading_rate_scope`: `"dataset"`, `"track"`,
      or `"global"` sharing.
    - `driven_transition_indices`: transition indices driven by contact. For
      column-sum-zero generators, use `ordered_transition_index(n_states,
      to_state, from_state)` to avoid orientation mistakes.
    - `contact_drives`: dataset-ordered score arrays or callables for driven
      contact. Array entries are thresholded as `score < rc`; callables may be
      `fn(rc)` or `fn(times, rc)` and should return contact probabilities.
    - `rc_initial`, `rc_bounds`, `rc_candidate_values`: MAP settings for an
      rc-dependent contact drive.

    Priors use shape/rate Gamma parameters for rates. Pinned parameters are set
    on the corresponding node after model construction.
    """

    n_states: int
    dt: np.ndarray | float | None = None
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
    pol2_elbo_mode: Literal["native", "mean_field"] = "native"
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
    contact_drives: tuple[ContactInputSpec, ...] = ()
    driven_rate_initial: np.ndarray | float = np.float32(1.0)
    driven_rate_bounds: tuple[float, float] = (1e-6, 1.0)
    driven_prior_shape: float = 1.0
    driven_prior_rate: float = 0.0
    rc_initial: np.ndarray | float = np.float32(0.5)
    rc_bounds: tuple[float, float] = (1e-6, 1.0)
    rc_candidate_values: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.n_states < 2:
            raise ValueError("n_states must be at least 2.")
        if self.dt is not None:
            if self.time_grid is not None:
                raise ValueError("dt cannot be combined with time_grid.")
            object.__setattr__(self, "dt", _validate_positive_scalar(self.dt, "dt"))
        if self.time_grid is not None:
            _validate_time_grid(self.time_grid, "time_grid")
        if self.pol2_mode not in {"auto", "transfer", "mean_field", "exact", "sampler"}:
            raise ValueError(
                "pol2_mode must be 'auto', 'transfer', 'mean_field', 'exact', or 'sampler'."
            )
        if self.pol2_elbo_mode not in {"native", "mean_field"}:
            raise ValueError("pol2_elbo_mode must be 'native' or 'mean_field'.")
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
            raise ValueError("driven_transition_indices must be valid transition indices.")
        object.__setattr__(self, "driven_transition_indices", driven_indices)
        try:
            contact_drives = tuple(self.contact_drives)
        except TypeError as exc:
            raise ValueError("contact_drives must be a tuple of arrays or callables.") from exc
        if self.driven_transition_indices and not contact_drives:
            raise ValueError(
                "contact_drives must be set when driven_transition_indices is nonempty."
            )
        if not self.driven_transition_indices and contact_drives:
            raise ValueError("contact_drives requires driven_transition_indices.")
        object.__setattr__(self, "contact_drives", contact_drives)
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
                raise ValueError("rc_candidate_values must be a non-empty one-dimensional array.")
            if np.any(candidates <= 0.0):
                raise ValueError("rc_candidate_values must be positive.")
            object.__setattr__(self, "rc_candidate_values", candidates)

    def __str__(self) -> str:
        driven = (
            "none"
            if not self.driven_transition_indices
            else ", ".join(str(index) for index in self.driven_transition_indices)
        )
        return "\n".join(
            [
                "ModelConfig(",
                f"  n_states={self.n_states}, pol2_mode={self.pol2_mode!r}, "
                f"pol2_elbo_mode={self.pol2_elbo_mode!r},",
                f"  ms2_kernel={_kernel_summary(self.ms2_kernel)},",
                f"  transition_rate_scope={self.transition_rate_scope!r}, "
                f"loading_rate_scope={self.loading_rate_scope!r},",
                f"  driven_transition_indices={driven},",
                f"  rc_initial={np.asarray(self.rc_initial).item():.6g}, "
                f"rc_bounds={self.rc_bounds},",
                ")",
            ]
        )


@dataclass
class ViprodyneModel:
    """Top-level variational model.

    Parameters
    ----------
    datasets:
        One or more observed dataset plates. These contain observations,
        timing, and noise only.
    config:
        Model structure, priors, fitting backend choices, and rate sharing.

    Call `fit(...)` or `run_inference(...)` with CAVI keyword arguments such as
    `max_iterations`, `min_iterations`, `tolerance`, `rho`, and `progress`.
    """

    datasets: tuple[MS2Dataset, ...]
    config: ModelConfig
    graph: VariationalGraph = field(default_factory=VariationalGraph, init=False)
    dataset_nodes: dict[str, dict[str, str | list[str]]] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        try:
            self.datasets = tuple(self.datasets)
        except TypeError as exc:
            raise ValueError("datasets must be a tuple or list of MS2Dataset objects.") from exc
        if not self.datasets:
            raise ValueError("at least one dataset is required.")
        self.datasets = _assign_dataset_names(self.datasets)
        self.datasets = _assign_dataset_timing(self.datasets, self.config)
        _validate_rate_prefix_labels(self.datasets)
        if self.config.driven_transition_indices and len(self.config.contact_drives) != len(
            self.datasets
        ):
            raise ValueError(
                "contact_drives must have one entry per dataset, in dataset input order."
            )
        for dataset in self.datasets:
            self._time_grid(dataset)
        self._build_graph()

    def run_schedule(
        self, schedule: list[str] | tuple[str, ...] | None = None, rho: float = 1.0
    ) -> None:
        """Run a graph update schedule."""
        self.graph.run_schedule(schedule=schedule, rho=rho)

    def fit_cavi(self, config=None, **kwargs):
        """Run CAVI and return `CAVIResult` diagnostics.

        Pass either a `CAVIConfig` via `config` or CAVI keyword arguments, for
        example `max_iterations=200`, `tolerance=1e-3`, `rho=0.75`, or
        `progress=True`.
        """
        from viprodyne.fit import run_cavi

        config = self._cavi_config(config, kwargs)
        return run_cavi(self, config=config)

    def run_inference(
        self,
        config=None,
        *,
        posterior_times=None,
        state_times=None,
        loading_times=None,
        **kwargs,
    ) -> ModelInferenceResult:
        """Run CAVI and return structured posterior outputs.

        `config` may be a `CAVIConfig`. If omitted, keyword arguments are passed
        to `CAVIConfig`, so `model.run_inference(tolerance=1e-3,
        max_iterations=100, progress=True)` is equivalent to constructing the
        config object explicitly.
        """
        config = self._cavi_config(config, kwargs)
        from viprodyne.fit import run_cavi

        cavi_result = run_cavi(self, config=config)
        return self.inference_result(
            cavi=cavi_result,
            include_elbo_terms=bool(config.compute_elbo),
            posterior_times=posterior_times,
            state_times=state_times,
            loading_times=loading_times,
        )

    def fit(
        self,
        config=None,
        *,
        posterior_times=None,
        state_times=None,
        loading_times=None,
        **kwargs,
    ) -> ModelInferenceResult:
        """Alias for `run_inference`.

        This is the shortest public fitting entry point. Use CAVI keyword
        arguments here directly, for example `model.fit(tolerance=1e-3,
        max_iterations=200, progress=True)`.
        """
        return self.run_inference(
            config=config,
            posterior_times=posterior_times,
            state_times=state_times,
            loading_times=loading_times,
            **kwargs,
        )

    def inference_result(
        self,
        cavi=None,
        include_elbo_terms: bool = True,
        posterior_times=None,
        state_times=None,
        loading_times=None,
    ) -> ModelInferenceResult:
        """Collect current graph moments into structured inference outputs."""
        return ModelInferenceResult(
            cavi=cavi,
            datasets={
                dataset.name: self.dataset_result(
                    dataset.name,
                    posterior_times=posterior_times,
                    state_times=state_times,
                    loading_times=loading_times,
                )
                for dataset in self.datasets
            },
            elbo_terms=self.compute_elbo_terms() if include_elbo_terms else None,
        )

    def dataset_result(
        self,
        dataset_name: str,
        *,
        posterior_times=None,
        state_times=None,
        loading_times=None,
    ) -> DatasetInferenceResult:
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
            self.graph.moments.get(str(polymerase_name)) if polymerase_name is not None else {}
        )
        contact_name = nodes["contact_drive"]
        contact = self.graph.moments.get(str(contact_name)) if contact_name is not None else {}
        polymerase_node = (
            self.graph.nodes[str(polymerase_name)] if polymerase_name is not None else None
        )
        sampling_times = None
        if polymerase_node is not None:
            sampling_times = np.asarray(
                polymerase_node.sampling_times,
                dtype=FLOAT_DTYPE,
            ).copy()
        elif dataset.sampling_times is not None:
            sampling_times = np.asarray(dataset.sampling_times, dtype=FLOAT_DTYPE).copy()
        state_times = _resolve_time_spec(
            state_times if state_times is not None else posterior_times,
            dataset.name,
            "state_times",
        )
        loading_times = _resolve_time_spec(
            loading_times if loading_times is not None else posterior_times,
            dataset.name,
            "loading_times",
        )
        state_posterior, state_posterior_times = _state_posterior_at_times(
            self.graph.nodes[str(nodes["promoter"])],
            promoter,
            state_times,
        )
        loading_posterior, loading_posterior_times = _loading_posterior_at_times(
            polymerase_node,
            polymerase,
            "load_probabilities",
            loading_times,
        )
        loading_posterior_rate, _ = _loading_posterior_at_times(
            polymerase_node,
            polymerase,
            "posterior_rate",
            loading_times,
        )
        loading_mask = _loading_mask_at_times(polymerase_node, polymerase, loading_times)

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
            state_posterior=state_posterior,
            loading_posterior=loading_posterior,
            predicted_signal=_optional_float_moment(polymerase, "predicted_signal"),
            loading_mask=loading_mask,
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
            state_posterior_times=state_posterior_times,
            loading_posterior_times=loading_posterior_times,
            loading_posterior_rate=loading_posterior_rate,
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
        return np.asarray(sum(float(value) for value in terms.values()), dtype=FLOAT_DTYPE)

    def _cavi_config(self, config, kwargs):
        from viprodyne.fit import CAVIConfig

        if config is not None and kwargs:
            raise ValueError("pass either config or keyword arguments, not both.")
        if config is None:
            return CAVIConfig(**kwargs)
        return config

    def _build_graph(self) -> None:
        for dataset_index, dataset in enumerate(self.datasets):
            self.dataset_nodes[dataset.name] = self._add_dataset_plate(
                dataset,
                dataset_index,
            )

    def _add_dataset_plate(
        self,
        dataset: MS2Dataset,
        dataset_index: int,
    ) -> dict[str, str | list[str]]:
        time_grid = self._time_grid(dataset)
        initial_name = f"{dataset.name}:pi"
        initial = InitialStateProb(
            name=initial_name,
            prior_concentration=self._initial_concentration(),
        )
        self.graph.add_node(initial)

        transition_names = self._add_transition_rates(dataset.name)
        contact_drive_name = self._add_contact_drive(dataset, transition_names, dataset_index)
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
            sampler_fine_grid = self._sampler_fine_grid(pol2_observation)
            polymerase_loading_times = (
                sampler_fine_grid
                if pol2_observation.mode == "sampler"
                else pol2_observation.loading_times
            )
            polymerase = PolymeraseLoadings(
                name=polymerase_name,
                observed=dataset.observed,
                design_matrix=pol2_observation.design_matrix,
                noise_std=dataset.noise_std,
                finite_mask=dataset.finite_mask,
                mode=pol2_observation.mode,
                elbo_mode=self.config.pol2_elbo_mode,
                window_weights=pol2_observation.window_weights,
                observation_starts=pol2_observation.observation_starts,
                loading_times=polymerase_loading_times,
                sampling_times=pol2_observation.sampling_times,
                fine_grid=sampler_fine_grid,
                elbo_design_matrix=self._pol2_elbo_design_matrix(
                    dataset,
                    pol2_observation,
                    polymerase_loading_times,
                ),
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
        return self.config.transition_rate_scopes.get(index, self.config.transition_rate_scope)

    def _loading_rate_scope(self, state: int) -> RateScope:
        if self.config.shared_loading_rates:
            return "global"
        if state in self.config.shared_loading_rate_states:
            return "global"
        return self.config.loading_rate_scopes.get(state, self.config.loading_rate_scope)

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
            dataset.n_traces for dataset in self.datasets if dataset.name == dataset_name
        )
        if parameter.shape == ():
            return np.full((n_traces,), parameter, dtype=FLOAT_DTYPE)
        if parameter.shape == (n_traces,):
            return parameter.astype(FLOAT_DTYPE)
        raise ValueError(f"{name} must be scalar or have shape (n_traces,) for track scope.")

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
            return _validate_time_grid(self.config.sampler_fine_grid, "sampler_fine_grid")
        return np.asarray(observation.loading_times, dtype=FLOAT_DTYPE)

    def _pol2_elbo_design_matrix(
        self,
        dataset: MS2Dataset,
        observation: MS2ObservationModel,
        loading_times: np.ndarray,
    ) -> np.ndarray | None:
        if self.config.pol2_elbo_mode != "mean_field":
            return None
        if observation.mode == "mean_field" and observation.design_matrix is not None:
            return np.asarray(observation.design_matrix, dtype=FLOAT_DTYPE)
        kernel = resolve_ms2_kernel(
            self.config.ms2_kernel,
            self.config.t_rise,
            self.config.t_plateau,
            self.config.rna_intensity,
        )
        if kernel is None:
            return None
        if observation.mode == "sampler":
            time_grid = _loading_times_to_time_grid(loading_times, "sampler_fine_grid")
        else:
            time_grid = self._time_grid(dataset)
        return build_ms2_observation_model(
            time_grid=time_grid,
            n_observations=dataset.n_timepoints,
            kernel=kernel,
            sampling_times=dataset.sampling_times,
            mode="mean_field",
            tolerance=self.config.kernel_support_tolerance,
        ).design_matrix

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
        drive_node = contact_drive_name if self._is_driven_transition(transition_index) else None
        return RateEdge(
            n_states=self.config.n_states,
            to_state=to_state,
            from_state=from_state,
            rate_node=transition_name,
            drive_node=drive_node,
            transition_index=transition_index,
        )

    def _add_contact_drive(
        self,
        dataset: MS2Dataset,
        transition_names: list[str],
        dataset_index: int,
    ) -> str | None:
        if not self.config.driven_transition_indices:
            return None
        time_grid = self._time_grid(dataset)
        n_intervals = time_grid.size - 1
        contact_name = f"{dataset.name}:rc"
        try:
            contact_drive = self.config.contact_drives[dataset_index]
        except IndexError as exc:
            raise ValueError(
                "contact_drives must have one entry per dataset, in dataset input order."
            ) from exc

        if callable(contact_drive):
            probability_fn = _wrap_contact_probability_fn(contact_drive)
            candidate_values = self.config.rc_candidate_values
        else:
            contact_score = _validate_contact_array(
                contact_drive,
                n_intervals=n_intervals,
                name="contact score",
                clip=False,
            )

            def threshold_contact_probability(times, rc):
                del times
                return _threshold_contact_score(
                    contact_score,
                    rc,
                    less_than=True,
                )

            probability_fn = threshold_contact_probability
            candidate_values = self.config.rc_candidate_values
            if candidate_values is None:
                candidate_values = _default_rc_candidate_values(
                    contact_score, self.config.rc_bounds
                )

        objective = self._contact_drive_objective(
            probability_fn=probability_fn,
            transition_names=transition_names,
            time_grid=time_grid,
        )
        self.graph.add_node(
            RcNode(
                name=contact_name,
                value=self.config.rc_initial,
                time_grid=time_grid,
                contact_probability_fn=probability_fn,
                bounds=self.config.rc_bounds,
                objective_fn=objective,
                candidate_values=candidate_values,
            )
        )
        return contact_name

    def _contact_drive_objective(
        self,
        *,
        probability_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
        transition_names: list[str],
        time_grid: np.ndarray,
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
                interval_durations = np.asarray(moments["interval_durations"], dtype=FLOAT_DTYPE)
                p_contact = _interval_contact_probability(
                    probability_fn(time_grid, rc_value),
                    n_intervals=interval_durations.size,
                )
                for _, (to_state, from_state), rate_name in driven_edges:
                    if rate_name not in blanket_moments:
                        continue
                    rate = np.asarray(
                        blanket_moments[rate_name]["mean"],
                        dtype=FLOAT_DTYPE,
                    )
                    dt_broadcast = np.broadcast_to(
                        interval_durations,
                        occupancy[..., from_state].shape,
                    ).astype(FLOAT_DTYPE)
                    gamma_from = occupancy[..., from_state] / dt_broadcast
                    gamma_jump = jumps[..., to_state, from_state] / dt_broadcast
                    p_broadcast = np.broadcast_to(p_contact, gamma_from.shape).astype(FLOAT_DTYPE)
                    total += _contact_survival_log_likelihood(
                        log_rate=np.log(np.clip(rate, 1e-20, None)).astype(FLOAT_DTYPE),
                        gamma_jump=gamma_jump,
                        gamma_from=gamma_from,
                        p_contact=p_broadcast,
                        dt=dt_broadcast,
                    )
            return float(total)

        return objective

    def _time_grid(self, dataset: MS2Dataset) -> np.ndarray:
        if dataset.time_grid is not None:
            return _validate_time_grid(dataset.time_grid, f"{dataset.name}.time_grid")
        if self.config.dt is not None:
            dt = _validate_positive_scalar(self.config.dt, "dt")
            return np.arange(dataset.n_timepoints + 1, dtype=FLOAT_DTYPE) * dt
        if self.config.time_grid is not None:
            return _validate_time_grid(self.config.time_grid, "time_grid")
        raise ValueError("each dataset needs dt or sampling_times, or ModelConfig.dt must be set.")


def _validate_time_grid(time_grid: np.ndarray, name: str) -> np.ndarray:
    time_grid = np.asarray(time_grid, dtype=FLOAT_DTYPE)
    if time_grid.ndim != 1 or time_grid.size < 2:
        raise ValueError(f"{name} must be one-dimensional with at least two entries.")
    if np.any(np.diff(time_grid) <= 0):
        raise ValueError(f"{name} must be strictly increasing.")
    return time_grid


def _validate_positive_scalar(value: np.ndarray | float, name: str) -> np.float32:
    scalar = np.asarray(value, dtype=FLOAT_DTYPE)
    if scalar.shape != () or np.any(scalar <= 0.0):
        raise ValueError(f"{name} must be a positive scalar.")
    return np.float32(scalar)


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


def _interval_contact_probability(contact: np.ndarray, *, n_intervals: int) -> np.ndarray:
    return _validate_contact_array(
        contact,
        n_intervals=n_intervals,
        name="contact probability",
    )


def _validate_contact_array(
    values: np.ndarray,
    *,
    n_intervals: int,
    name: str,
    clip: bool = True,
) -> np.ndarray:
    values = np.asarray(values, dtype=FLOAT_DTYPE)
    if values.ndim == 0:
        out = np.full((n_intervals,), values, dtype=FLOAT_DTYPE)
    elif values.shape[-1] == n_intervals + 1:
        out = values[..., 1:]
    elif values.shape[-1] == n_intervals:
        out = values
    else:
        raise ValueError(f"{name} last axis must match intervals or grid points.")
    if clip:
        out = np.clip(out, 0.0, 1.0)
    return np.asarray(out, dtype=FLOAT_DTYPE)


def _wrap_contact_probability_fn(
    probability_fn: Callable | None,
) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    if probability_fn is None:
        raise ValueError("contact drive function must be callable.")

    def wrapped(time_grid: np.ndarray, rc_value: np.ndarray) -> np.ndarray:
        try:
            return np.asarray(probability_fn(time_grid, rc_value), dtype=FLOAT_DTYPE)
        except TypeError as first_error:
            try:
                return np.asarray(probability_fn(rc_value), dtype=FLOAT_DTYPE)
            except TypeError:
                raise first_error

    return wrapped


def _kernel_summary(kernel) -> str:
    if isinstance(kernel, str):
        return repr(kernel)
    if isinstance(kernel, ProximalKernel):
        return "ProximalKernel(...)"
    if kernel is None:
        return "None"
    return getattr(kernel, "__name__", kernel.__class__.__name__)


def _default_rc_candidate_values(
    contact_score: np.ndarray,
    bounds: tuple[float, float],
) -> np.ndarray:
    lo, hi = map(np.float32, bounds)
    values = np.unique(np.ravel(np.asarray(contact_score, dtype=FLOAT_DTYPE)))
    values = values[np.isfinite(values)]
    values = values[(values > lo) & (values < hi)]
    breaks = np.unique(np.concatenate([np.asarray([lo, hi], dtype=FLOAT_DTYPE), values]))
    if breaks.size == 1:
        return breaks.astype(FLOAT_DTYPE)
    midpoints = (breaks[:-1] + breaks[1:]) / np.float32(2.0)
    candidates = np.unique(
        np.concatenate([np.asarray([lo, hi], dtype=FLOAT_DTYPE), midpoints.astype(FLOAT_DTYPE)])
    )
    return candidates[(candidates >= lo) & (candidates <= hi)].astype(FLOAT_DTYPE)


def _contact_survival_log_likelihood(
    *,
    log_rate: np.ndarray,
    gamma_jump: np.ndarray,
    gamma_from: np.ndarray,
    p_contact: np.ndarray,
    dt: float | np.ndarray,
) -> float:
    log_rate = np.asarray(log_rate, dtype=FLOAT_DTYPE)
    gamma_jump = np.asarray(gamma_jump, dtype=FLOAT_DTYPE)
    gamma_from = np.asarray(gamma_from, dtype=FLOAT_DTYPE)
    p_contact = np.asarray(p_contact, dtype=FLOAT_DTYPE)
    dt = np.broadcast_to(np.asarray(dt, dtype=FLOAT_DTYPE), gamma_from.shape)
    if gamma_jump.shape != gamma_from.shape or gamma_from.shape != p_contact.shape:
        raise ValueError("gamma_jump, gamma_from, and p_contact must have matching shapes.")
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
    gamma_jump = np.broadcast_to(gamma_jump, batch_shape + (n_intervals,)).astype(FLOAT_DTYPE)
    gamma_from = np.broadcast_to(gamma_from, batch_shape + (n_intervals,)).astype(FLOAT_DTYPE)
    p_contact = np.broadcast_to(p_contact, batch_shape + (n_intervals,)).astype(FLOAT_DTYPE)
    dt = np.broadcast_to(dt, batch_shape + (n_intervals,)).astype(FLOAT_DTYPE)
    total = 0.0
    for index in np.ndindex(batch_shape):
        stats = ContactSurvivalStats.from_posteriors(
            gamma_jump=gamma_jump[index],
            gamma_from=gamma_from[index],
            p_contact=p_contact[index],
            dt=dt[index],
        )
        total += contact_survival_log_profile(float(log_rate[index]), stats)
    return float(total)


def _resolve_time_spec(spec, dataset_name: str, name: str) -> np.ndarray | None:
    if spec is None:
        return None
    if isinstance(spec, Mapping):
        if dataset_name not in spec:
            return None
        spec = spec[dataset_name]
        if spec is None:
            return None
    return _validate_result_times(spec, name)


def _validate_result_times(times, name: str) -> np.ndarray:
    values = np.asarray(times, dtype=FLOAT_DTYPE)
    if values.ndim == 0:
        values = values[None]
    if values.ndim != 1 or values.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array.")
    if np.any(~np.isfinite(values)):
        raise ValueError(f"{name} must contain only finite values.")
    if np.any(np.diff(values) <= 0):
        raise ValueError(f"{name} must be strictly increasing.")
    return values.astype(FLOAT_DTYPE)


def _validate_times_inside(query: np.ndarray, source: np.ndarray, name: str) -> np.ndarray:
    query = _validate_result_times(query, name)
    source = np.asarray(source, dtype=FLOAT_DTYPE)
    if query[0] < source[0] or query[-1] > source[-1]:
        raise ValueError(f"{name} must lie within the fitted posterior time range.")
    return query


def _state_posterior_at_times(
    promoter_node,
    promoter_moments: dict,
    state_times: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    source_times = np.asarray(promoter_node.time_grid, dtype=FLOAT_DTYPE)
    if state_times is None:
        return (
            np.asarray(promoter_moments["posterior"], dtype=FLOAT_DTYPE).copy(),
            source_times.copy(),
        )
    query = _validate_times_inside(state_times, source_times, "state_times")
    if getattr(promoter_node, "solution", None) is None:
        raise ValueError("state posterior interpolation requires an updated PromoterState node.")
    interpolated = [
        np.asarray(promoter_node.solution.marginal_at(float(time)), dtype=FLOAT_DTYPE)
        for time in query
    ]
    return np.stack(interpolated, axis=1).astype(FLOAT_DTYPE), query.copy()


def _loading_posterior_at_times(
    polymerase_node,
    polymerase_moments: dict,
    key: str,
    loading_times: np.ndarray | None,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    if polymerase_node is None or key not in polymerase_moments:
        return None, None
    values = np.asarray(polymerase_moments[key], dtype=FLOAT_DTYPE)
    source_times = _loading_times_from_node(polymerase_node, values.shape[-1])
    if loading_times is None:
        return values.copy(), source_times.copy()
    query = _validate_times_inside(loading_times, source_times, "loading_times")
    return _resample_last_axis(values, source_times, query), query.copy()


def _loading_mask_at_times(
    polymerase_node,
    polymerase_moments: dict,
    loading_times: np.ndarray | None,
) -> np.ndarray | None:
    if polymerase_node is None or "loading_mask" not in polymerase_moments:
        return None
    values = np.asarray(polymerase_moments["loading_mask"], dtype=bool)
    source_times = _loading_times_from_node(polymerase_node, values.shape[-1])
    if loading_times is None:
        return values.copy()
    query = _validate_times_inside(loading_times, source_times, "loading_times")
    return _resample_bool_last_axis(values, source_times, query)


def _loading_times_from_node(polymerase_node, n_loadings: int) -> np.ndarray:
    if getattr(polymerase_node, "loading_times", None) is not None:
        times = np.asarray(polymerase_node.loading_times, dtype=FLOAT_DTYPE)
    elif getattr(polymerase_node, "fine_grid", None) is not None:
        times = np.asarray(polymerase_node.fine_grid, dtype=FLOAT_DTYPE)
    else:
        times = np.arange(n_loadings, dtype=FLOAT_DTYPE)
    if times.shape != (n_loadings,):
        raise ValueError("loading posterior times do not match the posterior length.")
    return times


def _resample_last_axis(values: np.ndarray, source: np.ndarray, query: np.ndarray) -> np.ndarray:
    flat = values.reshape((-1, values.shape[-1]))
    out = np.stack(
        [np.interp(query, source, row).astype(FLOAT_DTYPE) for row in flat],
        axis=0,
    )
    return out.reshape(values.shape[:-1] + (query.size,)).astype(FLOAT_DTYPE)


def _resample_bool_last_axis(
    values: np.ndarray, source: np.ndarray, query: np.ndarray
) -> np.ndarray:
    indices = np.searchsorted(source, query, side="left")
    indices = np.clip(indices, 0, source.size - 1)
    left = np.clip(indices - 1, 0, source.size - 1)
    use_left = np.abs(query - source[left]) <= np.abs(query - source[indices])
    nearest = np.where(use_left, left, indices)
    return np.take(values, nearest, axis=-1).astype(bool)


def _loading_times_to_time_grid(loading_times: np.ndarray, name: str) -> np.ndarray:
    loading_times = _validate_result_times(loading_times, name)
    if loading_times.size < 2:
        raise ValueError(f"{name} must contain at least two points.")
    spacing = np.diff(loading_times)
    if np.any(spacing <= 0):
        raise ValueError(f"{name} must be strictly increasing.")
    final_edge = loading_times[-1] + spacing[-1]
    return np.concatenate([loading_times, np.asarray([final_edge], dtype=FLOAT_DTYPE)]).astype(
        FLOAT_DTYPE
    )


def _optional_float_moment(moments: dict, key: str) -> np.ndarray | None:
    if key not in moments:
        return None
    return np.asarray(moments[key], dtype=FLOAT_DTYPE).copy()


def _optional_bool_moment(moments: dict, key: str) -> np.ndarray | None:
    if key not in moments:
        return None
    return np.asarray(moments[key], dtype=bool).copy()


def _validate_index_tuple(indices: tuple[int, ...], upper_bound: int, name: str) -> tuple[int, ...]:
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
            raise ValueError(f"{label_type} {label!r} is reserved for global rate nodes.")


def _assign_dataset_names(datasets: tuple[MS2Dataset, ...]) -> tuple[MS2Dataset, ...]:
    if any(not isinstance(dataset, MS2Dataset) for dataset in datasets):
        raise ValueError("datasets must contain only MS2Dataset objects.")
    explicit_names = [dataset.name for dataset in datasets if dataset.name is not None]
    if len(set(explicit_names)) != len(explicit_names):
        raise ValueError("explicit dataset names must be unique.")
    used_names = set(explicit_names)
    named: list[MS2Dataset] = []
    auto_index = 0
    for dataset in datasets:
        if dataset.name is not None:
            named.append(dataset)
            continue
        while True:
            candidate = f"dataset_{auto_index}"
            auto_index += 1
            if candidate not in used_names:
                break
        used_names.add(candidate)
        named.append(replace(dataset, name=candidate))
    return tuple(named)


def _assign_dataset_timing(
    datasets: tuple[MS2Dataset, ...],
    config: ModelConfig,
) -> tuple[MS2Dataset, ...]:
    if config.dt is None:
        return datasets
    timed: list[MS2Dataset] = []
    for dataset in datasets:
        if dataset.dt is None and dataset.sampling_times is None and dataset.time_grid is None:
            timed.append(replace(dataset, dt=config.dt))
        else:
            timed.append(dataset)
    return tuple(timed)


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
