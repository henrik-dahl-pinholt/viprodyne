"""Domain-specific variational nodes for MS2-like imaging models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize_scalar

from viprodyne.core.bernoulli_transfer_pol2 import (
    bernoulli_transfer_posterior,
    enumerate_binary_configurations,
    exact_bernoulli_posterior,
)
from viprodyne.core.contact_survival import (
    ContactSurvivalStats,
    optimize_contact_survival_rate_map,
)
from viprodyne.core.mf_pol2_finder import fit_mean_field_bernoulli
from viprodyne.core.pol2_sampler import (
    Pol2SamplerResult,
    ThermodynamicIntegrationResult,
    compute_log_z,
    sample_loadings,
)
from viprodyne.core.rate_edges import RateEdge
from viprodyne.core.tilted_ctmc import TiltedCTMC, TiltedCTMCSolution
from viprodyne.variational.base import MomentDict, UpdateContext, VariationalNode
from viprodyne.variational.distributions import DirichletNode, GammaNode

FLOAT_DTYPE = np.float32


@dataclass(frozen=True)
class _LoadingPriorStats:
    probabilities: np.ndarray
    rate_names: tuple[str, ...]
    rate_means: np.ndarray
    rate_intensity: np.ndarray
    state_probabilities: np.ndarray
    interval_durations: np.ndarray


@dataclass(frozen=True)
class _LoadingRateMoments:
    names: tuple[str, ...]
    means: np.ndarray
    expected_logs: np.ndarray | None = None
    shapes: np.ndarray | None = None
    rates: np.ndarray | None = None
    gamma_mask: np.ndarray | None = None


@dataclass
class InitialStateProb(DirichletNode):
    """Dirichlet node for per-dataset promoter initial-state probabilities."""

    def update(self, context: UpdateContext) -> None:
        counts = _sum_child_stat(context, "initial_state_counts")
        if counts is not None:
            counts = _match_parameter_shape(counts, np.asarray(self.prior_concentration).shape)
            self.set_posterior_from_counts(counts, rho=context.rho)

    def sample_states(self, rng: np.random.Generator | None = None, size=None) -> np.ndarray:
        """Sample initial promoter states from the current categorical mean."""
        rng = np.random.default_rng() if rng is None else rng
        probabilities = np.asarray(self.moments()["mean"], dtype=FLOAT_DTYPE)
        if probabilities.ndim == 1:
            return rng.choice(probabilities.size, size=size, p=probabilities)
        sample_size = () if size is None else ((size,) if isinstance(size, int) else tuple(size))
        out = np.empty(sample_size + probabilities.shape[:-1], dtype=int)
        for index in np.ndindex(probabilities.shape[:-1]):
            out[(...,) + index] = rng.choice(probabilities.shape[-1], size=sample_size, p=probabilities[index])
        return out


@dataclass
class LoadingRate(GammaNode):
    """Gamma node for one Pol2 loading-rate parameter."""

    state_index: int | None = None

    def moments(self) -> MomentDict:
        moments = super().moments()
        if self.state_index is not None:
            moments["state_index"] = np.asarray(self.state_index, dtype=np.int32)
        return moments

    def update(self, context: UpdateContext) -> None:
        counts = _sum_child_loading_stat(context, "loading_counts", self.name, self.state_index)
        exposure = _sum_child_loading_stat(
            context,
            "loading_exposure",
            self.name,
            self.state_index,
        )
        if counts is None or exposure is None:
            return
        counts = _match_parameter_shape(counts, np.asarray(self.prior_shape).shape)
        exposure = _match_parameter_shape(exposure, np.asarray(self.prior_rate).shape)
        self.set_posterior_from_sufficient_statistics(counts, exposure, rho=context.rho)


@dataclass
class TransitionRate(GammaNode):
    """Gamma node for one promoter transition-rate parameter."""

    n_states: int = 2
    to_state: int = 1
    from_state: int = 0

    @property
    def edge(self) -> RateEdge:
        """Return this rate as a promoter transition edge."""
        return RateEdge(
            n_states=self.n_states,
            to_state=self.to_state,
            from_state=self.from_state,
            rate_node=self.name,
        )

    def update(self, context: UpdateContext) -> None:
        counts = _sum_child_transition_counts(context, self.to_state, self.from_state)
        exposure = _sum_child_transition_exposure(context, self.from_state)
        if counts is None or exposure is None:
            return
        counts = _match_parameter_shape(counts, np.asarray(self.prior_shape).shape)
        exposure = _match_parameter_shape(exposure, np.asarray(self.prior_rate).shape)
        self.set_posterior_from_sufficient_statistics(counts, exposure, rho=context.rho)


@dataclass
class DrivenRateMap(VariationalNode):
    """MAP node for a contact-driven transition rate with bounded support."""

    name: str
    initial_rate: np.ndarray | float
    rate_bounds: tuple[float, float]
    prior_shape: float = 1.0
    prior_rate: float = 0.0
    pinned_value: np.ndarray | float | None = None
    xatol: float = 1e-4
    maxiter: int = 80

    def __post_init__(self) -> None:
        VariationalNode.__init__(self, self.name)
        lo_rate, hi_rate = self.rate_bounds
        if not 0 < lo_rate < hi_rate:
            raise ValueError("rate_bounds must satisfy 0 < lower < upper.")
        self.rate = np.asarray(self.initial_rate, dtype=FLOAT_DTYPE)
        self.log_profile_value = np.asarray(0.0, dtype=FLOAT_DTYPE)
        self.optimizer_info: dict[str, float | bool] = {}
        if self.pinned_value is not None:
            self.pin(self.pinned_value)

    @property
    def is_pinned(self) -> bool:
        return self.pinned_value is not None

    def pin(self, value) -> None:
        value = np.asarray(value, dtype=FLOAT_DTYPE)
        if np.any(value <= 0.0):
            raise ValueError("pinned driven rates must be positive.")
        self.pinned_value = value
        self.rate = value

    def unpin(self) -> None:
        self.pinned_value = None

    def update(self, context: UpdateContext) -> None:
        if self.is_pinned:
            return
        stats = _collect_contact_survival_stats(context, rate_node_name=self.name)
        if not stats:
            return
        result = optimize_contact_survival_rate_map(
            stats,
            rate_bounds=self.rate_bounds,
            prior_shape=self.prior_shape,
            prior_rate=self.prior_rate,
            xatol=self.xatol,
            maxiter=self.maxiter,
        )
        self.rate = np.asarray(result["rate"], dtype=FLOAT_DTYPE)
        self.log_profile_value = np.asarray(result["value"], dtype=FLOAT_DTYPE)
        self.optimizer_info = result

    def moments(self) -> MomentDict:
        rate = np.asarray(self.rate, dtype=FLOAT_DTYPE)
        return {
            "mean": rate,
            "expected_log": np.log(rate).astype(FLOAT_DTYPE),
            "map_rate": rate,
            "is_driven": True,
        }

    def entropy(self) -> float:
        return 0.0

    def prior_log_density(self) -> float:
        """Return the MAP rate prior term without path-likelihood statistics."""
        if self.is_pinned:
            return 0.0
        rate = jnp.asarray(self.rate, dtype=jnp.float32)
        prior_shape = jnp.asarray(self.prior_shape, dtype=jnp.float32)
        prior_rate = jnp.asarray(self.prior_rate, dtype=jnp.float32)
        value = (
            (prior_shape - 1.0) * jnp.log(jnp.clip(rate, 1e-20, None))
            - prior_rate * rate
        )
        if float(self.prior_rate) > 0.0:
            value = value + prior_shape * jnp.log(prior_rate) - jax.scipy.special.gammaln(
                prior_shape
            )
        return float(jnp.sum(value))

    def elbo_contribution(self) -> float:
        return self.prior_log_density()

    def sample(self, rng: np.random.Generator | None = None, size=None):
        return _deterministic_sample(np.asarray(self.rate, dtype=FLOAT_DTYPE), size)


@dataclass
class ObservedIntensity(VariationalNode):
    """Deterministic node that emits observed MS2 intensity data and noise."""

    name: str
    observed: np.ndarray
    noise_std: np.ndarray | float
    mask: np.ndarray | None = None

    def __post_init__(self) -> None:
        VariationalNode.__init__(self, self.name)
        self.observed = np.asarray(self.observed, dtype=FLOAT_DTYPE)
        if self.observed.ndim != 2:
            raise ValueError("observed must have shape (n_traces, n_timepoints).")
        self.noise_std = np.asarray(self.noise_std, dtype=FLOAT_DTYPE)
        finite_mask = np.isfinite(self.observed)
        if self.mask is not None:
            mask = np.asarray(self.mask, dtype=bool)
            if mask.shape != self.observed.shape:
                raise ValueError("mask must have the same shape as observed.")
            finite_mask &= mask
        self.finite_mask = finite_mask

    def moments(self) -> MomentDict:
        return {
            "observed": self.observed,
            "noise_std": self.noise_std,
            "finite_mask": self.finite_mask,
        }

    def entropy(self) -> float:
        return 0.0

    def sample(self, rng: np.random.Generator | None = None, size=None):
        return _deterministic_sample(self.observed, size)


@dataclass
class PromoterState(VariationalNode):
    """Tilted-CTMC node for the variational promoter-state path."""

    name: str
    time_grid: np.ndarray
    n_states: int
    rate_edges: tuple[RateEdge, ...]
    initial_probability_node: str | None = None
    initial_probabilities: np.ndarray | None = None
    potentials: np.ndarray | None = None
    solution: TiltedCTMCSolution | None = field(default=None, init=False)
    tilted_generator: np.ndarray | None = field(default=None, init=False)
    tilt_potentials: np.ndarray | None = field(default=None, init=False)
    drive_probabilities: dict[str, np.ndarray] = field(default_factory=dict, init=False)
    initial_log_normalizer: np.ndarray = field(
        default_factory=lambda: np.asarray(0.0, dtype=FLOAT_DTYPE),
        init=False,
    )
    child_tilt_potentials: np.ndarray | None = field(default=None, init=False)
    elbo_value: np.float32 = field(default=np.float32(0.0), init=False)

    def __post_init__(self) -> None:
        VariationalNode.__init__(self, self.name)
        self.time_grid = np.asarray(self.time_grid, dtype=FLOAT_DTYPE)
        self.rate_edges = tuple(self.rate_edges)
        if self.initial_probabilities is None:
            self.initial_probabilities = np.full(self.n_states, 1.0 / self.n_states, dtype=FLOAT_DTYPE)
        else:
            self.initial_probabilities = np.asarray(self.initial_probabilities, dtype=FLOAT_DTYPE)
        if self.potentials is not None:
            self.potentials = np.asarray(self.potentials, dtype=FLOAT_DTYPE)
        for edge in self.rate_edges:
            if edge.n_states != self.n_states:
                raise ValueError("all rate_edges must match n_states.")

    def update(self, context: UpdateContext) -> None:
        parent_moments = context.parent_moments()
        initial = self._initial_probabilities_from_parents(parent_moments)
        generator, potentials = self._generator_and_potentials_from_parent_rates(parent_moments)
        child_potentials = _promoter_loading_child_potentials(
            child_moments=context.child_moments(),
            blanket_moments=context.blanket_moments(),
            time_grid=self.time_grid,
            n_states=self.n_states,
        )
        self.child_tilt_potentials = None if child_potentials is None else np.asarray(
            child_potentials,
            dtype=FLOAT_DTYPE,
        )
        if child_potentials is not None:
            potentials = _add_user_potentials(potentials, child_potentials)
        if self.potentials is not None:
            potentials = _add_user_potentials(potentials, self.potentials)
        self.tilted_generator = np.asarray(generator, dtype=FLOAT_DTYPE)
        self.tilt_potentials = np.asarray(potentials, dtype=FLOAT_DTYPE)
        self.solution = TiltedCTMC(
            generator=generator,
            time_grid=self.time_grid,
            initial_probabilities=initial,
            potentials=potentials,
        ).solve()

    def moments(self) -> MomentDict:
        if self.solution is None:
            initial = np.asarray(self.initial_probabilities, dtype=FLOAT_DTYPE)
            if initial.ndim == 1:
                initial = initial[None, :]
            return {
                "posterior": initial[:, None, :],
                "initial_state_counts": initial,
            }
        posterior = np.asarray(self.solution.posterior, dtype=FLOAT_DTYPE)
        occupancy = np.asarray(self.solution.expected_occupancy(), dtype=FLOAT_DTYPE)
        jumps = np.asarray(self.solution.expected_jumps(), dtype=FLOAT_DTYPE)
        interval_durations = np.diff(self.time_grid).astype(FLOAT_DTYPE)
        interval_state_probabilities = occupancy / interval_durations[None, :, None]
        moments: MomentDict = {
            "posterior": posterior,
            "expected_occupancy": occupancy,
            "interval_durations": interval_durations,
            "interval_state_probabilities": interval_state_probabilities.astype(FLOAT_DTYPE),
            "expected_jumps": jumps,
            "transition_counts": np.sum(jumps, axis=1, dtype=FLOAT_DTYPE),
            "transition_exposure": np.sum(occupancy, axis=1, dtype=FLOAT_DTYPE),
            "initial_state_counts": posterior[:, 0],
            "log_partition": np.asarray(self.solution.log_partition, dtype=FLOAT_DTYPE),
        }
        stats_by_rate = self._contact_survival_stats(occupancy, jumps)
        if stats_by_rate:
            moments["contact_survival_stats_by_rate"] = stats_by_rate
        self.elbo_value = _promoter_path_entropy_contribution(
            solution=self.solution,
            occupancy=occupancy,
            initial_log_normalizer=self.initial_log_normalizer,
            child_potentials=self.child_tilt_potentials,
        )
        moments["elbo"] = np.asarray(self.elbo_value, dtype=FLOAT_DTYPE)
        moments["local_elbo"] = np.asarray(self.elbo_value, dtype=FLOAT_DTYPE)
        return moments

    def entropy(self) -> float:
        return 0.0

    def elbo_contribution(self) -> float:
        return float(self.elbo_value)

    def sample(self, rng: np.random.Generator | None = None, size=None):
        if self.solution is None:
            probabilities = np.asarray(self.initial_probabilities, dtype=FLOAT_DTYPE)
            rng = np.random.default_rng() if rng is None else rng
            return rng.choice(probabilities.size, size=size, p=probabilities)
        n_samples = 1 if size is None else int(size)
        samples = self.solution.sample_grid_paths(n_samples, rng=rng)
        return samples[:, 0] if size is None else samples

    def _initial_probabilities_from_parents(self, parent_moments: dict[str, MomentDict]) -> np.ndarray:
        if self.initial_probability_node is None:
            self.initial_log_normalizer = np.asarray(0.0, dtype=FLOAT_DTYPE)
            return np.asarray(self.initial_probabilities, dtype=FLOAT_DTYPE)
        try:
            moments = parent_moments[self.initial_probability_node]
        except KeyError as exc:
            raise KeyError(
                f"initial_probability_node {self.initial_probability_node!r} is not a parent."
            ) from exc
        if "expected_log" not in moments:
            self.initial_log_normalizer = np.asarray(0.0, dtype=FLOAT_DTYPE)
            return np.asarray(moments["mean"], dtype=FLOAT_DTYPE)
        expected_log = np.asarray(moments["expected_log"], dtype=FLOAT_DTYPE)
        max_log = np.max(expected_log, axis=-1, keepdims=True)
        weights = np.exp(expected_log - max_log).astype(FLOAT_DTYPE)
        normalizer = np.sum(weights, axis=-1, keepdims=True, dtype=FLOAT_DTYPE)
        self.initial_log_normalizer = np.squeeze(
            max_log + np.log(normalizer).astype(FLOAT_DTYPE),
            axis=-1,
        ).astype(FLOAT_DTYPE)
        return (weights / normalizer).astype(FLOAT_DTYPE)

    def _generator_and_potentials_from_parent_rates(
        self,
        parent_moments: dict[str, MomentDict],
    ) -> tuple[np.ndarray, np.ndarray]:
        n_edges = self.n_states * (self.n_states - 1)
        n_intervals = self.time_grid.size - 1
        edge_values = []
        batch_shapes = []
        self.drive_probabilities = {}
        for edge in self.rate_edges:
            try:
                moments = parent_moments[edge.rate_node]
            except KeyError as exc:
                raise KeyError(f"rate node {edge.rate_node!r} is not a parent.") from exc
            mean = np.asarray(moments["mean"], dtype=FLOAT_DTYPE)
            if "expected_log" in moments:
                tilted_rate = np.exp(np.asarray(moments["expected_log"], dtype=FLOAT_DTYPE)).astype(
                    FLOAT_DTYPE
                )
            else:
                tilted_rate = np.asarray(np.clip(mean, 1e-20, None), dtype=FLOAT_DTYPE)
            drive = None
            if edge.drive_node is not None:
                try:
                    drive = _interval_drive_probability(
                        parent_moments[edge.drive_node],
                        n_intervals,
                    )
                except KeyError as exc:
                    raise KeyError(f"drive node {edge.drive_node!r} is not a parent.") from exc
                batch_shapes.append(drive.shape[:-1])
            batch_shapes.extend([mean.shape, tilted_rate.shape])
            edge_values.append((edge, mean, tilted_rate, drive))
        batch_shape = np.broadcast_shapes(*batch_shapes)
        offdiag = np.zeros(batch_shape + (n_intervals, n_edges), dtype=FLOAT_DTYPE)
        potentials = np.zeros(batch_shape + (n_intervals, self.n_states), dtype=FLOAT_DTYPE)
        dt = np.broadcast_to(
            np.diff(self.time_grid).astype(FLOAT_DTYPE),
            batch_shape + (n_intervals,),
        )
        for edge, mean, tilted_rate, drive in edge_values:
            mean_interval = _broadcast_parameter_over_intervals(mean, batch_shape, n_intervals)
            tilted_interval = _broadcast_parameter_over_intervals(
                tilted_rate,
                batch_shape,
                n_intervals,
            )
            if drive is None:
                q_rate = tilted_interval
                effective_exit = mean_interval
            else:
                drive_interval = np.broadcast_to(drive, batch_shape + (n_intervals,)).astype(
                    FLOAT_DTYPE
                )
                q_rate = drive_interval * tilted_interval
                effective_exit = _contact_survival_effective_rate(
                    mean_interval,
                    drive_interval,
                    dt,
                )
                self.drive_probabilities[edge.rate_node] = drive_interval.reshape(
                    (-1, n_intervals)
                )
            offdiag[..., :, edge.transition_index] = q_rate
            potentials[..., :, edge.from_state] += q_rate - effective_exit
        generator = _wrap_interval_generators(offdiag, self.rate_edges, self.n_states)
        if batch_shape == ():
            return generator, potentials
        return (
            generator.reshape((-1, n_intervals, self.n_states, self.n_states)),
            potentials.reshape((-1, n_intervals, self.n_states)),
        )

    def _contact_survival_stats(
        self,
        occupancy: np.ndarray,
        jumps: np.ndarray,
    ) -> dict[str, ContactSurvivalStats]:
        if not self.drive_probabilities:
            return {}
        dt = np.diff(np.asarray(self.time_grid, dtype=FLOAT_DTYPE))
        stats: dict[str, ContactSurvivalStats] = {}
        for edge in self.rate_edges:
            if edge.rate_node not in self.drive_probabilities:
                continue
            p_contact = self.drive_probabilities[edge.rate_node]
            p_contact = np.broadcast_to(p_contact, occupancy.shape[:2]).astype(FLOAT_DTYPE)
            dt_broadcast = np.broadcast_to(dt, occupancy.shape[:2]).astype(FLOAT_DTYPE)
            stats[edge.rate_node] = ContactSurvivalStats.from_posteriors(
                gamma_jump=jumps[..., edge.to_state, edge.from_state] / dt_broadcast,
                gamma_from=occupancy[..., edge.from_state] / dt_broadcast,
                p_contact=p_contact,
                dt=dt_broadcast,
            )
        return stats


@dataclass
class PolymeraseLoadings(VariationalNode):
    """Variational node for Bernoulli Pol2 loading variables."""

    name: str
    observed: np.ndarray
    prior_probabilities: np.ndarray | None = None
    design_matrix: np.ndarray | None = None
    noise_std: np.ndarray | float = 1.0
    finite_mask: np.ndarray | None = None
    mode: Literal["mean_field", "exact", "transfer", "sampler"] = "mean_field"
    window_weights: np.ndarray | None = None
    observation_starts: np.ndarray | None = None
    sampling_times: np.ndarray | None = None
    fine_grid: np.ndarray | None = None
    sampler_rates_on_grid: np.ndarray | None = None
    rise_time: np.ndarray | float = np.float32(1.0)
    plateau_time: np.ndarray | float = np.float32(0.0)
    rna_intensity: np.ndarray | float = np.float32(1.0)
    sampler_seed: int = 0
    sampler_iterations: int = 15_000
    sampler_repeats: int = 100
    sampler_compute_elbo: bool = False
    sampler_elbo_iterations: int = 10_000
    sampler_elbo_steps: int = 10
    sampler_elbo_repeats: int = 20
    load_probabilities: np.ndarray | None = field(default=None, init=False)
    posterior_rate: np.ndarray | None = field(default=None, init=False)
    expected_loading_counts: np.ndarray | None = field(default=None, init=False)
    predicted_signal: np.ndarray | None = field(default=None, init=False)
    objective_value: np.float32 = field(default=np.float32(0.0), init=False)
    entropy_value: np.float32 | None = field(default=None, init=False)
    posterior_probabilities: np.ndarray | None = field(default=None, init=False)
    configurations: np.ndarray | None = field(default=None, init=False)
    sampler_result: Pol2SamplerResult | None = field(default=None, init=False)
    sampler_log_z_result: ThermodynamicIntegrationResult | None = field(default=None, init=False)
    loading_mask: np.ndarray | None = field(default=None, init=False)
    loading_counts_by_rate: dict[str, np.ndarray] = field(default_factory=dict, init=False)
    loading_exposure_by_rate: dict[str, np.ndarray] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        VariationalNode.__init__(self, self.name)
        self.observed = np.asarray(self.observed, dtype=FLOAT_DTYPE)
        if self.observed.ndim != 2:
            raise ValueError("observed must have shape (n_traces, n_timepoints).")
        if self.design_matrix is not None:
            self.design_matrix = np.asarray(self.design_matrix, dtype=FLOAT_DTYPE)
        if self.prior_probabilities is None:
            self.prior_probabilities = np.full(
                self._infer_n_loadings(),
                0.5,
                dtype=FLOAT_DTYPE,
            )
        else:
            self.prior_probabilities = np.asarray(self.prior_probabilities, dtype=FLOAT_DTYPE)
        self.noise_std = np.asarray(self.noise_std, dtype=FLOAT_DTYPE)
        if self.finite_mask is None:
            self.finite_mask = np.isfinite(self.observed)
        else:
            mask = np.asarray(self.finite_mask, dtype=bool)
            if mask.shape != self.observed.shape:
                raise ValueError("finite_mask must have the same shape as observed.")
            self.finite_mask = mask & np.isfinite(self.observed)
        if self.window_weights is not None:
            self.window_weights = np.asarray(self.window_weights, dtype=FLOAT_DTYPE)
        if self.observation_starts is not None:
            self.observation_starts = np.asarray(self.observation_starts, dtype=np.int32)
        if self.sampling_times is not None:
            self.sampling_times = np.asarray(self.sampling_times, dtype=FLOAT_DTYPE)
        if self.fine_grid is not None:
            self.fine_grid = _validate_sampler_fine_grid(self.fine_grid)
        if self.sampler_rates_on_grid is not None:
            self.sampler_rates_on_grid = np.asarray(self.sampler_rates_on_grid, dtype=FLOAT_DTYPE)
        self.rise_time = np.asarray(self.rise_time, dtype=FLOAT_DTYPE)
        self.plateau_time = np.asarray(self.plateau_time, dtype=FLOAT_DTYPE)
        self.rna_intensity = np.asarray(self.rna_intensity, dtype=FLOAT_DTYPE)
        self.loading_mask = _loading_support_mask(
            finite_mask=np.asarray(self.finite_mask, dtype=bool),
            n_loadings=self._infer_n_loadings(),
            design_matrix=self.design_matrix,
            window_weights=self.window_weights,
            observation_starts=self.observation_starts,
            sampling_times=self.sampling_times,
            fine_grid=self.fine_grid,
            rise_time=self.rise_time,
            plateau_time=self.plateau_time,
            rna_intensity=self.rna_intensity,
        )
        self.update_from_current_inputs()

    def update(self, context: UpdateContext) -> None:
        parent_moments = context.parent_moments()
        prior_stats = _load_prior_stats_from_parents(
            parent_moments,
            int(np.asarray(self.prior_probabilities).shape[-1]),
        )
        if prior_stats is not None:
            self.prior_probabilities = prior_stats.probabilities
            if self.mode == "sampler":
                self.sampler_rates_on_grid = _loading_intensity_from_prior_stats(prior_stats)
        for moments in parent_moments.values():
            if "load_prior_probabilities" in moments:
                self.prior_probabilities = np.asarray(
                    moments["load_prior_probabilities"],
                    dtype=FLOAT_DTYPE,
                )
        self.update_from_current_inputs()
        if prior_stats is not None:
            self._set_loading_sufficient_statistics(prior_stats)
        else:
            self.loading_counts_by_rate = {}
            self.loading_exposure_by_rate = {}

    def update_from_current_inputs(self) -> None:
        if self.mode == "exact":
            self._update_exact()
        elif self.mode == "transfer":
            self._update_transfer()
        elif self.mode == "sampler":
            self._update_sampler()
        elif self.mode == "mean_field":
            self._update_mean_field()
        else:
            raise ValueError("mode must be 'mean_field', 'exact', 'transfer', or 'sampler'.")

    def moments(self) -> MomentDict:
        moments: MomentDict = {
            "elbo": np.asarray(self.objective_value, dtype=FLOAT_DTYPE),
            "local_elbo": np.asarray(self.objective_value, dtype=FLOAT_DTYPE),
        }
        if self.load_probabilities is not None:
            moments["load_probabilities"] = np.asarray(self.load_probabilities, dtype=FLOAT_DTYPE)
        if self.posterior_rate is not None:
            moments["posterior_rate"] = np.asarray(self.posterior_rate, dtype=FLOAT_DTYPE)
        if self.expected_loading_counts is not None:
            moments["expected_loading_counts"] = np.asarray(
                self.expected_loading_counts,
                dtype=FLOAT_DTYPE,
            )
        if self.predicted_signal is not None:
            moments["predicted_signal"] = np.asarray(self.predicted_signal, dtype=FLOAT_DTYPE)
        if self.entropy_value is not None:
            moments["entropy"] = np.asarray(self.entropy_value, dtype=FLOAT_DTYPE)
        if self.loading_counts_by_rate:
            moments["loading_counts_by_rate"] = {
                name: np.asarray(value, dtype=FLOAT_DTYPE)
                for name, value in self.loading_counts_by_rate.items()
            }
        if self.loading_exposure_by_rate:
            moments["loading_exposure_by_rate"] = {
                name: np.asarray(value, dtype=FLOAT_DTYPE)
                for name, value in self.loading_exposure_by_rate.items()
            }
        if self.loading_mask is not None:
            moments["loading_mask"] = np.asarray(self.loading_mask, dtype=bool)
        if self.mode in {"exact", "transfer", "sampler"}:
            moments["log_partition"] = np.asarray(self.objective_value, dtype=FLOAT_DTYPE)
        return moments

    def entropy(self) -> float:
        if self.entropy_value is None:
            raise NotImplementedError(
                "Pol2 loading entropy is not available for this mode without posterior "
                "sufficient statistics."
            )
        return float(self.entropy_value)

    def elbo_contribution(self) -> float:
        return float(self.objective_value)

    def sample(self, rng: np.random.Generator | None = None, size=None):
        rng = np.random.default_rng() if rng is None else rng
        if self.posterior_probabilities is not None and self.configurations is not None:
            indices = rng.choice(
                self.posterior_probabilities.size,
                size=size,
                p=self.posterior_probabilities,
            )
            return self.configurations[indices].astype(np.int32)
        if self.mode in {"transfer", "sampler"}:
            raise NotImplementedError(
                f"{self.mode} mode currently exposes marginals but not joint samples."
            )
        if self.load_probabilities is None:
            raise NotImplementedError("Pol2 loading samples require posterior probabilities.")
        probabilities = np.asarray(self.load_probabilities, dtype=FLOAT_DTYPE)
        sample_size = probabilities.shape if size is None else (size,) + probabilities.shape
        return rng.binomial(1, probabilities, size=sample_size).astype(np.int32)

    def _update_exact(self) -> None:
        if self.design_matrix is None:
            raise ValueError("design_matrix is required for exact Pol2 loading updates.")
        n_loadings = int(np.asarray(self.prior_probabilities).shape[-1])
        configurations = enumerate_binary_configurations(n_loadings)
        prior = _batch_loading_prior(self.prior_probabilities, self.observed.shape[0])
        noise = _batch_noise(self.noise_std, self.observed.shape)
        log_z, marginals, _, predicted, posterior_probabilities = jax.vmap(
            exact_bernoulli_posterior,
            in_axes=(0, 0, None, 0, 0, None),
        )(
            jnp.asarray(self.observed),
            jnp.asarray(prior),
            jnp.asarray(self.design_matrix),
            jnp.asarray(noise),
            jnp.asarray(self.finite_mask),
            configurations,
        )
        self.objective_value = np.asarray(np.sum(np.asarray(log_z)), dtype=FLOAT_DTYPE)
        self.load_probabilities = np.asarray(marginals, dtype=FLOAT_DTYPE)
        self.predicted_signal = np.asarray(predicted, dtype=FLOAT_DTYPE)
        self.posterior_probabilities = np.asarray(posterior_probabilities, dtype=FLOAT_DTYPE)
        self.configurations = np.asarray(configurations, dtype=FLOAT_DTYPE)
        self.entropy_value = np.asarray(
            _categorical_entropy(self.posterior_probabilities),
            dtype=FLOAT_DTYPE,
        )

    def _update_transfer(self) -> None:
        if self.window_weights is None or self.observation_starts is None:
            raise ValueError("window_weights and observation_starts are required for transfer mode.")
        prior = _batch_loading_prior(self.prior_probabilities, self.observed.shape[0])
        noise = _batch_noise(self.noise_std, self.observed.shape)
        log_z, marginals, predicted, entropy, _, _ = jax.vmap(
            bernoulli_transfer_posterior,
            in_axes=(0, 0, None, None, 0, 0),
        )(
            jnp.asarray(self.observed),
            jnp.asarray(prior),
            jnp.asarray(self.window_weights),
            jnp.asarray(self.observation_starts),
            jnp.asarray(noise),
            jnp.asarray(self.finite_mask),
        )
        self.objective_value = np.asarray(np.sum(np.asarray(log_z)), dtype=FLOAT_DTYPE)
        self.load_probabilities = np.asarray(marginals, dtype=FLOAT_DTYPE)
        self.predicted_signal = np.asarray(predicted, dtype=FLOAT_DTYPE)
        self.posterior_probabilities = None
        self.configurations = None
        self.entropy_value = np.asarray(np.sum(np.asarray(entropy)), dtype=FLOAT_DTYPE)

    def _update_mean_field(self) -> None:
        if self.design_matrix is None:
            raise ValueError("design_matrix is required for mean-field Pol2 loading updates.")
        prior = _batch_loading_prior(self.prior_probabilities, self.observed.shape[0])
        noise = _batch_noise(self.noise_std, self.observed.shape)
        results = [
            fit_mean_field_bernoulli(
                observed=observed,
                prior_probabilities=prior_trace,
                design_matrix=self.design_matrix,
                noise_std=float(np.ravel(noise_trace)[0]),
                mask=mask,
            )
            for observed, prior_trace, noise_trace, mask in zip(
                self.observed,
                prior,
                noise,
                self.finite_mask,
            )
        ]
        load_probabilities = np.stack(
            [result.load_probabilities for result in results],
            axis=0,
        )
        predicted_signal = np.stack([result.predicted_signal for result in results], axis=0)
        objective_value = np.sum([result.elbo for result in results], dtype=FLOAT_DTYPE)
        self.load_probabilities = np.asarray(load_probabilities, dtype=FLOAT_DTYPE)
        self.predicted_signal = np.asarray(predicted_signal, dtype=FLOAT_DTYPE)
        self.objective_value = np.asarray(objective_value, dtype=FLOAT_DTYPE)
        self.posterior_probabilities = None
        self.configurations = None
        self.entropy_value = np.asarray(
            _bernoulli_entropy(self.load_probabilities),
            dtype=FLOAT_DTYPE,
        )

    def _update_sampler(self) -> None:
        if self.sampling_times is None or self.fine_grid is None:
            raise ValueError("sampling_times and fine_grid are required for sampler mode.")
        if self.sampler_rates_on_grid is None:
            self.posterior_rate = np.zeros(
                (self.observed.shape[0], self._infer_n_loadings()),
                dtype=FLOAT_DTYPE,
            )
            self.expected_loading_counts = None
            self.load_probabilities = None
            self.predicted_signal = np.zeros_like(self.observed, dtype=FLOAT_DTYPE)
            self.objective_value = np.asarray(0.0, dtype=FLOAT_DTYPE)
            self.entropy_value = np.asarray(0.0, dtype=FLOAT_DTYPE)
            return
        result = sample_loadings(
            observed=jnp.asarray(self.observed),
            noise_std=jnp.asarray(self.noise_std),
            rise_time=float(np.asarray(self.rise_time, dtype=FLOAT_DTYPE)),
            plateau_time=float(np.asarray(self.plateau_time, dtype=FLOAT_DTYPE)),
            sampling_times=jnp.asarray(self.sampling_times),
            rates_on_grid=jnp.asarray(self.sampler_rates_on_grid),
            fine_grid=jnp.asarray(self.fine_grid),
            seed=int(self.sampler_seed),
            rna_intensity=jnp.asarray(self.rna_intensity),
            n_iter=int(self.sampler_iterations),
            nrepeat=int(self.sampler_repeats),
        )
        posterior_rate = np.asarray(result.posterior_rate, dtype=FLOAT_DTYPE)
        predicted_signal = np.asarray(result.predicted_signal, dtype=FLOAT_DTYPE)
        grid_dt = np.asarray(self.fine_grid[1] - self.fine_grid[0], dtype=FLOAT_DTYPE)
        self.sampler_result = result
        self.posterior_rate = posterior_rate.astype(FLOAT_DTYPE)
        self.expected_loading_counts = (self.posterior_rate * grid_dt).astype(FLOAT_DTYPE)
        self.load_probabilities = np.clip(
            1.0 - np.exp(-self.expected_loading_counts),
            1e-7,
            1.0 - 1e-7,
        ).astype(FLOAT_DTYPE)
        self.predicted_signal = predicted_signal.astype(FLOAT_DTYPE)
        self.entropy_value = np.asarray(0.0, dtype=FLOAT_DTYPE)
        if self.sampler_compute_elbo:
            log_z_result = compute_log_z(
                observed=jnp.asarray(self.observed),
                noise_std=jnp.asarray(self.noise_std),
                rise_time=float(np.asarray(self.rise_time, dtype=FLOAT_DTYPE)),
                plateau_time=float(np.asarray(self.plateau_time, dtype=FLOAT_DTYPE)),
                sampling_times=jnp.asarray(self.sampling_times),
                rates_on_grid=jnp.asarray(self.sampler_rates_on_grid),
                fine_grid=jnp.asarray(self.fine_grid),
                seed=int(self.sampler_seed),
                rna_intensity=jnp.asarray(self.rna_intensity),
                n_iter=int(self.sampler_elbo_iterations),
                n_steps=int(self.sampler_elbo_steps),
                nrepeat=int(self.sampler_elbo_repeats),
            )
            self.sampler_log_z_result = log_z_result
            self.objective_value = np.asarray(log_z_result.log_z, dtype=FLOAT_DTYPE)
        else:
            self.sampler_log_z_result = None
            self.objective_value = np.asarray(0.0, dtype=FLOAT_DTYPE)

    def _infer_n_loadings(self) -> int:
        if self.design_matrix is not None:
            return int(self.design_matrix.shape[1])
        if self.window_weights is not None and self.observation_starts is not None:
            window_weights = np.asarray(self.window_weights, dtype=FLOAT_DTYPE)
            observation_starts = np.asarray(self.observation_starts, dtype=np.int32)
            return int(np.max(observation_starts) + window_weights.shape[-1])
        if self.fine_grid is not None:
            return int(np.asarray(self.fine_grid).size)
        raise ValueError(
            "prior_probabilities can only be omitted when design_matrix or transfer "
            "window_weights/observation_starts, or sampler fine_grid define the loading grid."
        )

    def _set_loading_sufficient_statistics(self, prior_stats: _LoadingPriorStats) -> None:
        counts_source = (
            self.expected_loading_counts
            if self.expected_loading_counts is not None
            else self.load_probabilities
        )
        if counts_source is None:
            self.loading_counts_by_rate = {}
            self.loading_exposure_by_rate = {}
            return
        load_counts = np.asarray(counts_source, dtype=FLOAT_DTYPE)
        loading_mask = _batch_loading_mask(self.loading_mask, load_counts.shape)
        load_counts = load_counts * loading_mask
        counts_by_state = np.sum(
            load_counts[..., :, None] * prior_stats.state_probabilities,
            axis=-2,
            dtype=FLOAT_DTYPE,
        )
        exposure_by_state = np.sum(
            prior_stats.state_probabilities
            * loading_mask[..., :, None]
            * prior_stats.interval_durations[..., None],
            axis=-2,
            dtype=FLOAT_DTYPE,
        )
        self.loading_counts_by_rate = {
            name: np.asarray(counts_by_state[..., index], dtype=FLOAT_DTYPE)
            for index, name in enumerate(prior_stats.rate_names)
        }
        self.loading_exposure_by_rate = {
            name: np.asarray(exposure_by_state[..., index], dtype=FLOAT_DTYPE)
            for index, name in enumerate(prior_stats.rate_names)
        }


@dataclass
class RcNode(VariationalNode):
    """MAP-only node for the contact-drive hyperparameter ``rc``."""

    name: str
    value: np.ndarray | float
    time_grid: np.ndarray
    contact_probability_fn: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None
    bounds: tuple[float, float] | None = None
    objective_fn: Callable[[np.ndarray, UpdateContext], float] | None = None
    candidate_values: np.ndarray | None = None
    pinned: bool = False
    xatol: float = 1e-4
    maxiter: int = 80
    objective_value: np.float32 = field(default=np.float32(0.0), init=False)
    candidate_objective_values: np.ndarray | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        VariationalNode.__init__(self, self.name)
        self.value = np.asarray(self.value, dtype=FLOAT_DTYPE)
        self.time_grid = np.asarray(self.time_grid, dtype=FLOAT_DTYPE)
        if np.any(self.value <= 0.0):
            raise ValueError("rc must be positive.")
        if self.candidate_values is not None:
            candidates = np.asarray(self.candidate_values, dtype=FLOAT_DTYPE)
            if candidates.ndim != 1 or candidates.size == 0:
                raise ValueError("candidate_values must be a non-empty one-dimensional array.")
            if np.any(candidates <= 0.0):
                raise ValueError("candidate_values must be positive.")
            self.candidate_values = candidates

    def update(self, context: UpdateContext) -> None:
        if self.pinned or self.objective_fn is None or self.bounds is None:
            return
        lo, hi = self.bounds
        if not 0 < lo < hi:
            raise ValueError("bounds must satisfy 0 < lower < upper.")
        if self.candidate_values is not None:
            candidates = self.candidate_values[
                (self.candidate_values >= np.float32(lo))
                & (self.candidate_values <= np.float32(hi))
            ]
            if candidates.size == 0:
                raise ValueError("candidate_values must include at least one value within bounds.")
            values = np.asarray(
                [
                    self.objective_fn(np.asarray(candidate, dtype=FLOAT_DTYPE), context)
                    for candidate in candidates
                ],
                dtype=FLOAT_DTYPE,
            )
            values = np.nan_to_num(values, nan=-np.inf)
            best_value = np.max(values)
            tied = np.flatnonzero(np.isclose(values, best_value, rtol=1e-6, atol=1e-6))
            current = float(np.asarray(self.value, dtype=FLOAT_DTYPE))
            best_index = int(tied[np.argmin(np.abs(candidates[tied] - current))])
            self.value = np.asarray(candidates[best_index], dtype=FLOAT_DTYPE)
            self.objective_value = np.asarray(values[best_index], dtype=FLOAT_DTYPE)
            self.candidate_values = candidates.astype(FLOAT_DTYPE)
            self.candidate_objective_values = values.astype(FLOAT_DTYPE)
            return

        def objective(rc_value):
            return -self.objective_fn(np.asarray(rc_value, dtype=FLOAT_DTYPE), context)

        result = minimize_scalar(
            objective,
            bounds=(lo, hi),
            method="bounded",
            options={"xatol": self.xatol, "maxiter": int(self.maxiter)},
        )
        candidates = [(lo, -objective(lo)), (hi, -objective(hi)), (result.x, -result.fun)]
        best_value, _ = max(candidates, key=lambda item: item[1])
        self.value = np.asarray(best_value, dtype=FLOAT_DTYPE)
        self.objective_value = np.asarray(max(value for _, value in candidates), dtype=FLOAT_DTYPE)
        self.candidate_objective_values = None

    def moments(self) -> MomentDict:
        value = np.asarray(self.value, dtype=FLOAT_DTYPE)
        moments: MomentDict = {
            "mean": value,
            "expected_log": np.log(value).astype(FLOAT_DTYPE),
            "rc": value,
            "map_objective": np.asarray(self.objective_value, dtype=FLOAT_DTYPE),
        }
        if self.contact_probability_fn is not None:
            moments["p_contact"] = np.asarray(
                self.contact_probability_fn(self.time_grid, value),
                dtype=FLOAT_DTYPE,
            )
        if self.candidate_values is not None:
            moments["candidate_values"] = np.asarray(self.candidate_values, dtype=FLOAT_DTYPE)
        if self.candidate_objective_values is not None:
            moments["candidate_objective_values"] = np.asarray(
                self.candidate_objective_values,
                dtype=FLOAT_DTYPE,
            )
        return moments

    def entropy(self) -> float:
        return 0.0

    def sample(self, rng: np.random.Generator | None = None, size=None):
        return _deterministic_sample(np.asarray(self.value, dtype=FLOAT_DTYPE), size)


def _interval_drive_probability(moments: MomentDict, n_intervals: int) -> np.ndarray:
    for key in ("p_contact", "drive_probability", "probability"):
        if key in moments:
            values = np.asarray(moments[key], dtype=FLOAT_DTYPE)
            break
    else:
        raise KeyError("drive moments must include p_contact, drive_probability, or probability.")
    if values.ndim == 0:
        return np.full((n_intervals,), values, dtype=FLOAT_DTYPE)
    if values.shape[-1] == n_intervals + 1:
        values = values[..., 1:]
    elif values.shape[-1] != n_intervals:
        raise ValueError("drive probability last axis must match intervals or grid points.")
    return np.clip(values, 0.0, 1.0).astype(FLOAT_DTYPE)


def _batch_loading_prior(prior_probabilities: np.ndarray, n_traces: int) -> np.ndarray:
    prior = np.asarray(prior_probabilities, dtype=FLOAT_DTYPE)
    if prior.ndim == 1:
        return np.broadcast_to(prior[None, :], (n_traces, prior.shape[0])).astype(FLOAT_DTYPE)
    if prior.ndim == 2 and prior.shape[0] == n_traces:
        return prior.astype(FLOAT_DTYPE)
    raise ValueError("batched prior_probabilities must have shape (n_traces, n_loadings).")


def _batch_noise(noise_std: np.ndarray, observed_shape: tuple[int, int]) -> np.ndarray:
    noise = np.asarray(noise_std, dtype=FLOAT_DTYPE)
    try:
        return np.broadcast_to(noise, observed_shape).astype(FLOAT_DTYPE)
    except ValueError as exc:
        if noise.shape == (observed_shape[0],):
            return np.broadcast_to(noise[:, None], observed_shape).astype(FLOAT_DTYPE)
        raise ValueError("noise_std must broadcast to observed shape.") from exc


def _batch_loading_mask(loading_mask: np.ndarray | None, target_shape: tuple[int, ...]) -> np.ndarray:
    if loading_mask is None:
        return np.ones(target_shape, dtype=FLOAT_DTYPE)
    mask = np.asarray(loading_mask, dtype=bool)
    return np.broadcast_to(mask, target_shape).astype(FLOAT_DTYPE)


def _loading_support_mask(
    finite_mask: np.ndarray,
    n_loadings: int,
    design_matrix: np.ndarray | None,
    window_weights: np.ndarray | None,
    observation_starts: np.ndarray | None,
    sampling_times: np.ndarray | None,
    fine_grid: np.ndarray | None,
    rise_time: np.ndarray,
    plateau_time: np.ndarray,
    rna_intensity: np.ndarray,
) -> np.ndarray:
    finite = np.asarray(finite_mask, dtype=bool)
    if finite.ndim != 2:
        raise ValueError("finite_mask must have shape (n_traces, n_timepoints).")
    n_traces = finite.shape[0]
    if design_matrix is not None:
        design = np.asarray(design_matrix, dtype=FLOAT_DTYPE)
        support = np.abs(design[:, :n_loadings]) > np.float32(1e-7)
        return (finite.astype(np.int32) @ support.astype(np.int32) > 0).astype(bool)
    if window_weights is not None and observation_starts is not None:
        return _transfer_loading_support_mask(
            finite,
            n_loadings,
            np.asarray(window_weights, dtype=FLOAT_DTYPE),
            np.asarray(observation_starts, dtype=np.int32),
        )
    if sampling_times is not None and fine_grid is not None:
        sample_times = np.asarray(sampling_times, dtype=FLOAT_DTYPE)
        loading_times = np.asarray(fine_grid, dtype=FLOAT_DTYPE)[:n_loadings]
        offsets = sample_times[:, None] - loading_times[None, :]
        support = _proximal_support_from_offsets(
            offsets,
            rise_time=rise_time,
            plateau_time=plateau_time,
            rna_intensity=rna_intensity,
        )
        return (finite.astype(np.int32) @ support.astype(np.int32) > 0).astype(bool)
    return np.ones((n_traces, n_loadings), dtype=bool)


def _transfer_loading_support_mask(
    finite_mask: np.ndarray,
    n_loadings: int,
    window_weights: np.ndarray,
    observation_starts: np.ndarray,
) -> np.ndarray:
    n_traces, n_observations = finite_mask.shape
    weights = np.asarray(window_weights, dtype=FLOAT_DTYPE)
    if weights.ndim == 1:
        weights = np.broadcast_to(weights[None, :], (n_observations, weights.shape[0]))
    if weights.shape[0] != n_observations:
        raise ValueError("window_weights must have one row per observation.")
    starts = np.asarray(observation_starts, dtype=np.int32)
    if starts.shape != (n_observations,):
        raise ValueError("observation_starts must have one entry per observation.")
    support = np.zeros((n_traces, n_loadings), dtype=bool)
    for obs_index in range(n_observations):
        active_offsets = np.flatnonzero(np.abs(weights[obs_index]) > np.float32(1e-7))
        if active_offsets.size == 0:
            continue
        indices = starts[obs_index] + active_offsets
        indices = indices[(0 <= indices) & (indices < n_loadings)]
        for index in indices:
            support[:, int(index)] |= finite_mask[:, obs_index]
    return support


def _proximal_support_from_offsets(
    offsets: np.ndarray,
    rise_time: np.ndarray,
    plateau_time: np.ndarray,
    rna_intensity: np.ndarray,
) -> np.ndarray:
    rise = np.asarray(rise_time, dtype=FLOAT_DTYPE)
    plateau = np.asarray(plateau_time, dtype=FLOAT_DTYPE)
    intensity = np.asarray(rna_intensity, dtype=FLOAT_DTYPE)
    offsets = np.asarray(offsets, dtype=FLOAT_DTYPE)
    if rise.shape == () and plateau.shape == () and intensity.shape == ():
        support_time = rise + plateau
        rising = (offsets >= 0.0) & (offsets < rise)
        plateau_region = (offsets >= rise) & (offsets <= support_time)
        values = np.where(rising, intensity * offsets / rise, np.float32(0.0))
        values = np.where(plateau_region, intensity, values)
        return np.abs(values) > np.float32(1e-7)
    support_time = np.max(rise + plateau).astype(FLOAT_DTYPE)
    return (
        (offsets > 0.0)
        & (offsets <= support_time)
        & bool(np.any(intensity > 0.0))
    )


def _load_prior_stats_from_parents(
    parent_moments: dict[str, MomentDict],
    n_loadings: int,
) -> _LoadingPriorStats | None:
    promoter = next(
        (moments for moments in parent_moments.values() if "interval_state_probabilities" in moments),
        None,
    )
    if promoter is None:
        return None
    loading_rate_moments = _state_loading_rate_moments_from_moments(parent_moments)
    if loading_rate_moments is None:
        return None
    state_probabilities = np.asarray(
        promoter["interval_state_probabilities"],
        dtype=FLOAT_DTYPE,
    )
    interval_durations = np.asarray(promoter["interval_durations"], dtype=FLOAT_DTYPE)
    n_states = state_probabilities.shape[-1]
    if loading_rate_moments.means.shape[-1] != n_states:
        raise ValueError("number of loading-rate parents must match promoter states.")
    if state_probabilities.shape[-2] < n_loadings:
        raise ValueError("promoter interval grid is shorter than the Pol2 loading grid.")
    state_probabilities = state_probabilities[..., :n_loadings, :]
    interval_durations = interval_durations[:n_loadings]
    per_state_log_load_probability = _expected_log_load_probability(
        loading_rate_moments,
        interval_durations,
    )
    per_state_log_no_load_probability = _expected_log_no_load_probability(
        loading_rate_moments,
        interval_durations,
    )
    expected_log_load = np.sum(
        state_probabilities * per_state_log_load_probability,
        axis=-1,
        dtype=FLOAT_DTYPE,
    )
    expected_log_no_load = np.sum(
        state_probabilities * per_state_log_no_load_probability,
        axis=-1,
        dtype=FLOAT_DTYPE,
    )
    expected_rate_logs = loading_rate_moments.expected_logs
    if expected_rate_logs is None:
        expected_rate_logs = np.log(np.clip(loading_rate_moments.means, 1e-20, None))
    rate_intensity = _expected_loading_intensity_from_state_logs(
        state_probabilities,
        expected_rate_logs,
    )
    probabilities = _load_probability_from_log_terms(expected_log_load, expected_log_no_load)
    return _LoadingPriorStats(
        probabilities=np.clip(probabilities, 1e-7, 1.0 - 1e-7).astype(FLOAT_DTYPE),
        rate_names=loading_rate_moments.names,
        rate_means=loading_rate_moments.means.astype(FLOAT_DTYPE),
        rate_intensity=rate_intensity.astype(FLOAT_DTYPE),
        state_probabilities=state_probabilities.astype(FLOAT_DTYPE),
        interval_durations=interval_durations.astype(FLOAT_DTYPE),
    )


def _state_loading_rate_moments_from_moments(
    moment_map: dict[str, MomentDict],
) -> _LoadingRateMoments | None:
    rates: list[
        tuple[int, str, np.ndarray, np.ndarray | None, np.ndarray | None, np.ndarray | None]
    ] = []
    fallback: list[tuple[str, np.ndarray, np.ndarray | None, np.ndarray | None, np.ndarray | None]] = []
    for name, moments in moment_map.items():
        if "mean" not in moments:
            continue
        mean = np.asarray(moments["mean"], dtype=FLOAT_DTYPE)
        expected_log = (
            np.asarray(moments["expected_log"], dtype=FLOAT_DTYPE)
            if "expected_log" in moments
            else None
        )
        shape = np.asarray(moments["shape"], dtype=FLOAT_DTYPE) if "shape" in moments else None
        rate = np.asarray(moments["rate"], dtype=FLOAT_DTYPE) if "rate" in moments else None
        if "state_index" in moments:
            rates.append(
                (
                    int(np.asarray(moments["state_index"])),
                    name,
                    mean,
                    expected_log,
                    shape,
                    rate,
                )
            )
        elif name.rsplit(":", 1)[-1].startswith("r"):
            fallback.append((name, mean, expected_log, shape, rate))
    if rates:
        rates.sort(key=lambda item: item[0])
        names = tuple(name for _, name, _, _, _, _ in rates)
        means = [mean for _, _, mean, _, _, _ in rates]
        expected_logs = [expected_log for _, _, _, expected_log, _, _ in rates]
        shapes = [shape for _, _, _, _, shape, _ in rates]
        gamma_rates = [rate for _, _, _, _, _, rate in rates]
    elif fallback:
        names = tuple(name for name, _, _, _, _ in fallback)
        means = [mean for _, mean, _, _, _ in fallback]
        expected_logs = [expected_log for _, _, expected_log, _, _ in fallback]
        shapes = [shape for _, _, _, shape, _ in fallback]
        gamma_rates = [rate for _, _, _, _, rate in fallback]
    else:
        return None
    mean_array = np.stack(means, axis=-1).astype(FLOAT_DTYPE)
    if mean_array.ndim not in {1, 2}:
        raise NotImplementedError("plated loading-rate priors are not implemented yet.")
    expected_log_array = None
    if any(expected_log is not None for expected_log in expected_logs):
        expected_log_array = np.stack(
            [
                expected_log if expected_log is not None else np.log(np.clip(mean, 1e-20, None))
                for expected_log, mean in zip(expected_logs, means)
            ],
            axis=-1,
        ).astype(FLOAT_DTYPE)
        if expected_log_array.ndim not in {1, 2}:
            raise NotImplementedError("plated loading-rate priors are not implemented yet.")
    has_gamma = tuple(
        shape is not None and rate is not None for shape, rate in zip(shapes, gamma_rates)
    )
    if any(has_gamma):
        shape_array = np.stack(
            [
                shape if shape is not None else np.full_like(mean, np.nan, dtype=FLOAT_DTYPE)
                for shape, mean in zip(shapes, means)
            ],
            axis=-1,
        ).astype(FLOAT_DTYPE)
        rate_array = np.stack(
            [
                rate if rate is not None else np.full_like(mean, np.nan, dtype=FLOAT_DTYPE)
                for rate, mean in zip(gamma_rates, means)
            ],
            axis=-1,
        ).astype(FLOAT_DTYPE)
        if shape_array.ndim not in {1, 2} or rate_array.ndim not in {1, 2}:
            raise NotImplementedError("plated loading-rate priors are not implemented yet.")
        return _LoadingRateMoments(
            names=names,
            means=mean_array,
            expected_logs=expected_log_array,
            shapes=shape_array,
            rates=rate_array,
            gamma_mask=np.asarray(has_gamma, dtype=bool),
        )
    return _LoadingRateMoments(names=names, means=mean_array, expected_logs=expected_log_array)


def _loading_intensity_from_prior_stats(prior_stats: _LoadingPriorStats) -> np.ndarray:
    return np.clip(prior_stats.rate_intensity, 0.0, None).astype(FLOAT_DTYPE)


def _expected_loading_intensity_from_state_logs(
    state_probabilities: np.ndarray,
    expected_rate_logs: np.ndarray,
) -> np.ndarray:
    states = np.asarray(state_probabilities, dtype=FLOAT_DTYPE)
    logs = np.asarray(expected_rate_logs, dtype=FLOAT_DTYPE)
    batch_shape = np.broadcast_shapes(states.shape[:-2], logs.shape[:-1])
    states = np.broadcast_to(states, batch_shape + states.shape[-2:]).astype(FLOAT_DTYPE)
    logs = np.broadcast_to(logs, batch_shape + logs.shape[-1:]).astype(FLOAT_DTYPE)
    expected_log_rate = np.sum(states * logs[..., None, :], axis=-1, dtype=FLOAT_DTYPE)
    return np.exp(expected_log_rate).astype(FLOAT_DTYPE)


def _expected_log_no_load_probability(
    loading_rate_moments: _LoadingRateMoments,
    interval_durations: np.ndarray,
) -> np.ndarray:
    dt = np.asarray(interval_durations, dtype=FLOAT_DTYPE)
    means = np.asarray(loading_rate_moments.means, dtype=FLOAT_DTYPE)
    dt_view = dt.reshape((1,) * (means.ndim - 1) + (dt.size, 1))
    return (-dt_view * means[..., None, :]).astype(FLOAT_DTYPE)


def _load_probability_from_log_terms(
    expected_log_load: np.ndarray,
    expected_log_no_load: np.ndarray,
) -> np.ndarray:
    log_load = np.asarray(expected_log_load, dtype=FLOAT_DTYPE)
    log_no_load = np.asarray(expected_log_no_load, dtype=FLOAT_DTYPE)
    max_log = np.maximum(log_load, log_no_load)
    load_weight = np.exp(log_load - max_log)
    no_load_weight = np.exp(log_no_load - max_log)
    return (load_weight / (load_weight + no_load_weight)).astype(FLOAT_DTYPE)


def _promoter_path_entropy_contribution(
    solution: TiltedCTMCSolution,
    occupancy: np.ndarray,
    initial_log_normalizer: np.ndarray,
    child_potentials: np.ndarray | None,
) -> np.float32:
    log_partition = np.asarray(solution.log_partition, dtype=FLOAT_DTYPE)
    initial_normalizer = np.asarray(initial_log_normalizer, dtype=FLOAT_DTYPE)
    total = np.sum(log_partition + np.broadcast_to(initial_normalizer, log_partition.shape))
    if child_potentials is None:
        return np.asarray(total, dtype=FLOAT_DTYPE)
    child = np.asarray(child_potentials, dtype=FLOAT_DTYPE)
    if child.ndim == 2:
        child = child[None, :, :]
    child = np.broadcast_to(child, np.asarray(occupancy).shape)
    child_expectation = np.sum(np.asarray(occupancy, dtype=FLOAT_DTYPE) * child, dtype=FLOAT_DTYPE)
    return np.asarray(total - child_expectation, dtype=FLOAT_DTYPE)


def _promoter_loading_child_potentials(
    child_moments: dict[str, MomentDict],
    blanket_moments: dict[str, MomentDict],
    time_grid: np.ndarray,
    n_states: int,
) -> np.ndarray | None:
    loading_rate_moments = _state_loading_rate_moments_from_moments(blanket_moments)
    if loading_rate_moments is None:
        return None
    if loading_rate_moments.means.shape[-1] != n_states:
        raise ValueError("number of loading-rate blanket nodes must match promoter states.")
    interval_durations = np.diff(np.asarray(time_grid, dtype=FLOAT_DTYPE))
    rate_batch_shape = loading_rate_moments.means.shape[:-1]
    potentials = None
    found = False
    for moments in child_moments.values():
        if "expected_loading_counts" in moments:
            found = True
            counts = np.asarray(moments["expected_loading_counts"], dtype=FLOAT_DTYPE)
            if counts.ndim not in {1, 2}:
                raise NotImplementedError("batched Pol2-to-promoter messages are not implemented yet.")
            if counts.shape[-1] > interval_durations.size:
                raise ValueError("Pol2 loading posterior is longer than the promoter interval grid.")
            n_loadings = counts.shape[-1]
            batch_shape = np.broadcast_shapes(rate_batch_shape, counts.shape[:-1])
            if potentials is None:
                potentials = np.zeros(batch_shape + (interval_durations.size, n_states), dtype=FLOAT_DTYPE)
            dt = interval_durations[:n_loadings]
            expected_logs = loading_rate_moments.expected_logs
            if expected_logs is None:
                expected_logs = np.log(np.clip(loading_rate_moments.means, 1e-20, None))
            expected_logs = np.broadcast_to(expected_logs, batch_shape + (n_states,))
            means = np.broadcast_to(loading_rate_moments.means, batch_shape + (n_states,))
            counts = np.broadcast_to(counts, batch_shape + (n_loadings,))
            loading_mask = _child_loading_mask(moments, batch_shape, n_loadings)
            interval_log_potential = (
                counts[..., :, None] * expected_logs[..., None, :]
                - dt.reshape((1,) * len(batch_shape) + (n_loadings, 1)) * means[..., None, :]
            )
            interval_log_potential *= loading_mask[..., :, None]
            potentials[..., :n_loadings, :] += (
                interval_log_potential
                / dt.reshape((1,) * len(batch_shape) + (n_loadings, 1))
            ).astype(FLOAT_DTYPE)
            continue
        if "load_probabilities" not in moments:
            continue
        found = True
        load_probabilities = np.asarray(moments["load_probabilities"], dtype=FLOAT_DTYPE)
        if load_probabilities.ndim not in {1, 2}:
            raise NotImplementedError("batched Pol2-to-promoter messages are not implemented yet.")
        if load_probabilities.shape[-1] > interval_durations.size:
            raise ValueError("Pol2 loading posterior is longer than the promoter interval grid.")
        n_loadings = load_probabilities.shape[-1]
        batch_shape = np.broadcast_shapes(rate_batch_shape, load_probabilities.shape[:-1])
        if potentials is None:
            potentials = np.zeros(batch_shape + (interval_durations.size, n_states), dtype=FLOAT_DTYPE)
        q_load = np.clip(load_probabilities, 0.0, 1.0)
        q_load = np.broadcast_to(q_load, batch_shape + (n_loadings,))
        loading_mask = _child_loading_mask(moments, batch_shape, n_loadings)
        dt = interval_durations[:n_loadings]
        log_load = _expected_log_load_probability(loading_rate_moments, dt)
        log_load = np.broadcast_to(log_load, batch_shape + (n_loadings, n_states))
        means = np.broadcast_to(loading_rate_moments.means, batch_shape + (n_states,))
        dt_view = dt.reshape((1,) * len(batch_shape) + (n_loadings, 1))
        log_no_load = -dt_view * means[..., None, :]
        interval_log_potential = (
            q_load[..., :, None] * log_load + (1.0 - q_load[..., :, None]) * log_no_load
        )
        interval_log_potential *= loading_mask[..., :, None]
        potentials[..., :n_loadings, :] += (interval_log_potential / dt_view).astype(FLOAT_DTYPE)
    if not found:
        return None
    if potentials is None:
        return None
    return potentials.astype(FLOAT_DTYPE)


def _child_loading_mask(
    moments: MomentDict,
    batch_shape: tuple[int, ...],
    n_loadings: int,
) -> np.ndarray:
    if "loading_mask" not in moments:
        return np.ones(batch_shape + (n_loadings,), dtype=FLOAT_DTYPE)
    mask = np.asarray(moments["loading_mask"], dtype=bool)
    return np.broadcast_to(mask[..., :n_loadings], batch_shape + (n_loadings,)).astype(FLOAT_DTYPE)


def _expected_log_load_probability(
    loading_rate_moments: _LoadingRateMoments,
    interval_durations: np.ndarray,
) -> np.ndarray:
    dt = np.asarray(interval_durations, dtype=FLOAT_DTYPE)
    means = np.asarray(loading_rate_moments.means, dtype=FLOAT_DTYPE)
    dt_view = dt.reshape((1,) * (means.ndim - 1) + (dt.size, 1))
    x = np.maximum(dt_view * means[..., None, :], np.float32(1e-20))
    expected_log = np.log(-np.expm1(-x)).astype(FLOAT_DTYPE)
    if loading_rate_moments.shapes is None or loading_rate_moments.rates is None:
        return expected_log

    shapes = np.asarray(loading_rate_moments.shapes, dtype=FLOAT_DTYPE)
    rates = np.asarray(loading_rate_moments.rates, dtype=FLOAT_DTYPE)
    terms = np.arange(1, 257, dtype=FLOAT_DTYPE)
    scaled = rates[None, ..., None, :] / (
        rates[None, ..., None, :] + terms.reshape((-1,) + (1,) * means.ndim + (1,)) * dt_view
    )
    series = np.sum(
        (scaled**shapes[None, ..., None, :])
        / terms.reshape((-1,) + (1,) * (expected_log.ndim)),
        axis=0,
    )
    gamma_mask = np.asarray(loading_rate_moments.gamma_mask, dtype=bool)
    mask = gamma_mask.reshape((1,) * (expected_log.ndim - 1) + (-1,))
    return np.where(mask, -series, expected_log).astype(FLOAT_DTYPE)


def _validate_sampler_fine_grid(fine_grid: np.ndarray) -> np.ndarray:
    fine_grid = np.asarray(fine_grid, dtype=FLOAT_DTYPE)
    if fine_grid.ndim != 1 or fine_grid.size < 2:
        raise ValueError("fine_grid must be one-dimensional with at least two entries.")
    dt = np.diff(fine_grid)
    if np.any(dt <= 0):
        raise ValueError("fine_grid must be strictly increasing.")
    if not np.allclose(dt, dt[0], rtol=1e-6, atol=1e-7):
        raise ValueError("sampler fine_grid must be uniformly spaced.")
    return fine_grid


def _broadcast_parameter_over_intervals(
    value: np.ndarray,
    batch_shape: tuple[int, ...],
    n_intervals: int,
) -> np.ndarray:
    value = np.asarray(value, dtype=FLOAT_DTYPE)
    return np.broadcast_to(value[..., None], batch_shape + (n_intervals,)).astype(FLOAT_DTYPE)


def _contact_survival_effective_rate(
    rate: np.ndarray,
    p_contact: np.ndarray,
    dt: np.ndarray,
) -> np.ndarray:
    rate = np.asarray(rate, dtype=FLOAT_DTYPE)
    p_contact = np.clip(np.asarray(p_contact, dtype=FLOAT_DTYPE), 0.0, 1.0)
    dt = np.asarray(dt, dtype=FLOAT_DTYPE)
    log_survival = np.where(
        p_contact >= 1.0,
        -rate * dt,
        np.log1p(-p_contact * (-np.expm1(-rate * dt))),
    )
    return (-log_survival / dt).astype(FLOAT_DTYPE)


def _wrap_interval_generators(
    offdiag: np.ndarray,
    rate_edges: tuple[RateEdge, ...],
    n_states: int,
) -> np.ndarray:
    generator = np.zeros(offdiag.shape[:-1] + (n_states, n_states), dtype=FLOAT_DTYPE)
    for edge in rate_edges:
        generator[..., edge.to_state, edge.from_state] = offdiag[..., edge.transition_index]
    exit_rates = np.sum(generator, axis=-2, dtype=FLOAT_DTYPE)
    for state in range(n_states):
        generator[..., state, state] = -exit_rates[..., state]
    return generator


def _add_user_potentials(base: np.ndarray, user_potentials: np.ndarray) -> np.ndarray:
    base = np.asarray(base, dtype=FLOAT_DTYPE)
    user = np.asarray(user_potentials, dtype=FLOAT_DTYPE)
    base_was_unbatched = base.ndim == 2
    user_was_unbatched = user.ndim == 2
    if base_was_unbatched:
        base = base[None, :, :]
    if user.ndim == 2:
        user = user[None, :, :]
    batch_shape = np.broadcast_shapes(base.shape[:-2], user.shape[:-2])
    target_shape = batch_shape + base.shape[-2:]
    result = np.broadcast_to(base, target_shape) + np.broadcast_to(user, target_shape)
    if base_was_unbatched and user_was_unbatched:
        return result[0].astype(FLOAT_DTYPE)
    return result.astype(FLOAT_DTYPE)


def _sum_child_stat(context: UpdateContext, key: str) -> np.ndarray | None:
    values = [
        np.asarray(moments[key], dtype=FLOAT_DTYPE)
        for moments in context.child_moments().values()
        if key in moments
    ]
    if not values:
        return None
    return _sum_stat_values(values)


def _sum_child_loading_stat(
    context: UpdateContext,
    key: str,
    rate_node_name: str,
    state_index: int | None,
) -> np.ndarray | None:
    keyed_name = f"{key}_by_rate"
    values = []
    for moments in context.child_moments().values():
        if keyed_name in moments:
            keyed = moments[keyed_name]
            if isinstance(keyed, dict) and rate_node_name in keyed:
                values.append(np.asarray(keyed[rate_node_name], dtype=FLOAT_DTYPE))
    if values:
        return _sum_stat_values(values)
    value = _sum_child_stat(context, key)
    if value is None or state_index is None:
        return value
    value = np.asarray(value, dtype=FLOAT_DTYPE)
    if value.ndim > 0 and value.shape[0] > state_index:
        return np.asarray(value[state_index], dtype=FLOAT_DTYPE)
    return value


def _sum_child_transition_counts(
    context: UpdateContext,
    to_state: int,
    from_state: int,
) -> np.ndarray | None:
    values = []
    for moments in context.child_moments().values():
        if "transition_counts" in moments:
            counts = np.asarray(moments["transition_counts"], dtype=FLOAT_DTYPE)
            values.append(counts[..., to_state, from_state])
        elif "expected_jumps" in moments:
            jumps = np.asarray(moments["expected_jumps"], dtype=FLOAT_DTYPE)
            values.append(np.sum(jumps[..., to_state, from_state], axis=-1, dtype=FLOAT_DTYPE))
    if not values:
        return None
    return _sum_stat_values(values)


def _sum_child_transition_exposure(context: UpdateContext, from_state: int) -> np.ndarray | None:
    values = []
    for moments in context.child_moments().values():
        if "transition_exposure" in moments:
            exposure = np.asarray(moments["transition_exposure"], dtype=FLOAT_DTYPE)
            values.append(exposure[..., from_state])
        elif "expected_occupancy" in moments:
            occupancy = np.asarray(moments["expected_occupancy"], dtype=FLOAT_DTYPE)
            values.append(np.sum(occupancy[..., from_state], axis=-1, dtype=FLOAT_DTYPE))
    if not values:
        return None
    return _sum_stat_values(values)


def _match_parameter_shape(value: np.ndarray, target_shape: tuple[int, ...]) -> np.ndarray:
    value = np.asarray(value, dtype=FLOAT_DTYPE)
    if target_shape == ():
        return np.asarray(np.sum(value, dtype=FLOAT_DTYPE), dtype=FLOAT_DTYPE)
    if value.shape == target_shape:
        return value.astype(FLOAT_DTYPE)
    if value.ndim > len(target_shape) and value.shape[-len(target_shape) :] == target_shape:
        leading_size = int(np.prod(value.shape[: -len(target_shape)]))
        return np.sum(
            value.reshape((leading_size,) + target_shape),
            axis=0,
            dtype=FLOAT_DTYPE,
        )
    return np.broadcast_to(value, target_shape).astype(FLOAT_DTYPE)


def _sum_stat_values(values: list[np.ndarray]) -> np.ndarray:
    try:
        return np.sum(np.stack(values), axis=0, dtype=FLOAT_DTYPE)
    except ValueError:
        return np.asarray(
            sum(float(np.sum(value, dtype=FLOAT_DTYPE)) for value in values),
            dtype=FLOAT_DTYPE,
        )


def _collect_contact_survival_stats(
    context: UpdateContext,
    rate_node_name: str | None = None,
) -> list[ContactSurvivalStats]:
    stats: list[ContactSurvivalStats] = []
    for moments in context.child_moments().values():
        if rate_node_name is not None and "contact_survival_stats_by_rate" in moments:
            keyed = moments["contact_survival_stats_by_rate"]
            if isinstance(keyed, dict) and rate_node_name in keyed:
                value = keyed[rate_node_name]
                if isinstance(value, ContactSurvivalStats):
                    stats.append(value)
                elif isinstance(value, (list, tuple)):
                    stats.extend(value)
        if "contact_survival_stats" in moments:
            value = moments["contact_survival_stats"]
            if isinstance(value, ContactSurvivalStats):
                stats.append(value)
            elif isinstance(value, (list, tuple)):
                stats.extend(value)
        elif {"expected_jumps", "gamma_from", "p_contact", "dt"} <= moments.keys():
            stats.append(
                ContactSurvivalStats(
                    expected_jumps=float(moments["expected_jumps"]),
                    gamma_from=np.asarray(moments["gamma_from"], dtype=FLOAT_DTYPE),
                    p_contact=np.asarray(moments["p_contact"], dtype=FLOAT_DTYPE),
                    dt=float(moments["dt"]),
                    log_contact_jump=float(moments.get("log_contact_jump", 0.0)),
                )
            )
        elif {"gamma_jump", "gamma_from", "p_contact", "dt"} <= moments.keys():
            stats.append(
                ContactSurvivalStats.from_posteriors(
                    gamma_jump=np.asarray(moments["gamma_jump"], dtype=FLOAT_DTYPE),
                    gamma_from=np.asarray(moments["gamma_from"], dtype=FLOAT_DTYPE),
                    p_contact=np.asarray(moments["p_contact"], dtype=FLOAT_DTYPE),
                    dt=float(moments["dt"]),
                )
            )
    return stats


def _deterministic_sample(value: np.ndarray, size=None) -> np.ndarray:
    value = np.asarray(value, dtype=FLOAT_DTYPE)
    if size is None:
        return value.copy()
    size = (size,) if isinstance(size, int) else tuple(size)
    return np.broadcast_to(value, size + value.shape).astype(FLOAT_DTYPE, copy=True)


def _bernoulli_entropy(probabilities: np.ndarray) -> np.float32:
    q = np.clip(np.asarray(probabilities, dtype=FLOAT_DTYPE), 1e-7, 1.0 - 1e-7)
    return np.asarray(-np.sum(q * np.log(q) + (1.0 - q) * np.log1p(-q)), dtype=FLOAT_DTYPE)


def _categorical_entropy(probabilities: np.ndarray) -> np.float32:
    p = np.asarray(probabilities, dtype=FLOAT_DTYPE)
    positive = p > 0.0
    return np.asarray(-np.sum(p[positive] * np.log(p[positive])), dtype=FLOAT_DTYPE)
