"""Domain-specific variational nodes for MS2 posterior models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal

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
from viprodyne.core.rate_edges import RateEdge
from viprodyne.core.tilted_ctmc import TiltedCTMC, TiltedCTMCSolution
from viprodyne.variational.base import MomentDict, UpdateContext, VariationalNode
from viprodyne.variational.distributions import DirichletNode, GammaNode

FLOAT_DTYPE = np.float32


@dataclass
class InitialStateProb(DirichletNode):
    """Dirichlet node for per-dataset promoter initial-state probabilities."""

    def update(self, context: UpdateContext) -> None:
        counts = _sum_child_stat(context, "initial_state_counts")
        if counts is not None:
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
        counts = _sum_child_stat(context, "loading_counts")
        exposure = _sum_child_stat(context, "loading_exposure")
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

    def elbo_contribution(self) -> float:
        return 0.0 if self.is_pinned else float(self.log_profile_value)

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
        return moments

    def entropy(self) -> float:
        return 0.0

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
            return np.asarray(self.initial_probabilities, dtype=FLOAT_DTYPE)
        try:
            return np.asarray(
                parent_moments[self.initial_probability_node]["mean"],
                dtype=FLOAT_DTYPE,
            )
        except KeyError as exc:
            raise KeyError(
                f"initial_probability_node {self.initial_probability_node!r} is not a parent."
            ) from exc

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
        dt = _constant_interval_duration(self.time_grid)
        stats: dict[str, ContactSurvivalStats] = {}
        for edge in self.rate_edges:
            if edge.rate_node not in self.drive_probabilities:
                continue
            p_contact = self.drive_probabilities[edge.rate_node]
            p_contact = np.broadcast_to(p_contact, occupancy.shape[:2]).astype(FLOAT_DTYPE)
            stats[edge.rate_node] = ContactSurvivalStats.from_posteriors(
                gamma_jump=jumps[..., edge.to_state, edge.from_state] / dt,
                gamma_from=occupancy[..., edge.from_state] / dt,
                p_contact=p_contact,
                dt=dt,
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
    mode: Literal["mean_field", "exact", "transfer"] = "mean_field"
    window_weights: np.ndarray | None = None
    observation_starts: np.ndarray | None = None
    load_probabilities: np.ndarray | None = field(default=None, init=False)
    predicted_signal: np.ndarray | None = field(default=None, init=False)
    objective_value: np.float32 = field(default=np.float32(0.0), init=False)
    entropy_value: np.float32 | None = field(default=None, init=False)
    posterior_probabilities: np.ndarray | None = field(default=None, init=False)
    configurations: np.ndarray | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        VariationalNode.__init__(self, self.name)
        self.observed = np.asarray(self.observed, dtype=FLOAT_DTYPE)
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
            self.finite_mask = np.asarray(self.finite_mask, dtype=bool) & np.isfinite(self.observed)
        if self.window_weights is not None:
            self.window_weights = np.asarray(self.window_weights, dtype=FLOAT_DTYPE)
        if self.observation_starts is not None:
            self.observation_starts = np.asarray(self.observation_starts, dtype=np.int32)
        self.update_from_current_inputs()

    def update(self, context: UpdateContext) -> None:
        parent_moments = context.parent_moments()
        derived_prior = _load_prior_probabilities_from_parents(
            parent_moments,
            self.prior_probabilities.size,
        )
        if derived_prior is not None:
            self.prior_probabilities = derived_prior
        for moments in parent_moments.values():
            if "load_prior_probabilities" in moments:
                self.prior_probabilities = np.asarray(
                    moments["load_prior_probabilities"],
                    dtype=FLOAT_DTYPE,
                )
        self.update_from_current_inputs()

    def update_from_current_inputs(self) -> None:
        if self.mode == "exact":
            self._update_exact()
        elif self.mode == "transfer":
            self._update_transfer()
        elif self.mode == "mean_field":
            self._update_mean_field()
        else:
            raise ValueError("mode must be 'mean_field', 'exact', or 'transfer'.")

    def moments(self) -> MomentDict:
        moments: MomentDict = {
            "elbo": np.asarray(self.objective_value, dtype=FLOAT_DTYPE),
            "local_elbo": np.asarray(self.objective_value, dtype=FLOAT_DTYPE),
        }
        if self.load_probabilities is not None:
            moments["load_probabilities"] = np.asarray(self.load_probabilities, dtype=FLOAT_DTYPE)
        if self.predicted_signal is not None:
            moments["predicted_signal"] = np.asarray(self.predicted_signal, dtype=FLOAT_DTYPE)
        if self.entropy_value is not None:
            moments["entropy"] = np.asarray(self.entropy_value, dtype=FLOAT_DTYPE)
        if self.mode in {"exact", "transfer"}:
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
        if self.mode == "transfer":
            raise NotImplementedError("transfer mode currently exposes marginals but not joint samples.")
        if self.load_probabilities is None:
            raise NotImplementedError("Pol2 loading samples require posterior probabilities.")
        probabilities = np.asarray(self.load_probabilities, dtype=FLOAT_DTYPE)
        sample_size = probabilities.shape if size is None else (size,) + probabilities.shape
        return rng.binomial(1, probabilities, size=sample_size).astype(np.int32)

    def _update_exact(self) -> None:
        if self.design_matrix is None:
            raise ValueError("design_matrix is required for exact Pol2 loading updates.")
        configurations = enumerate_binary_configurations(self.prior_probabilities.size)
        log_z, marginals, _, predicted, posterior_probabilities = exact_bernoulli_posterior(
            jnp.asarray(self.observed),
            jnp.asarray(self.prior_probabilities),
            jnp.asarray(self.design_matrix),
            jnp.asarray(self.noise_std),
            jnp.asarray(self.finite_mask),
            configurations,
        )
        self.objective_value = np.asarray(log_z, dtype=FLOAT_DTYPE)
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
        log_z, marginals, predicted, entropy, _, _ = bernoulli_transfer_posterior(
            jnp.asarray(self.observed),
            jnp.asarray(self.prior_probabilities),
            jnp.asarray(self.window_weights),
            jnp.asarray(self.observation_starts),
            jnp.asarray(self.noise_std),
            jnp.asarray(self.finite_mask),
        )
        self.objective_value = np.asarray(log_z, dtype=FLOAT_DTYPE)
        self.load_probabilities = np.asarray(marginals, dtype=FLOAT_DTYPE)
        self.predicted_signal = np.asarray(predicted, dtype=FLOAT_DTYPE)
        self.posterior_probabilities = None
        self.configurations = None
        self.entropy_value = np.asarray(entropy, dtype=FLOAT_DTYPE)

    def _update_mean_field(self) -> None:
        if self.design_matrix is None:
            raise ValueError("design_matrix is required for mean-field Pol2 loading updates.")
        result = fit_mean_field_bernoulli(
            observed=self.observed,
            prior_probabilities=self.prior_probabilities,
            design_matrix=self.design_matrix,
            noise_std=float(self.noise_std),
            mask=self.finite_mask,
        )
        self.load_probabilities = result.load_probabilities
        self.predicted_signal = result.predicted_signal
        self.objective_value = np.asarray(result.elbo, dtype=FLOAT_DTYPE)
        self.posterior_probabilities = None
        self.configurations = None
        self.entropy_value = np.asarray(
            _bernoulli_entropy(self.load_probabilities),
            dtype=FLOAT_DTYPE,
        )

    def _infer_n_loadings(self) -> int:
        if self.design_matrix is not None:
            return int(self.design_matrix.shape[1])
        if self.window_weights is not None and self.observation_starts is not None:
            window_weights = np.asarray(self.window_weights, dtype=FLOAT_DTYPE)
            observation_starts = np.asarray(self.observation_starts, dtype=np.int32)
            return int(np.max(observation_starts) + window_weights.shape[-1])
        raise ValueError(
            "prior_probabilities can only be omitted when design_matrix or transfer "
            "window_weights/observation_starts define the loading grid."
        )


@dataclass
class RcNode(VariationalNode):
    """MAP-only node for the contact-drive hyperparameter ``rc``."""

    name: str
    value: np.ndarray | float
    time_grid: np.ndarray
    contact_probability_fn: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None
    bounds: tuple[float, float] | None = None
    objective_fn: Callable[[np.ndarray, UpdateContext], float] | None = None
    pinned: bool = False
    xatol: float = 1e-4
    maxiter: int = 80

    def __post_init__(self) -> None:
        VariationalNode.__init__(self, self.name)
        self.value = np.asarray(self.value, dtype=FLOAT_DTYPE)
        self.time_grid = np.asarray(self.time_grid, dtype=FLOAT_DTYPE)
        if np.any(self.value <= 0.0):
            raise ValueError("rc must be positive.")

    def update(self, context: UpdateContext) -> None:
        if self.pinned or self.objective_fn is None or self.bounds is None:
            return
        lo, hi = self.bounds
        if not 0 < lo < hi:
            raise ValueError("bounds must satisfy 0 < lower < upper.")

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

    def moments(self) -> MomentDict:
        value = np.asarray(self.value, dtype=FLOAT_DTYPE)
        moments: MomentDict = {
            "mean": value,
            "expected_log": np.log(value).astype(FLOAT_DTYPE),
            "rc": value,
        }
        if self.contact_probability_fn is not None:
            moments["p_contact"] = np.asarray(
                self.contact_probability_fn(self.time_grid, value),
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


def _load_prior_probabilities_from_parents(
    parent_moments: dict[str, MomentDict],
    n_loadings: int,
) -> np.ndarray | None:
    promoter = next(
        (moments for moments in parent_moments.values() if "interval_state_probabilities" in moments),
        None,
    )
    if promoter is None:
        return None
    loading_rates = _state_loading_rates_from_parent_moments(parent_moments)
    if loading_rates is None:
        return None
    state_probabilities = np.asarray(
        promoter["interval_state_probabilities"],
        dtype=FLOAT_DTYPE,
    )
    interval_durations = np.asarray(promoter["interval_durations"], dtype=FLOAT_DTYPE)
    if state_probabilities.ndim == 3:
        if state_probabilities.shape[0] != 1:
            raise NotImplementedError("batched Pol2 loading priors are not implemented yet.")
        state_probabilities = state_probabilities[0]
    if state_probabilities.shape[-1] != loading_rates.size:
        raise ValueError("number of loading-rate parents must match promoter states.")
    if state_probabilities.shape[0] < n_loadings:
        raise ValueError("promoter interval grid is shorter than the Pol2 loading grid.")
    state_probabilities = state_probabilities[:n_loadings]
    interval_durations = interval_durations[:n_loadings]
    per_state_load_probability = 1.0 - np.exp(
        -interval_durations[:, None] * loading_rates[None, :]
    )
    probabilities = np.sum(
        state_probabilities * per_state_load_probability,
        axis=-1,
        dtype=FLOAT_DTYPE,
    )
    return np.clip(probabilities, 1e-7, 1.0 - 1e-7).astype(FLOAT_DTYPE)


def _state_loading_rates_from_parent_moments(
    parent_moments: dict[str, MomentDict],
) -> np.ndarray | None:
    rates: list[tuple[int, np.ndarray]] = []
    fallback: list[np.ndarray] = []
    for name, moments in parent_moments.items():
        if "mean" not in moments:
            continue
        if "state_index" in moments:
            rates.append(
                (
                    int(np.asarray(moments["state_index"])),
                    np.asarray(moments["mean"], dtype=FLOAT_DTYPE),
                )
            )
        elif name.rsplit(":", 1)[-1].startswith("r"):
            fallback.append(np.asarray(moments["mean"], dtype=FLOAT_DTYPE))
    if rates:
        rates.sort(key=lambda item: item[0])
        values = [value for _, value in rates]
    elif fallback:
        values = fallback
    else:
        return None
    rate_array = np.asarray(values, dtype=FLOAT_DTYPE)
    if rate_array.ndim != 1:
        raise NotImplementedError("plated loading-rate priors are not implemented yet.")
    return rate_array


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
    if base.ndim == 2:
        return (base + np.broadcast_to(user, base.shape)).astype(FLOAT_DTYPE)
    if user.ndim == 2:
        user = user[None, :, :]
    return (base + np.broadcast_to(user, base.shape)).astype(FLOAT_DTYPE)


def _constant_interval_duration(time_grid: np.ndarray) -> float:
    dt = np.diff(np.asarray(time_grid, dtype=FLOAT_DTYPE))
    if not np.allclose(dt, dt[0], rtol=1e-6, atol=1e-7):
        raise ValueError("contact-survival rate updates currently require a uniform time grid.")
    return float(dt[0])


def _sum_child_stat(context: UpdateContext, key: str) -> np.ndarray | None:
    values = [
        np.asarray(moments[key], dtype=FLOAT_DTYPE)
        for moments in context.child_moments().values()
        if key in moments
    ]
    if not values:
        return None
    return np.sum(np.stack(values), axis=0, dtype=FLOAT_DTYPE)


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
    return np.sum(np.stack(values), axis=0, dtype=FLOAT_DTYPE)


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
    return np.sum(np.stack(values), axis=0, dtype=FLOAT_DTYPE)


def _match_parameter_shape(value: np.ndarray, target_shape: tuple[int, ...]) -> np.ndarray:
    value = np.asarray(value, dtype=FLOAT_DTYPE)
    if target_shape == ():
        return np.asarray(np.sum(value, dtype=FLOAT_DTYPE), dtype=FLOAT_DTYPE)
    return np.broadcast_to(value, target_shape).astype(FLOAT_DTYPE)


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
