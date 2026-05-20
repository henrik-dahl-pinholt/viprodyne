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
from viprodyne.core.rate_edges import RateEdge, wrap_column_generator
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
        stats = _collect_contact_survival_stats(context)
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
        generator = self._generator_from_parent_rates(parent_moments)
        self.solution = TiltedCTMC(
            generator=generator,
            time_grid=self.time_grid,
            initial_probabilities=initial,
            potentials=self.potentials,
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
        return {
            "posterior": posterior,
            "expected_occupancy": occupancy,
            "expected_jumps": jumps,
            "transition_counts": np.sum(jumps, axis=1, dtype=FLOAT_DTYPE),
            "transition_exposure": np.sum(occupancy, axis=1, dtype=FLOAT_DTYPE),
            "initial_state_counts": posterior[:, 0],
            "log_partition": np.asarray(self.solution.log_partition, dtype=FLOAT_DTYPE),
        }

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

    def _generator_from_parent_rates(self, parent_moments: dict[str, MomentDict]) -> np.ndarray:
        n_edges = self.n_states * (self.n_states - 1)
        rate_values = []
        for edge in self.rate_edges:
            try:
                rate_values.append(np.asarray(parent_moments[edge.rate_node]["mean"], dtype=FLOAT_DTYPE))
            except KeyError as exc:
                raise KeyError(f"rate node {edge.rate_node!r} is not a parent.") from exc
        batch_shape = np.broadcast_shapes(*(value.shape for value in rate_values))
        offdiag = np.zeros(batch_shape + (n_edges,), dtype=FLOAT_DTYPE)
        for edge, value in zip(self.rate_edges, rate_values, strict=True):
            offdiag[..., edge.transition_index] = np.broadcast_to(value, batch_shape)
        if batch_shape == ():
            return wrap_column_generator(offdiag, self.n_states)
        flat = offdiag.reshape((-1, n_edges))
        generators = np.stack(
            [wrap_column_generator(row, self.n_states) for row in flat],
            axis=0,
        ).reshape(batch_shape + (self.n_states, self.n_states))
        return generators.reshape((-1, 1, self.n_states, self.n_states))


@dataclass
class PolymeraseLoadings(VariationalNode):
    """Variational node for Bernoulli Pol2 loading variables."""

    name: str
    observed: np.ndarray
    prior_probabilities: np.ndarray
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
        self.prior_probabilities = np.asarray(self.prior_probabilities, dtype=FLOAT_DTYPE)
        if self.design_matrix is not None:
            self.design_matrix = np.asarray(self.design_matrix, dtype=FLOAT_DTYPE)
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


def _collect_contact_survival_stats(context: UpdateContext) -> list[ContactSurvivalStats]:
    stats: list[ContactSurvivalStats] = []
    for moments in context.child_moments().values():
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
