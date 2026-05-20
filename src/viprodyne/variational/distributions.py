"""Reusable variational distributions for model parameters."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
from jax.scipy.special import digamma, gammaln

from viprodyne.variational.base import MomentDict, VariationalNode

FLOAT_DTYPE = np.float32


def _deterministic_sample(value: np.ndarray, size=None) -> np.ndarray:
    value = np.asarray(value, dtype=FLOAT_DTYPE)
    if size is None:
        return value.copy()
    size = (size,) if isinstance(size, int) else tuple(size)
    return np.broadcast_to(value, size + value.shape).astype(FLOAT_DTYPE, copy=True)


def _as_positive_array(value, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=FLOAT_DTYPE)
    if np.any(array <= 0):
        raise ValueError(f"{name} must be strictly positive.")
    return array


def _broadcast_pair(left, right, left_name: str, right_name: str) -> tuple[np.ndarray, np.ndarray]:
    left = _as_positive_array(left, left_name)
    right = _as_positive_array(right, right_name)
    try:
        return np.broadcast_arrays(left, right)
    except ValueError as exc:
        raise ValueError(f"{left_name} and {right_name} are not broadcast-compatible.") from exc


@dataclass
class GammaNode(VariationalNode):
    """Gamma variational node using shape/rate parameterization."""

    name: str
    prior_shape: np.ndarray | float
    prior_rate: np.ndarray | float
    shape: np.ndarray | float | None = None
    rate: np.ndarray | float | None = None
    pinned_value: np.ndarray | float | None = None

    def __post_init__(self) -> None:
        VariationalNode.__init__(self, self.name)
        prior_shape, prior_rate = _broadcast_pair(
            self.prior_shape, self.prior_rate, "prior_shape", "prior_rate"
        )
        self.prior_shape = prior_shape.astype(FLOAT_DTYPE)
        self.prior_rate = prior_rate.astype(FLOAT_DTYPE)
        if self.shape is None:
            self.shape = self.prior_shape.copy()
        if self.rate is None:
            self.rate = self.prior_rate.copy()
        self.shape, self.rate = _broadcast_pair(self.shape, self.rate, "shape", "rate")
        if self.shape.shape != self.prior_shape.shape:
            self.shape = np.broadcast_to(self.shape, self.prior_shape.shape).astype(FLOAT_DTYPE)
            self.rate = np.broadcast_to(self.rate, self.prior_shape.shape).astype(FLOAT_DTYPE)
        if self.pinned_value is not None:
            self.pin(self.pinned_value)

    @property
    def is_pinned(self) -> bool:
        return self.pinned_value is not None

    def pin(self, value) -> None:
        pinned = _as_positive_array(value, "pinned_value")
        self.pinned_value = np.broadcast_to(pinned, self.prior_shape.shape).astype(FLOAT_DTYPE)

    def unpin(self) -> None:
        self.pinned_value = None

    def set_posterior_from_sufficient_statistics(self, counts, exposure, rho: float = 1.0) -> None:
        """Apply the conjugate update for Poisson-process counts and exposure."""
        if self.is_pinned:
            return
        if not 0 < rho <= 1:
            raise ValueError("rho must be in (0, 1].")
        counts = np.asarray(counts, dtype=FLOAT_DTYPE)
        exposure = np.asarray(exposure, dtype=FLOAT_DTYPE)
        if np.any(counts < 0) or np.any(exposure < 0):
            raise ValueError("counts and exposure must be non-negative.")
        target_shape = self.prior_shape + np.broadcast_to(counts, self.prior_shape.shape)
        target_rate = self.prior_rate + np.broadcast_to(exposure, self.prior_shape.shape)
        self.shape = ((1.0 - rho) * self.shape + rho * target_shape).astype(FLOAT_DTYPE)
        self.rate = ((1.0 - rho) * self.rate + rho * target_rate).astype(FLOAT_DTYPE)

    def moments(self) -> MomentDict:
        if self.is_pinned:
            value = np.asarray(self.pinned_value, dtype=FLOAT_DTYPE)
            return {"mean": value, "expected_log": np.log(value).astype(FLOAT_DTYPE)}
        shape = np.asarray(self.shape, dtype=FLOAT_DTYPE)
        rate = np.asarray(self.rate, dtype=FLOAT_DTYPE)
        expected_log = np.asarray(
            digamma(jnp.asarray(shape)) - jnp.log(jnp.asarray(rate)),
            dtype=FLOAT_DTYPE,
        )
        return {
            "mean": (shape / rate).astype(FLOAT_DTYPE),
            "expected_log": expected_log,
            "shape": shape,
            "rate": rate,
        }

    def entropy(self) -> float:
        if self.is_pinned:
            return 0.0
        shape = jnp.asarray(self.shape, dtype=jnp.float32)
        rate = jnp.asarray(self.rate, dtype=jnp.float32)
        entropy = shape - jnp.log(rate) + gammaln(shape) + (1.0 - shape) * digamma(shape)
        return float(jnp.sum(entropy))

    def expected_log_prior(self) -> float:
        if self.is_pinned:
            return 0.0
        moments = self.moments()
        prior_shape = jnp.asarray(self.prior_shape, dtype=jnp.float32)
        prior_rate = jnp.asarray(self.prior_rate, dtype=jnp.float32)
        value = (
            prior_shape * jnp.log(prior_rate)
            - gammaln(prior_shape)
            + (prior_shape - 1.0) * jnp.asarray(moments["expected_log"], dtype=jnp.float32)
            - prior_rate * jnp.asarray(moments["mean"], dtype=jnp.float32)
        )
        return float(jnp.sum(value))

    def elbo_contribution(self) -> float:
        return self.expected_log_prior() + self.entropy()

    def sample(self, rng: np.random.Generator | None = None, size=None):
        rng = np.random.default_rng() if rng is None else rng
        if self.is_pinned:
            return _deterministic_sample(np.asarray(self.pinned_value, dtype=FLOAT_DTYPE), size)
        return np.asarray(
            rng.gamma(shape=self.shape, scale=1.0 / self.rate, size=size),
            dtype=FLOAT_DTYPE,
        )

    def sample_prior(self, rng: np.random.Generator | None = None, size=None):
        rng = np.random.default_rng() if rng is None else rng
        if self.is_pinned:
            return _deterministic_sample(np.asarray(self.pinned_value, dtype=FLOAT_DTYPE), size)
        return np.asarray(
            rng.gamma(
                shape=self.prior_shape,
                scale=1.0 / self.prior_rate,
                size=size,
            ),
            dtype=FLOAT_DTYPE,
        )


@dataclass
class DirichletNode(VariationalNode):
    """Dirichlet variational node."""

    name: str
    prior_concentration: np.ndarray
    concentration: np.ndarray | None = None
    pinned_value: np.ndarray | None = None

    def __post_init__(self) -> None:
        VariationalNode.__init__(self, self.name)
        self.prior_concentration = _as_positive_array(
            self.prior_concentration, "prior_concentration"
        )
        if self.prior_concentration.ndim == 0:
            raise ValueError("Dirichlet concentration must have at least one category axis.")
        if self.concentration is None:
            self.concentration = self.prior_concentration.copy()
        self.concentration = _as_positive_array(self.concentration, "concentration")
        if self.concentration.shape != self.prior_concentration.shape:
            raise ValueError("concentration must have the same shape as prior_concentration.")
        if self.pinned_value is not None:
            self.pin(self.pinned_value)

    @property
    def is_pinned(self) -> bool:
        return self.pinned_value is not None

    def pin(self, value) -> None:
        value = np.asarray(value, dtype=FLOAT_DTYPE)
        if value.shape != self.prior_concentration.shape:
            raise ValueError("pinned_value must have the same shape as prior_concentration.")
        if np.any(value < 0):
            raise ValueError("pinned probabilities must be non-negative.")
        total = np.sum(value, axis=-1, keepdims=True)
        if np.any(total <= 0):
            raise ValueError("pinned probabilities must have positive mass.")
        self.pinned_value = (value / total).astype(FLOAT_DTYPE)

    def unpin(self) -> None:
        self.pinned_value = None

    def set_posterior_from_counts(self, counts, rho: float = 1.0) -> None:
        if self.is_pinned:
            return
        if not 0 < rho <= 1:
            raise ValueError("rho must be in (0, 1].")
        counts = np.asarray(counts, dtype=FLOAT_DTYPE)
        if counts.shape != self.prior_concentration.shape:
            raise ValueError("counts must have the same shape as prior_concentration.")
        if np.any(counts < 0):
            raise ValueError("counts must be non-negative.")
        target = self.prior_concentration + counts
        self.concentration = ((1.0 - rho) * self.concentration + rho * target).astype(
            FLOAT_DTYPE
        )

    def moments(self) -> MomentDict:
        if self.is_pinned:
            probs = np.asarray(self.pinned_value, dtype=FLOAT_DTYPE)
            expected_log = np.full_like(probs, -np.inf, dtype=FLOAT_DTYPE)
            positive = probs > 0.0
            expected_log[positive] = np.log(probs[positive]).astype(FLOAT_DTYPE)
            return {"mean": probs, "expected_log": expected_log}
        concentration = np.asarray(self.concentration, dtype=FLOAT_DTYPE)
        total = np.sum(concentration, axis=-1, keepdims=True)
        expected_log = np.asarray(
            digamma(jnp.asarray(concentration)) - digamma(jnp.asarray(total)),
            dtype=FLOAT_DTYPE,
        )
        return {
            "mean": (concentration / total).astype(FLOAT_DTYPE),
            "expected_log": expected_log,
        }

    def entropy(self) -> float:
        if self.is_pinned:
            return 0.0
        concentration = jnp.asarray(self.concentration, dtype=jnp.float32)
        total = jnp.sum(concentration, axis=-1)
        k = concentration.shape[-1]
        log_beta = jnp.sum(gammaln(concentration), axis=-1) - gammaln(total)
        entropy = log_beta + (total - k) * digamma(total)
        entropy -= jnp.sum((concentration - 1.0) * digamma(concentration), axis=-1)
        return float(jnp.sum(entropy))

    def expected_log_prior(self) -> float:
        if self.is_pinned:
            return 0.0
        moments = self.moments()
        prior = jnp.asarray(self.prior_concentration, dtype=jnp.float32)
        prior_total = jnp.sum(prior, axis=-1)
        log_norm = gammaln(prior_total) - jnp.sum(gammaln(prior), axis=-1)
        value = log_norm + jnp.sum(
            (prior - 1.0) * jnp.asarray(moments["expected_log"], dtype=jnp.float32),
            axis=-1,
        )
        return float(jnp.sum(value))

    def elbo_contribution(self) -> float:
        return self.expected_log_prior() + self.entropy()

    def sample(self, rng: np.random.Generator | None = None, size=None):
        rng = np.random.default_rng() if rng is None else rng
        if self.is_pinned:
            return _deterministic_sample(np.asarray(self.pinned_value, dtype=FLOAT_DTYPE), size)
        return _sample_dirichlet(rng, np.asarray(self.concentration, dtype=FLOAT_DTYPE), size=size)

    def sample_prior(self, rng: np.random.Generator | None = None, size=None):
        rng = np.random.default_rng() if rng is None else rng
        if self.is_pinned:
            return _deterministic_sample(np.asarray(self.pinned_value, dtype=FLOAT_DTYPE), size)
        return _sample_dirichlet(
            rng,
            np.asarray(self.prior_concentration, dtype=FLOAT_DTYPE),
            size=size,
        )


@dataclass
class DeltaNode(VariationalNode):
    """Deterministic node for fixed known values or MAP-only parameters."""

    name: str
    value: np.ndarray | float
    log_safe: bool = True

    def __post_init__(self) -> None:
        VariationalNode.__init__(self, self.name)
        self.value = np.asarray(self.value, dtype=FLOAT_DTYPE)

    def set_value(self, value) -> None:
        self.value = np.asarray(value, dtype=FLOAT_DTYPE)

    def moments(self) -> MomentDict:
        moments: MomentDict = {"mean": self.value}
        if self.log_safe and np.all(self.value > 0):
            moments["expected_log"] = np.log(self.value).astype(FLOAT_DTYPE)
        return moments

    def entropy(self) -> float:
        return 0.0

    def sample(self, rng: np.random.Generator | None = None, size=None):
        if size is None:
            return np.asarray(self.value).copy()
        return _deterministic_sample(np.asarray(self.value, dtype=FLOAT_DTYPE), size)


def _sample_dirichlet(
    rng: np.random.Generator,
    concentration: np.ndarray,
    size=None,
) -> np.ndarray:
    """Sample from possibly plated Dirichlet distributions along the last axis."""
    concentration = np.asarray(concentration, dtype=FLOAT_DTYPE)
    if concentration.ndim == 1:
        return rng.dirichlet(concentration, size=size).astype(FLOAT_DTYPE)
    size = () if size is None else ((size,) if isinstance(size, int) else tuple(size))
    gamma = rng.gamma(shape=concentration, scale=1.0, size=size + concentration.shape)
    return (gamma / np.sum(gamma, axis=-1, keepdims=True)).astype(FLOAT_DTYPE)
