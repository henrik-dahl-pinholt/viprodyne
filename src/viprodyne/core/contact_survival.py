"""Contact-survival objective for driven transition-rate updates."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize_scalar

CONTACT_PROB_FLOOR = 1e-20


@dataclass(frozen=True)
class ContactSurvivalStats:
    """Sufficient statistics for a contact-driven transition-rate profile.

    The driven transition has discrete-step survival probability

        1 - p_contact(t) * (1 - exp(-k * dt)).

    ``expected_jumps`` is the posterior expected number of driven jumps. ``gamma_from``
    is the posterior probability of being in the source state at each interval.
    """

    expected_jumps: float
    gamma_from: np.ndarray
    p_contact: np.ndarray
    dt: float
    log_contact_jump: float = 0.0

    def __post_init__(self) -> None:
        gamma_from = np.asarray(self.gamma_from, dtype=float)
        p_contact = np.asarray(self.p_contact, dtype=float)
        if gamma_from.shape != p_contact.shape:
            raise ValueError("gamma_from and p_contact must have the same shape.")
        if self.dt <= 0:
            raise ValueError("dt must be positive.")
        if self.expected_jumps < 0:
            raise ValueError("expected_jumps must be non-negative.")
        object.__setattr__(self, "gamma_from", gamma_from)
        object.__setattr__(self, "p_contact", np.clip(p_contact, 0.0, 1.0))

    @classmethod
    def from_posteriors(
        cls,
        gamma_jump: np.ndarray,
        gamma_from: np.ndarray,
        p_contact: np.ndarray,
        dt: float,
        contact_prob_floor: float = CONTACT_PROB_FLOOR,
    ) -> "ContactSurvivalStats":
        """Build stats from posterior jump density and source-state occupancy arrays."""
        gamma_jump = np.nan_to_num(np.asarray(gamma_jump, dtype=float), nan=0.0)
        p_contact = np.asarray(p_contact, dtype=float)
        if gamma_jump.shape != p_contact.shape:
            raise ValueError("gamma_jump and p_contact must have the same shape.")
        expected_jumps = float(np.sum(gamma_jump) * dt)
        log_contact_jump = float(
            np.sum(gamma_jump * np.log(np.clip(p_contact, contact_prob_floor, None))) * dt
        )
        return cls(
            expected_jumps=expected_jumps,
            gamma_from=np.nan_to_num(gamma_from, nan=0.0),
            p_contact=p_contact,
            dt=dt,
            log_contact_jump=log_contact_jump,
        )

    @property
    def exposure_if_always_contact(self) -> float:
        """Return sum_t gamma_from(t) * dt, used by analytic p_contact=1 checks."""
        return float(np.sum(self.gamma_from) * self.dt)


def contact_survival_log_profile(
    log_rate: float,
    stats: ContactSurvivalStats | list[ContactSurvivalStats] | tuple[ContactSurvivalStats, ...],
    prior_shape: float = 1.0,
    prior_rate: float = 0.0,
) -> float:
    """Evaluate the unnormalized log profile for a contact-driven rate.

    The profile is over ``k`` but is evaluated at ``log(k)`` for stable bounded
    optimization. No Jacobian term is added; this matches MAP over the rate itself.
    """
    if prior_shape <= 0:
        raise ValueError("prior_shape must be positive.")
    if prior_rate < 0:
        raise ValueError("prior_rate must be non-negative.")
    terms = list(stats) if isinstance(stats, (list, tuple)) else [stats]
    rate = float(np.exp(log_rate))
    expected_jumps = sum(term.expected_jumps for term in terms)
    value = (expected_jumps + prior_shape - 1.0) * float(log_rate) - prior_rate * rate
    value += sum(term.log_contact_jump for term in terms)
    for term in terms:
        survival = _log_contact_survival(rate, term.dt, term.p_contact)
        value += float(np.sum(term.gamma_from * survival))
    return float(value)


def _log_contact_survival(rate: float, dt: float, p_contact: np.ndarray) -> np.ndarray:
    """Compute log(1 - p * (1 - exp(-rate * dt))) without p=1 cancellation."""
    p_contact = np.asarray(p_contact, dtype=float)
    survival = np.empty_like(p_contact, dtype=float)
    always_contact = p_contact >= 1.0
    survival[always_contact] = -float(rate) * float(dt)
    if np.any(~always_contact):
        p = p_contact[~always_contact]
        survival[~always_contact] = np.log1p(-p * (-np.expm1(-float(rate) * float(dt))))
    return survival


def optimize_contact_survival_rate_map(
    stats: ContactSurvivalStats | list[ContactSurvivalStats] | tuple[ContactSurvivalStats, ...],
    rate_bounds: tuple[float, float],
    prior_shape: float = 1.0,
    prior_rate: float = 0.0,
    xatol: float = 1e-4,
    maxiter: int = 80,
) -> dict[str, float | bool]:
    """Optimize a contact-survival MAP rate under finite positive bounds."""
    lo_rate, hi_rate = map(float, rate_bounds)
    if not 0 < lo_rate < hi_rate:
        raise ValueError("rate_bounds must satisfy 0 < lower < upper.")
    lo = float(np.log(lo_rate))
    hi = float(np.log(hi_rate))

    def objective(log_rate: float) -> float:
        return -contact_survival_log_profile(log_rate, stats, prior_shape, prior_rate)

    result = minimize_scalar(
        objective,
        bounds=(lo, hi),
        method="bounded",
        options={"xatol": xatol, "maxiter": int(maxiter)},
    )
    candidates = [(lo, -objective(lo)), (hi, -objective(hi)), (float(result.x), -float(result.fun))]
    best_log_rate, best_value = max(candidates, key=lambda item: item[1])
    return {
        "rate": float(np.exp(best_log_rate)),
        "log_rate": float(best_log_rate),
        "value": float(best_value),
        "hit_lower": bool(np.isclose(best_log_rate, lo, atol=2.0 * xatol)),
        "hit_upper": bool(np.isclose(best_log_rate, hi, atol=2.0 * xatol)),
        "success": bool(result.success),
    }
