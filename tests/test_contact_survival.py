import numpy as np
import pytest

from viprodyne.core.contact_survival import (
    ContactSurvivalStats,
    contact_survival_log_profile,
    optimize_contact_survival_rate_map,
)


def test_contact_survival_reduces_to_poisson_process_when_contact_is_one():
    gamma_from = np.array([[0.2, 0.8], [0.6, 0.4]])
    stats = ContactSurvivalStats(
        expected_jumps=3.0,
        gamma_from=gamma_from,
        p_contact=np.ones_like(gamma_from),
        dt=0.5,
    )

    prior_shape = 2.0
    prior_rate = 1.5
    expected_map = (stats.expected_jumps + prior_shape - 1.0) / (
        prior_rate + stats.exposure_if_always_contact
    )
    opt = optimize_contact_survival_rate_map(
        stats,
        rate_bounds=(1e-6, 100.0),
        prior_shape=prior_shape,
        prior_rate=prior_rate,
    )

    assert opt["rate"] == pytest.approx(expected_map, rel=5e-4)
    assert not opt["hit_lower"]
    assert not opt["hit_upper"]


def test_contact_survival_log_profile_matches_manual_formula():
    stats = ContactSurvivalStats(
        expected_jumps=2.0,
        gamma_from=np.array([0.25, 0.75]),
        p_contact=np.array([0.5, 0.2]),
        dt=0.1,
        log_contact_jump=-0.3,
    )
    log_rate = np.log(1.7)
    rate = np.exp(log_rate)
    manual = (2.0 + 1.2 - 1.0) * log_rate - 0.4 * rate - 0.3
    manual += np.sum(stats.gamma_from * np.log1p(-stats.p_contact * (1.0 - np.exp(-rate * stats.dt))))

    assert contact_survival_log_profile(log_rate, stats, prior_shape=1.2, prior_rate=0.4) == pytest.approx(manual)


def test_contact_survival_stats_from_posteriors_scales_jump_density_by_dt():
    gamma_jump = np.array([1.0, 2.0, np.nan])
    gamma_from = np.array([0.5, 0.5, 1.0])
    p_contact = np.array([0.2, 0.4, 0.8])

    stats = ContactSurvivalStats.from_posteriors(gamma_jump, gamma_from, p_contact, dt=0.25)

    assert stats.expected_jumps == pytest.approx(0.75)
    assert stats.log_contact_jump == pytest.approx(0.25 * (np.log(0.2) + 2.0 * np.log(0.4)))


def test_contact_survival_stats_validate_shapes():
    with pytest.raises(ValueError, match="same shape"):
        ContactSurvivalStats(expected_jumps=1.0, gamma_from=np.ones(2), p_contact=np.ones(3), dt=1.0)

