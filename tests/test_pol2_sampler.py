import jax
import jax.numpy as jnp
import numpy as np

from viprodyne.core.pol2_sampler import (
    compute_log_z,
    ms2_kernel,
    sample_loadings,
    setup_sampler,
    support_indices,
)


def test_sampler_setup_returns_float32_cdfs_and_integrated_rates():
    time_grid = jnp.asarray([0.0, 0.25, 0.5], dtype=jnp.float32)
    rates = jnp.asarray([[1.0, 2.0, 1.0], [0.5, 0.5, 1.0]], dtype=jnp.float32)

    cdf, integrated = setup_sampler(time_grid, rates)

    assert cdf.dtype == jnp.float32
    assert integrated.dtype == jnp.float32
    np.testing.assert_allclose(np.asarray(cdf[:, -1]), [1.0, 1.0])
    np.testing.assert_allclose(np.asarray(integrated), [1.0, 0.5])


def test_ms2_kernel_and_support_indices_are_causal():
    sampling_times = jnp.asarray([0.0, 0.5, 1.0, 1.5], dtype=jnp.float32)
    event_times = jnp.asarray([0.25, 1.0], dtype=jnp.float32)
    offsets = jnp.arange(3, dtype=jnp.int32)

    indices = support_indices(event_times, sampling_times, offsets)
    values = ms2_kernel(
        jnp.asarray([-0.1, 0.0, 0.25, 0.5, 0.75], dtype=jnp.float32),
        jnp.asarray(0.5, dtype=jnp.float32),
        jnp.asarray(1.0, dtype=jnp.float32),
    )

    np.testing.assert_array_equal(np.asarray(indices), [[1, 2, 3], [2, 3, 4]])
    np.testing.assert_allclose(np.asarray(values), [0.0, 0.0, 0.5, 1.0, 1.0])


def test_sample_loadings_smoke_runs_with_missing_data():
    observed = jnp.asarray([[0.2, jnp.nan, 0.8]], dtype=jnp.float32)
    sampling_times = jnp.asarray([0.0, 0.5, 1.0], dtype=jnp.float32)
    fine_grid = jnp.asarray([0.0, 0.5, 1.0], dtype=jnp.float32)
    rates = jnp.asarray([[0.7, 0.8, 0.6]], dtype=jnp.float32)

    result = sample_loadings(
        observed=observed,
        noise_std=jnp.asarray(0.5, dtype=jnp.float32),
        rise_time=0.5,
        plateau_time=0.0,
        sampling_times=sampling_times,
        rates_on_grid=rates,
        fine_grid=fine_grid,
        seed=1,
        rna_intensity=jnp.asarray(1.0, dtype=jnp.float32),
        n_iter=24,
        nrepeat=2,
    )
    result.posterior_rate.block_until_ready()

    assert result.posterior_rate.shape == (1, 3)
    assert result.predicted_signal.shape == (1, 3)
    assert result.posterior_rate.dtype == jnp.float32
    assert np.all(np.isfinite(np.asarray(result.posterior_rate)))
    assert np.all(np.isfinite(np.asarray(result.predicted_signal)))


def test_compute_log_z_smoke_runs_and_returns_float32():
    observed = jnp.asarray([[0.1, 0.5]], dtype=jnp.float32)
    sampling_times = jnp.asarray([0.0, 0.5], dtype=jnp.float32)
    fine_grid = jnp.asarray([0.0, 0.5], dtype=jnp.float32)
    rates = jnp.asarray([[0.4, 0.6]], dtype=jnp.float32)

    result = compute_log_z(
        observed=observed,
        noise_std=jnp.asarray(0.7, dtype=jnp.float32),
        rise_time=0.5,
        plateau_time=0.0,
        sampling_times=sampling_times,
        rates_on_grid=rates,
        fine_grid=fine_grid,
        seed=2,
        rna_intensity=jnp.asarray(1.0, dtype=jnp.float32),
        n_iter=16,
        n_steps=3,
        nrepeat=2,
    )
    result.log_z.block_until_ready()

    assert isinstance(result.log_z, jax.Array)
    assert result.log_z.dtype == jnp.float32
    assert result.beta_grid.dtype == jnp.float32
    assert np.isfinite(float(result.log_z))
