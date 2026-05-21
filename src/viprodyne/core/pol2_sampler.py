"""JAX reversible-jump sampler for continuous Pol2 loading events."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

FLOAT_DTYPE = jnp.float32


@dataclass(frozen=True)
class Pol2SamplerResult:
    """Posterior summaries from the Pol2 loading sampler."""

    posterior_rate: jax.Array
    predicted_signal: jax.Array
    particle_trace: jax.Array
    energy_trace: jax.Array
    final_count_grid: jax.Array
    final_signal: jax.Array
    mean_energy: jax.Array


@dataclass(frozen=True)
class ThermodynamicIntegrationResult:
    """Thermodynamic-integration estimate of the sampler log partition."""

    log_z: jax.Array
    final_energy_plus_log_prior: jax.Array
    beta_grid: jax.Array
    energies: jax.Array
    log_prior_sums: jax.Array


@jax.jit
def setup_sampler(
    time_grid: jax.Array, rate_values: jax.Array
) -> tuple[jax.Array, jax.Array]:
    """Precompute inverse-CDF tables and integrated loading rates."""
    time_grid = jnp.asarray(time_grid, dtype=FLOAT_DTYPE)
    rates = jnp.clip(jnp.asarray(rate_values, dtype=FLOAT_DTYPE), 0.0, None)
    rate_sum = jnp.sum(rates, axis=-1)
    safe_rate_sum = jnp.maximum(rate_sum, jnp.finfo(FLOAT_DTYPE).tiny)
    rate_pdf = rates / safe_rate_sum[:, None]
    rate_cdf = jnp.cumsum(rate_pdf, axis=-1)
    rate_cdf = rate_cdf.at[:, -1].set(1.0)
    dt = time_grid[1] - time_grid[0]
    integrated_rates = rate_sum * dt
    return rate_cdf.astype(FLOAT_DTYPE), integrated_rates.astype(FLOAT_DTYPE)


_search_per_track = jax.vmap(
    lambda cdf, u: jnp.searchsorted(cdf, u, side="left"),
    in_axes=(0, 0),
)
_search_constant_grid = jax.vmap(
    lambda grid, value: jnp.searchsorted(grid, value, side="left"),
    in_axes=(None, 0),
)
_choice_per_track = jax.vmap(
    lambda key, n, probabilities: jax.random.choice(key, n, p=probabilities),
    in_axes=(0, None, 0),
)


@jax.jit
def proposal_gen(
    key: jax.Array,
    time_grid: jax.Array,
    cdfs: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Sample event times from the loading-rate proposal density."""
    u = jax.random.uniform(key, shape=(cdfs.shape[0],), dtype=FLOAT_DTYPE)
    indices = _search_per_track(cdfs, u)
    indices = jnp.minimum(indices, time_grid.shape[0] - 1)
    return time_grid[indices], indices


@jax.jit
def support_indices(
    event_times: jax.Array,
    sampling_times: jax.Array,
    offsets: jax.Array,
) -> jax.Array:
    """Return observation indices whose kernel support may overlap events."""
    first = _search_constant_grid(sampling_times, event_times)
    return first[:, None] + offsets[None, :]


@jax.jit
def ms2_kernel(
    age: jax.Array, rise_time: jax.Array, support_time: jax.Array
) -> jax.Array:
    """Piecewise-linear MS2 kernel: linear rise followed by a plateau."""
    age = jnp.asarray(age, dtype=FLOAT_DTYPE)
    rise_time = jnp.asarray(rise_time, dtype=FLOAT_DTYPE)
    support_time = jnp.asarray(support_time, dtype=FLOAT_DTYPE)
    in_support = (age >= 0.0) & (age <= support_time)
    ramp = jnp.where(age <= rise_time, age / rise_time, 1.0)
    return jnp.where(in_support, ramp, 0.0).astype(FLOAT_DTYPE)


@jax.jit
def _kernel_values(
    event_times: jax.Array,
    sampling_times: jax.Array,
    offsets: jax.Array,
    rise_time: jax.Array,
    support_time: jax.Array,
    rna_intensity: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    indices = support_indices(event_times, sampling_times, offsets)
    valid = (indices >= 0) & (indices < sampling_times.shape[0])
    safe_indices = jnp.clip(indices, 0, sampling_times.shape[0] - 1)
    selected_times = sampling_times[safe_indices]
    scale = jnp.broadcast_to(rna_intensity, (event_times.shape[0],))
    values = ms2_kernel(selected_times - event_times[:, None], rise_time, support_time)
    values = jnp.where(valid, values * scale[:, None], 0.0)
    return safe_indices, valid, values.astype(FLOAT_DTYPE)


@jax.jit
def _add_point_update(
    event_times: jax.Array,
    grid_indices: jax.Array,
    signal: jax.Array,
    offsets: jax.Array,
    count_grid: jax.Array,
    log_prior_sum: jax.Array,
    rise_time: jax.Array,
    support_time: jax.Array,
    rna_intensity: jax.Array,
    sampling_times: jax.Array,
    signs: jax.Array,
    rate_at_event: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    track_index = jnp.arange(signal.shape[0])[:, None]
    count_grid = count_grid.at[jnp.arange(count_grid.shape[0]), grid_indices].add(signs)
    indices, _, values = _kernel_values(
        event_times,
        sampling_times,
        offsets,
        rise_time,
        support_time,
        rna_intensity,
    )
    signal = signal.at[track_index, indices].add(signs[:, None] * values)
    log_prior_sum = log_prior_sum + jnp.log(rate_at_event + 1e-9) * signs
    return count_grid, signal, log_prior_sum.astype(FLOAT_DTYPE)


@jax.jit
def papangelou(
    event_times: jax.Array,
    loading_rates: jax.Array,
    sampling_times: jax.Array,
    data_field: jax.Array,
    precision_field: jax.Array,
    signal: jax.Array,
    rise_time: jax.Array,
    support_time: jax.Array,
    rna_intensity: jax.Array,
    beta: jax.Array,
    offsets: jax.Array,
) -> jax.Array:
    """Compute the Papangelou conditional intensity for proposed events."""
    indices, valid, values = _kernel_values(
        event_times,
        sampling_times,
        offsets,
        rise_time,
        support_time,
        rna_intensity,
    )
    track_index = jnp.arange(data_field.shape[0])[:, None]
    data_selected = jnp.where(valid, data_field[track_index, indices], 0.0)
    precision_selected = jnp.where(valid, precision_field[track_index, indices], 0.0)
    signal_selected = jnp.where(valid, signal[track_index, indices], 0.0)
    data_term = jnp.sum(data_selected * values, axis=-1)
    self_term = 0.5 * jnp.sum(precision_selected * values * values, axis=-1)
    interaction_term = jnp.sum(precision_selected * signal_selected * values, axis=-1)
    return loading_rates * jnp.exp(beta * (data_term - self_term - interaction_term))


@jax.jit
def _sample_death_indices(key: jax.Array, count_grid: jax.Array) -> jax.Array:
    count_sums = jnp.sum(count_grid, axis=1)
    uniform = jnp.full_like(count_grid, 1.0 / count_grid.shape[1])
    probabilities = jnp.where(
        count_sums[:, None] > 0.0, count_grid / count_sums[:, None], uniform
    )
    keys = jax.random.split(key, probabilities.shape[0])
    return _choice_per_track(keys, probabilities.shape[1], probabilities)


@jax.jit
def _birth_move(
    key: jax.Array,
    count_grid: jax.Array,
    log_prior_sum: jax.Array,
    signal: jax.Array,
    rates_on_grid: jax.Array,
    rate_cdfs: jax.Array,
    sampling_times: jax.Array,
    rise_time: jax.Array,
    support_time: jax.Array,
    rna_intensity: jax.Array,
    integrated_rates: jax.Array,
    offsets: jax.Array,
    data_field: jax.Array,
    precision_field: jax.Array,
    fine_grid: jax.Array,
    beta: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    event_times, grid_indices = proposal_gen(key, fine_grid, rate_cdfs)
    rates = rates_on_grid[jnp.arange(event_times.shape[0]), grid_indices]
    papangelou_values = papangelou(
        event_times,
        rates,
        sampling_times,
        data_field,
        precision_field,
        signal,
        rise_time,
        support_time,
        rna_intensity,
        beta,
        offsets,
    )
    n_particles = jnp.sum(count_grid, axis=1)
    proposal_density = rates / jnp.maximum(
        integrated_rates, jnp.finfo(FLOAT_DTYPE).tiny
    )
    alphas = jnp.minimum(
        1.0, papangelou_values / (n_particles + 1.0) / proposal_density
    )
    accept_key = jax.random.split(key, 2)[1]
    accepted = (
        jax.random.uniform(accept_key, shape=alphas.shape, dtype=FLOAT_DTYPE) < alphas
    )
    return _add_point_update(
        event_times,
        grid_indices,
        signal,
        offsets,
        count_grid,
        log_prior_sum,
        rise_time,
        support_time,
        rna_intensity,
        sampling_times,
        accepted.astype(FLOAT_DTYPE),
        rates,
    )


@jax.jit
def _death_move(
    key: jax.Array,
    count_grid: jax.Array,
    log_prior_sum: jax.Array,
    signal: jax.Array,
    rates_on_grid: jax.Array,
    rate_cdfs: jax.Array,
    sampling_times: jax.Array,
    rise_time: jax.Array,
    support_time: jax.Array,
    rna_intensity: jax.Array,
    integrated_rates: jax.Array,
    offsets: jax.Array,
    data_field: jax.Array,
    precision_field: jax.Array,
    fine_grid: jax.Array,
    beta: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    del rate_cdfs
    n_particles = jnp.sum(count_grid, axis=1)
    grid_indices = _sample_death_indices(key, count_grid)
    event_times = fine_grid[grid_indices]
    rates = rates_on_grid[jnp.arange(grid_indices.shape[0]), grid_indices]
    remove_sign = -(n_particles > 0.0).astype(FLOAT_DTYPE)
    count_grid, signal, log_prior_sum = _add_point_update(
        event_times,
        grid_indices,
        signal,
        offsets,
        count_grid,
        log_prior_sum,
        rise_time,
        support_time,
        rna_intensity,
        sampling_times,
        remove_sign,
        rates,
    )
    papangelou_values = papangelou(
        event_times,
        rates,
        sampling_times,
        data_field,
        precision_field,
        signal,
        rise_time,
        support_time,
        rna_intensity,
        beta,
        offsets,
    )
    proposal_density = rates / jnp.maximum(
        integrated_rates, jnp.finfo(FLOAT_DTYPE).tiny
    )
    alphas = jnp.minimum(
        1.0,
        proposal_density * n_particles / jnp.maximum(papangelou_values, 1e-9),
    )
    accept_key = jax.random.split(key, 2)[1]
    rejected = (
        jax.random.uniform(accept_key, shape=alphas.shape, dtype=FLOAT_DTYPE) > alphas
    ) & (n_particles > 0.0)
    return _add_point_update(
        event_times,
        grid_indices,
        signal,
        offsets,
        count_grid,
        log_prior_sum,
        rise_time,
        support_time,
        rna_intensity,
        sampling_times,
        rejected.astype(FLOAT_DTYPE),
        rates,
    )


@jax.jit
def _move(
    key: jax.Array,
    count_grid: jax.Array,
    log_prior_sum: jax.Array,
    signal: jax.Array,
    rates_on_grid: jax.Array,
    rate_cdfs: jax.Array,
    sampling_times: jax.Array,
    rise_time: jax.Array,
    support_time: jax.Array,
    rna_intensity: jax.Array,
    integrated_rates: jax.Array,
    offsets: jax.Array,
    data_field: jax.Array,
    precision_field: jax.Array,
    fine_grid: jax.Array,
    beta: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    move_key, branch_key = jax.random.split(key)
    do_birth = jax.random.bernoulli(branch_key, p=0.5)
    args = (
        move_key,
        count_grid,
        log_prior_sum,
        signal,
        rates_on_grid,
        rate_cdfs,
        sampling_times,
        rise_time,
        support_time,
        rna_intensity,
        integrated_rates,
        offsets,
        data_field,
        precision_field,
        fine_grid,
        beta,
    )
    return jax.lax.cond(
        do_birth, lambda x: _birth_move(*x), lambda x: _death_move(*x), args
    )


@jax.jit
def _online_mean(
    mean: jax.Array, value: jax.Array, count: jax.Array, start_iter: jax.Array
) -> jax.Array:
    update = count >= start_iter
    new_mean = mean + (value - mean) / (count - start_iter + 1)
    return jax.lax.cond(update, lambda _: new_mean, lambda _: mean, operand=None)


@jax.jit
def _energy(
    data_field: jax.Array,
    signal: jax.Array,
    precision_field: jax.Array,
    noise: jax.Array,
) -> jax.Array:
    norm = jnp.sum(
        0.5
        * jnp.log(2.0 * jnp.pi * noise[:, None] ** 2)
        * noise[:, None] ** 2
        * precision_field,
        axis=1,
    )
    return (
        -0.5 * jnp.sum(data_field * data_field * noise[:, None] ** 2, axis=1)
        + jnp.sum(data_field * signal, axis=1)
        - 0.5 * jnp.sum(precision_field * signal * signal, axis=1)
        - norm
    ).astype(FLOAT_DTYPE)


def _prepare_repeated_inputs(
    observed: jax.Array,
    noise_std: jax.Array,
    rates_on_grid: jax.Array,
    rna_intensity: jax.Array,
    nrepeat: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    observed = jnp.asarray(observed, dtype=FLOAT_DTYPE)
    if observed.ndim == 1:
        observed = observed[None, :]
    rates_on_grid = jnp.asarray(rates_on_grid, dtype=FLOAT_DTYPE)
    if rates_on_grid.ndim == 1:
        rates_on_grid = rates_on_grid[None, :]
    base_ntraj = observed.shape[0]
    if rates_on_grid.shape[0] == 1 and base_ntraj > 1:
        rates_on_grid = jnp.broadcast_to(
            rates_on_grid,
            (base_ntraj, rates_on_grid.shape[1]),
        )
    elif rates_on_grid.shape[0] != base_ntraj:
        raise ValueError(
            "rates_on_grid must have shape (n_traces, n_grid) to match observed."
        )
    noise = jnp.asarray(noise_std, dtype=FLOAT_DTYPE)
    if noise.ndim == 0:
        noise = jnp.full((base_ntraj,), noise, dtype=FLOAT_DTYPE)
    rna = jnp.asarray(rna_intensity, dtype=FLOAT_DTYPE)
    if rna.ndim == 0:
        rna = jnp.full((base_ntraj,), rna, dtype=FLOAT_DTYPE)
    return (
        jnp.tile(observed, (nrepeat, 1)),
        jnp.tile(noise, nrepeat),
        jnp.tile(rates_on_grid, (nrepeat, 1)),
        jnp.tile(rna, nrepeat),
    )


def _support_offsets(
    sampling_times: jax.Array, rise_time: float, plateau_time: float
) -> jax.Array:
    times = jnp.asarray(sampling_times, dtype=FLOAT_DTYPE)
    dt = times[1] - times[0]
    n_support = (
        int(jnp.ceil((jnp.asarray(rise_time + plateau_time, dtype=FLOAT_DTYPE)) / dt))
        + 1
    )
    return jnp.arange(n_support, dtype=jnp.int32)


def sample_loadings(
    observed: jax.Array,
    noise_std: jax.Array | float,
    rise_time: float,
    plateau_time: float,
    sampling_times: jax.Array,
    rates_on_grid: jax.Array,
    fine_grid: jax.Array,
    seed: int,
    rna_intensity: jax.Array | float,
    n_iter: int = 15_000,
    beta: float = 1.0,
    nrepeat: int = 100,
) -> Pol2SamplerResult:
    """Sample continuous Pol2 loading events with reversible-jump MCMC."""
    observed_rep, noise, rates_rep, rna = _prepare_repeated_inputs(
        observed,
        noise_std,
        rates_on_grid,
        rna_intensity,
        nrepeat,
    )
    sampling_times = jnp.asarray(sampling_times, dtype=FLOAT_DTYPE)
    fine_grid = jnp.asarray(fine_grid, dtype=FLOAT_DTYPE)
    mask = jnp.isfinite(observed_rep)
    data_field = jnp.where(mask, observed_rep / (noise[:, None] ** 2), 0.0)
    precision_field = jnp.where(mask, 1.0 / (noise[:, None] ** 2), 0.0)
    rate_cdfs, integrated_rates = setup_sampler(fine_grid, rates_rep)
    offsets = _support_offsets(sampling_times, rise_time, plateau_time)
    support_time = jnp.asarray(rise_time + plateau_time, dtype=FLOAT_DTYPE)
    init_params = (
        rates_rep,
        rate_cdfs,
        sampling_times,
        jnp.asarray(rise_time, dtype=FLOAT_DTYPE),
        support_time,
        rna,
        integrated_rates,
        offsets,
        data_field,
        precision_field,
        fine_grid,
        jnp.asarray(beta, dtype=FLOAT_DTYPE),
        jnp.asarray(n_iter, dtype=jnp.int32),
        noise,
    )
    ntraj = observed_rep.shape[0]
    count_grid = jnp.zeros((ntraj, fine_grid.shape[0]), dtype=FLOAT_DTYPE)
    signal = jnp.zeros_like(observed_rep, dtype=FLOAT_DTYPE)
    carry_init = (
        init_params,
        jnp.asarray(0, dtype=jnp.int32),
        jnp.zeros_like(count_grid),
        jnp.zeros_like(signal),
        count_grid,
        signal,
        jnp.zeros((ntraj,), dtype=FLOAT_DTYPE),
        jnp.zeros((ntraj,), dtype=FLOAT_DTYPE),
    )
    keys = jax.random.split(jax.random.PRNGKey(seed), int(n_iter))
    (
        _,
        _,
        mean_count_grid,
        mean_signal,
        final_count_grid,
        final_signal,
        _,
        mean_energy,
    ), (particle_trace, energy_trace) = jax.lax.scan(_scan_sampler, carry_init, keys)
    base_ntraj = ntraj // nrepeat
    grid_dt = fine_grid[1] - fine_grid[0]
    posterior_rate = (
        mean_count_grid.reshape(nrepeat, base_ntraj, -1).mean(axis=0) / grid_dt
    )
    predicted_signal = mean_signal.reshape(nrepeat, base_ntraj, -1).mean(axis=0)
    mean_energy = mean_energy.reshape(nrepeat, base_ntraj).mean(axis=0)
    return Pol2SamplerResult(
        posterior_rate=posterior_rate.astype(FLOAT_DTYPE),
        predicted_signal=predicted_signal.astype(FLOAT_DTYPE),
        particle_trace=particle_trace.astype(FLOAT_DTYPE),
        energy_trace=energy_trace.astype(FLOAT_DTYPE),
        final_count_grid=final_count_grid.astype(FLOAT_DTYPE),
        final_signal=final_signal.astype(FLOAT_DTYPE),
        mean_energy=mean_energy.astype(FLOAT_DTYPE),
    )


@jax.jit
def _scan_sampler(carry, key):
    (
        params,
        count,
        mean_grid,
        mean_signal,
        count_grid,
        signal,
        log_prior_sum,
        mean_energy,
    ) = carry
    (
        rates_on_grid,
        rate_cdfs,
        sampling_times,
        rise_time,
        support_time,
        rna_intensity,
        integrated_rates,
        offsets,
        data_field,
        precision_field,
        fine_grid,
        beta,
        n_iter,
        noise,
    ) = params
    count_grid, signal, log_prior_sum = _move(
        key,
        count_grid,
        log_prior_sum,
        signal,
        rates_on_grid,
        rate_cdfs,
        sampling_times,
        rise_time,
        support_time,
        rna_intensity,
        integrated_rates,
        offsets,
        data_field,
        precision_field,
        fine_grid,
        beta,
    )
    burn_in = n_iter // 2
    mean_grid = _online_mean(mean_grid, count_grid, count, burn_in)
    mean_signal = _online_mean(mean_signal, signal, count, burn_in)
    current_energy = _energy(data_field, signal, precision_field, noise) + log_prior_sum
    mean_energy = _online_mean(mean_energy, current_energy, count, burn_in)
    count = count + 1
    return (
        params,
        count,
        mean_grid,
        mean_signal,
        count_grid,
        signal,
        log_prior_sum,
        mean_energy,
    ), (jnp.sum(count_grid, axis=1), current_energy)


def compute_log_z(
    observed: jax.Array,
    noise_std: jax.Array | float,
    rise_time: float,
    plateau_time: float,
    sampling_times: jax.Array,
    rates_on_grid: jax.Array,
    fine_grid: jax.Array,
    seed: int,
    rna_intensity: jax.Array | float,
    n_iter: int = 10_000,
    n_steps: int = 10,
    nrepeat: int = 20,
) -> ThermodynamicIntegrationResult:
    """Estimate the log normalizing constant by thermodynamic integration."""
    observed_rep, noise, rates_rep, rna = _prepare_repeated_inputs(
        observed,
        noise_std,
        rates_on_grid,
        rna_intensity,
        nrepeat,
    )
    sampling_times = jnp.asarray(sampling_times, dtype=FLOAT_DTYPE)
    fine_grid = jnp.asarray(fine_grid, dtype=FLOAT_DTYPE)
    mask = jnp.isfinite(observed_rep)
    data_field = jnp.where(mask, observed_rep / (noise[:, None] ** 2), 0.0)
    precision_field = jnp.where(mask, 1.0 / (noise[:, None] ** 2), 0.0)
    rate_cdfs, integrated_rates = setup_sampler(fine_grid, rates_rep)
    offsets = _support_offsets(sampling_times, rise_time, plateau_time)
    beta_grid = jnp.concatenate(
        [
            jnp.zeros((1,), dtype=FLOAT_DTYPE),
            jnp.logspace(
                jnp.log10(jnp.asarray(1.0 / n_steps / 10.0, dtype=FLOAT_DTYPE)),
                jnp.asarray(0.0, dtype=FLOAT_DTYPE),
                n_steps,
            ).astype(FLOAT_DTYPE),
        ]
    )

    def run_beta(beta):
        ntraj = observed_rep.shape[0]
        count_grid = jnp.zeros((ntraj, fine_grid.shape[0]), dtype=FLOAT_DTYPE)
        signal = jnp.zeros_like(observed_rep, dtype=FLOAT_DTYPE)
        carry_init = (
            jnp.asarray(0, dtype=jnp.int32),
            count_grid,
            signal,
            jnp.zeros((ntraj,), dtype=FLOAT_DTYPE),
            jnp.zeros((ntraj,), dtype=FLOAT_DTYPE),
        )
        keys = jax.random.split(jax.random.PRNGKey(seed), int(n_iter))
        final_state, _ = jax.lax.scan(
            _scan_thermodynamic,
            (
                carry_init,
                rates_rep,
                rate_cdfs,
                sampling_times,
                jnp.asarray(rise_time, dtype=FLOAT_DTYPE),
                jnp.asarray(rise_time + plateau_time, dtype=FLOAT_DTYPE),
                rna,
                integrated_rates,
                offsets,
                data_field,
                precision_field,
                fine_grid,
                beta,
                jnp.asarray(n_iter, dtype=jnp.int32),
                noise,
            ),
            keys,
        )
        _, _, _, log_prior_sum, mean_energy = final_state[0]
        base_ntraj = ntraj // nrepeat
        mean_energy = mean_energy.reshape(nrepeat, base_ntraj).mean(axis=0)
        log_prior_sum = log_prior_sum.reshape(nrepeat, base_ntraj).mean(axis=0)
        return jnp.sum(mean_energy), jnp.sum(log_prior_sum)

    energies, log_prior_sums = jax.vmap(run_beta)(beta_grid)
    log_z = jnp.trapezoid(energies, beta_grid)
    return ThermodynamicIntegrationResult(
        log_z=log_z.astype(FLOAT_DTYPE),
        final_energy_plus_log_prior=(energies[-1] + log_prior_sums[-1]).astype(
            FLOAT_DTYPE
        ),
        beta_grid=beta_grid.astype(FLOAT_DTYPE),
        energies=energies.astype(FLOAT_DTYPE),
        log_prior_sums=log_prior_sums.astype(FLOAT_DTYPE),
    )


@jax.jit
def _scan_thermodynamic(carry_with_params, key):
    (
        carry,
        rates_on_grid,
        rate_cdfs,
        sampling_times,
        rise_time,
        support_time,
        rna_intensity,
        integrated_rates,
        offsets,
        data_field,
        precision_field,
        fine_grid,
        beta,
        n_iter,
        noise,
    ) = carry_with_params
    count, count_grid, signal, log_prior_sum, mean_energy = carry
    count_grid, signal, log_prior_sum = _move(
        key,
        count_grid,
        log_prior_sum,
        signal,
        rates_on_grid,
        rate_cdfs,
        sampling_times,
        rise_time,
        support_time,
        rna_intensity,
        integrated_rates,
        offsets,
        data_field,
        precision_field,
        fine_grid,
        beta,
    )
    current_energy = _energy(data_field, signal, precision_field, noise)
    mean_energy = _online_mean(mean_energy, current_energy, count, n_iter // 2)
    count = count + 1
    return (
        (count, count_grid, signal, log_prior_sum, mean_energy),
        rates_on_grid,
        rate_cdfs,
        sampling_times,
        rise_time,
        support_time,
        rna_intensity,
        integrated_rates,
        offsets,
        data_field,
        precision_field,
        fine_grid,
        beta,
        n_iter,
        noise,
    ), (jnp.sum(count_grid, axis=1), current_energy)
