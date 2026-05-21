"""MS2 kernel specifications and internal observation-model builders."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

FLOAT_DTYPE = np.float32
KernelFunction = Callable[[jnp.ndarray], jnp.ndarray]


@dataclass(frozen=True)
class ProximalKernel:
    """Ramp-then-plateau kernel for a proximal MS2 cassette.

    The kernel rises linearly over `t_rise`, stays at `rna_intensity` for
    `t_plateau`, and is zero outside that support.
    """

    t_rise: np.ndarray | float = np.float32(1.0)
    t_plateau: np.ndarray | float = np.float32(0.0)
    rna_intensity: np.ndarray | float = np.float32(1.0)

    def __post_init__(self) -> None:
        object.__setattr__(self, "t_rise", np.asarray(self.t_rise, dtype=FLOAT_DTYPE))
        object.__setattr__(self, "t_plateau", np.asarray(self.t_plateau, dtype=FLOAT_DTYPE))
        object.__setattr__(
            self,
            "rna_intensity",
            np.asarray(self.rna_intensity, dtype=FLOAT_DTYPE),
        )
        if np.any(np.asarray(self.t_rise) <= 0.0):
            raise ValueError("t_rise must be positive.")
        if np.any(np.asarray(self.t_plateau) < 0.0):
            raise ValueError("t_plateau must be non-negative.")
        if np.any(np.asarray(self.rna_intensity) < 0.0):
            raise ValueError("rna_intensity must be non-negative.")

    def __call__(self, time_offsets: jnp.ndarray) -> jnp.ndarray:
        return proximal_kernel(
            time_offsets,
            jnp.asarray(self.t_rise, dtype=jnp.float32),
            jnp.asarray(self.t_plateau, dtype=jnp.float32),
            jnp.asarray(self.rna_intensity, dtype=jnp.float32),
        )


@dataclass(frozen=True)
class MS2ObservationModel:
    """Internal Pol2 observation representation derived from an MS2 kernel."""

    mode: str
    sampling_times: np.ndarray
    loading_times: np.ndarray
    design_matrix: np.ndarray | None = None
    window_weights: np.ndarray | None = None
    observation_starts: np.ndarray | None = None


def proximal_kernel(
    time_offsets: jnp.ndarray,
    t_rise: jnp.ndarray,
    t_plateau: jnp.ndarray,
    rna_intensity: jnp.ndarray = jnp.asarray(1.0, dtype=jnp.float32),
) -> jnp.ndarray:
    """Evaluate the proximal ramp-then-plateau MS2 kernel."""
    time_offsets = jnp.asarray(time_offsets, dtype=jnp.float32)
    t_rise = jnp.asarray(t_rise, dtype=jnp.float32)
    t_plateau = jnp.asarray(t_plateau, dtype=jnp.float32)
    rna_intensity = jnp.asarray(rna_intensity, dtype=jnp.float32)
    support_time = t_rise + t_plateau
    rising = (time_offsets >= 0.0) & (time_offsets < t_rise)
    plateau = (time_offsets >= t_rise) & (time_offsets <= support_time)
    ramp = rna_intensity * time_offsets / t_rise
    values = jnp.where(rising, ramp, 0.0)
    values = jnp.where(plateau, rna_intensity, values)
    return values.astype(jnp.float32)


def evaluate_ms2_kernel(time_offsets: jnp.ndarray, kernel: KernelFunction) -> jnp.ndarray:
    """Evaluate a configured MS2 kernel on time offsets."""
    offsets = jnp.asarray(time_offsets, dtype=jnp.float32)
    return jnp.asarray(kernel(offsets), dtype=jnp.float32)


def resolve_ms2_kernel(
    kernel: ProximalKernel | str | KernelFunction | None,
    t_rise: np.ndarray | float,
    t_plateau: np.ndarray | float,
    rna_intensity: np.ndarray | float,
) -> KernelFunction | None:
    """Normalize public kernel configuration to a JAX-compatible callable."""
    if kernel is None:
        return None
    if isinstance(kernel, ProximalKernel):
        return kernel
    if callable(kernel):
        return kernel
    name = str(kernel).lower()
    if name != "proximal":
        raise ValueError("unknown MS2 kernel name.")
    return ProximalKernel(
        t_rise=t_rise,
        t_plateau=t_plateau,
        rna_intensity=rna_intensity,
    )


def build_ms2_observation_model(
    time_grid: np.ndarray,
    n_observations: int,
    kernel: KernelFunction,
    sampling_times: np.ndarray | None = None,
    mode: str = "transfer",
    tolerance: float = 1e-7,
) -> MS2ObservationModel:
    """Build internal dense or transfer Pol2 observation inputs."""
    time_grid = _validate_time_grid(time_grid)
    n_observations = int(n_observations)
    if n_observations <= 0:
        raise ValueError("n_observations must be positive.")
    loading_times = time_grid[:-1]
    sampling_times_arr = _sampling_times(time_grid, n_observations, sampling_times)
    if mode not in {"transfer", "mean_field", "exact", "sampler"}:
        raise ValueError("mode must be 'transfer', 'mean_field', 'exact', or 'sampler'.")
    if mode == "sampler":
        return MS2ObservationModel(
            mode="sampler",
            sampling_times=sampling_times_arr,
            loading_times=loading_times,
        )
    if mode == "transfer":
        window_weights, starts = transfer_windows_from_kernel(
            sampling_times_arr,
            loading_times,
            kernel,
            tolerance=tolerance,
        )
        return MS2ObservationModel(
            mode="transfer",
            sampling_times=sampling_times_arr,
            loading_times=loading_times,
            window_weights=window_weights,
            observation_starts=starts,
        )
    design = _kernel_design_matrix(sampling_times_arr, loading_times, kernel)
    return MS2ObservationModel(
        mode=mode,
        sampling_times=sampling_times_arr,
        loading_times=loading_times,
        design_matrix=design,
    )


def transfer_windows_from_design(
    design_matrix: np.ndarray,
    tolerance: float = 1e-7,
) -> tuple[np.ndarray, np.ndarray]:
    """Compress a banded design matrix into row-specific transfer windows."""
    design = np.asarray(design_matrix, dtype=FLOAT_DTYPE)
    if design.ndim != 2:
        raise ValueError("design_matrix must be two-dimensional.")
    if design.shape[0] == 0 or design.shape[1] == 0:
        raise ValueError("design_matrix must be non-empty.")
    support = np.abs(design) > np.float32(tolerance)
    first = np.zeros(design.shape[0], dtype=np.int32)
    last = np.zeros(design.shape[0], dtype=np.int32)
    previous_start = np.int32(0)
    max_width = np.int32(1)
    for row in range(design.shape[0]):
        columns = np.flatnonzero(support[row])
        if columns.size == 0:
            first[row] = previous_start
            last[row] = previous_start
            continue
        first[row] = np.int32(columns[0])
        last[row] = np.int32(columns[-1])
        previous_start = first[row]
        max_width = np.maximum(max_width, np.int32(last[row] - first[row] + 1))
    max_start = np.int32(max(0, design.shape[1] - int(max_width)))
    starts = np.minimum(first, max_start).astype(np.int32)
    if np.any(np.diff(starts) < 0):
        raise ValueError("MS2 kernel support must move monotonically through loading time.")
    windows = np.zeros((design.shape[0], int(max_width)), dtype=FLOAT_DTYPE)
    for row, start in enumerate(starts):
        stop = min(int(start) + int(max_width), design.shape[1])
        span = stop - int(start)
        windows[row, :span] = design[row, int(start) : stop]
    return windows.astype(FLOAT_DTYPE), starts


def transfer_windows_from_kernel(
    sampling_times: np.ndarray,
    loading_times: np.ndarray,
    kernel: KernelFunction,
    tolerance: float = 1e-7,
) -> tuple[np.ndarray, np.ndarray]:
    """Build row-specific transfer windows without materializing a dense matrix."""
    sampling_times = np.asarray(sampling_times, dtype=FLOAT_DTYPE)
    loading_times = np.asarray(loading_times, dtype=FLOAT_DTYPE)
    if sampling_times.ndim != 1 or loading_times.ndim != 1:
        raise ValueError("sampling_times and loading_times must be one-dimensional.")
    if sampling_times.size == 0 or loading_times.size == 0:
        raise ValueError("sampling_times and loading_times must be non-empty.")
    first = np.zeros(sampling_times.size, dtype=np.int32)
    last = np.zeros(sampling_times.size, dtype=np.int32)
    previous_start = np.int32(0)
    max_width = np.int32(1)
    for row, sample_time in enumerate(sampling_times):
        values = _kernel_row(sample_time, loading_times, kernel)
        columns = np.flatnonzero(np.abs(values) > np.float32(tolerance))
        if columns.size == 0:
            first[row] = previous_start
            last[row] = previous_start
            continue
        first[row] = np.int32(columns[0])
        last[row] = np.int32(columns[-1])
        previous_start = first[row]
        max_width = np.maximum(max_width, np.int32(last[row] - first[row] + 1))
    max_start = np.int32(max(0, loading_times.size - int(max_width)))
    starts = np.minimum(first, max_start).astype(np.int32)
    if np.any(np.diff(starts) < 0):
        raise ValueError("MS2 kernel support must move monotonically through loading time.")
    windows = np.zeros((sampling_times.size, int(max_width)), dtype=FLOAT_DTYPE)
    for row, sample_time in enumerate(sampling_times):
        values = _kernel_row(sample_time, loading_times, kernel)
        start = int(starts[row])
        stop = min(start + int(max_width), loading_times.size)
        span = stop - start
        windows[row, :span] = values[start:stop]
    return windows.astype(FLOAT_DTYPE), starts


def _kernel_row(
    sample_time: np.ndarray | float,
    loading_times: np.ndarray,
    kernel: KernelFunction,
) -> np.ndarray:
    offsets = np.asarray(sample_time, dtype=FLOAT_DTYPE) - loading_times
    return np.asarray(evaluate_ms2_kernel(jnp.asarray(offsets), kernel), dtype=FLOAT_DTYPE)


def _kernel_design_matrix(
    sampling_times: np.ndarray,
    loading_times: np.ndarray,
    kernel: KernelFunction,
) -> np.ndarray:
    offsets = sampling_times[:, None] - loading_times[None, :]
    return np.asarray(evaluate_ms2_kernel(jnp.asarray(offsets), kernel), dtype=FLOAT_DTYPE)


def _sampling_times(
    time_grid: np.ndarray,
    n_observations: int,
    sampling_times: np.ndarray | None,
) -> np.ndarray:
    if sampling_times is None:
        n_loadings = time_grid.size - 1
        if n_observations > n_loadings:
            raise ValueError(
                "observed has more entries than loading intervals; provide a longer "
                "time_grid or explicit sampling_times."
            )
        return np.asarray(time_grid[1 : n_observations + 1], dtype=FLOAT_DTYPE)
    sampling_times_arr = np.asarray(sampling_times, dtype=FLOAT_DTYPE)
    if sampling_times_arr.shape != (n_observations,):
        raise ValueError("sampling_times must have one entry per observation.")
    if np.any(np.diff(sampling_times_arr) <= 0):
        raise ValueError("sampling_times must be strictly increasing.")
    return sampling_times_arr


def _validate_time_grid(time_grid: np.ndarray) -> np.ndarray:
    time_grid = np.asarray(time_grid, dtype=FLOAT_DTYPE)
    if time_grid.ndim != 1 or time_grid.size < 2:
        raise ValueError("time_grid must be one-dimensional with at least two entries.")
    if np.any(np.diff(time_grid) <= 0):
        raise ValueError("time_grid must be strictly increasing.")
    return time_grid
