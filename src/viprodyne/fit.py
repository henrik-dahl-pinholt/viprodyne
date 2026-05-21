"""Coordinate-ascent variational inference runners."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

FLOAT_DTYPE = np.float32


class CAVIModel(Protocol):
    """Protocol implemented by model objects that can run CAVI."""

    graph: object

    def cavi_schedule(self) -> tuple[str, ...]: ...

    def parameter_node_names(self) -> tuple[str, ...]: ...

    def compute_elbo(self) -> np.float32: ...


@dataclass(frozen=True)
class CAVIConfig:
    """Configuration for coordinate-ascent variational inference.

    `run_cavi` monitors convergence from parameter changes and computes the
    ELBO only after the final sweep when `compute_elbo=True`.
    """

    max_iterations: int = 100
    min_iterations: int = 2
    tolerance: float = 1e-4
    absolute_tolerance: float = 1e-6
    rho: float = 1.0
    schedule: tuple[str, ...] | None = None
    parameter_nodes: tuple[str, ...] | None = None
    compute_elbo: bool = True

    def __post_init__(self) -> None:
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be positive.")
        if self.min_iterations < 0:
            raise ValueError("min_iterations must be non-negative.")
        if self.min_iterations > self.max_iterations:
            raise ValueError("min_iterations cannot exceed max_iterations.")
        if self.tolerance < 0:
            raise ValueError("tolerance must be non-negative.")
        if self.absolute_tolerance < 0:
            raise ValueError("absolute_tolerance must be non-negative.")
        if not 0 < self.rho <= 1:
            raise ValueError("rho must be in (0, 1].")


@dataclass(frozen=True)
class CAVIIteration:
    """Convergence diagnostic for one CAVI sweep."""

    iteration: int
    max_parameter_change: np.float32
    converged: bool


@dataclass(frozen=True)
class CAVIResult:
    """Diagnostics from a coordinate-ascent variational inference run."""

    converged: bool
    n_iterations: int
    max_parameter_change: np.float32
    elbo: np.float32 | None
    history: tuple[CAVIIteration, ...] = field(default_factory=tuple)
    schedule: tuple[str, ...] = field(default_factory=tuple)
    parameter_nodes: tuple[str, ...] = field(default_factory=tuple)


def run_cavi(model: CAVIModel, config: CAVIConfig | None = None) -> CAVIResult:
    """Run CAVI and compute ELBO only after the final sweep."""
    config = CAVIConfig() if config is None else config
    schedule = tuple(config.schedule) if config.schedule is not None else model.cavi_schedule()
    parameter_nodes = (
        tuple(config.parameter_nodes)
        if config.parameter_nodes is not None
        else model.parameter_node_names()
    )
    previous = _parameter_snapshot(model.graph, parameter_nodes)
    history: list[CAVIIteration] = []
    converged = False
    max_change = np.asarray(np.inf, dtype=FLOAT_DTYPE)
    for iteration in range(1, config.max_iterations + 1):
        model.graph.run_schedule(schedule=schedule, rho=config.rho)
        current = _parameter_snapshot(model.graph, parameter_nodes)
        max_change = _max_snapshot_change(
            previous,
            current,
            absolute_tolerance=config.absolute_tolerance,
        )
        converged = bool(
            iteration >= config.min_iterations
            and float(max_change) <= float(config.tolerance)
        )
        history.append(
            CAVIIteration(
                iteration=iteration,
                max_parameter_change=np.asarray(max_change, dtype=FLOAT_DTYPE),
                converged=converged,
            )
        )
        previous = current
        if converged:
            break
    elbo = model.compute_elbo() if config.compute_elbo else None
    return CAVIResult(
        converged=converged,
        n_iterations=len(history),
        max_parameter_change=np.asarray(max_change, dtype=FLOAT_DTYPE),
        elbo=None if elbo is None else np.asarray(elbo, dtype=FLOAT_DTYPE),
        history=tuple(history),
        schedule=schedule,
        parameter_nodes=parameter_nodes,
    )


def _parameter_snapshot(graph, parameter_nodes: tuple[str, ...]) -> dict[str, np.ndarray]:
    snapshot: dict[str, np.ndarray] = {}
    for node_name in parameter_nodes:
        node = graph.nodes[node_name]
        parts = []
        for attribute in ("concentration", "shape", "rate", "value"):
            if hasattr(node, attribute):
                value = getattr(node, attribute)
                if value is not None:
                    parts.append(np.ravel(np.asarray(value, dtype=FLOAT_DTYPE)))
        if not parts:
            moments = graph.moments.get(node_name)
            if "mean" in moments:
                parts.append(np.ravel(np.asarray(moments["mean"], dtype=FLOAT_DTYPE)))
        snapshot[node_name] = (
            np.concatenate(parts).astype(FLOAT_DTYPE) if parts else np.empty(0, dtype=FLOAT_DTYPE)
        )
    return snapshot


def _max_snapshot_change(
    previous: dict[str, np.ndarray],
    current: dict[str, np.ndarray],
    absolute_tolerance: float,
) -> np.float32:
    max_change = np.asarray(0.0, dtype=FLOAT_DTYPE)
    for key in sorted(set(previous) | set(current)):
        old = previous.get(key)
        new = current.get(key)
        if old is None or new is None or old.shape != new.shape:
            return np.asarray(np.inf, dtype=FLOAT_DTYPE)
        if old.size == 0:
            continue
        denominator = np.maximum(
            np.maximum(np.abs(old), np.abs(new)),
            np.asarray(absolute_tolerance, dtype=FLOAT_DTYPE),
        )
        change = np.max(np.abs(new - old) / denominator)
        max_change = np.maximum(max_change, np.asarray(change, dtype=FLOAT_DTYPE))
    return np.asarray(max_change, dtype=FLOAT_DTYPE)
