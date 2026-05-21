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
    progress: bool = False
    progress_every: int = 1

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
        if self.progress_every < 1:
            raise ValueError("progress_every must be positive.")


@dataclass(frozen=True)
class CAVIIteration:
    """Convergence diagnostic for one CAVI sweep."""

    iteration: int
    max_parameter_change: np.float32
    converged: bool
    parameter_changes: dict[str, np.float32] = field(default_factory=dict)


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

    def __str__(self) -> str:
        status = "converged" if self.converged else "not converged"
        elbo = "not computed" if self.elbo is None else f"{float(self.elbo):.6g}"
        lines = [
            f"CAVIResult({status}, iterations={self.n_iterations}, "
            f"max_change={float(self.max_parameter_change):.3g}, elbo={elbo})"
        ]
        if self.history:
            last = self.history[-1]
            pending = {
                name: change
                for name, change in last.parameter_changes.items()
                if float(change) > 0.0
            }
            if pending:
                worst = sorted(
                    pending.items(),
                    key=lambda item: float(item[1]),
                    reverse=True,
                )[:5]
                lines.append(
                    "  largest node changes: "
                    + ", ".join(f"{name}={float(change):.3g}" for name, change in worst)
                )
        return "\n".join(lines)


def run_cavi(model: CAVIModel, config: CAVIConfig | None = None) -> CAVIResult:
    """Run CAVI and compute ELBO only after the final sweep."""
    config = CAVIConfig() if config is None else config
    schedule = (
        tuple(config.schedule) if config.schedule is not None else model.cavi_schedule()
    )
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
        parameter_changes = _snapshot_changes(
            previous,
            current,
            absolute_tolerance=config.absolute_tolerance,
        )
        max_change = _max_snapshot_change(parameter_changes)
        converged = bool(
            iteration >= config.min_iterations
            and float(max_change) <= float(config.tolerance)
        )
        history.append(
            CAVIIteration(
                iteration=iteration,
                max_parameter_change=np.asarray(max_change, dtype=FLOAT_DTYPE),
                converged=converged,
                parameter_changes=parameter_changes,
            )
        )
        if config.progress and (
            iteration == 1
            or iteration % config.progress_every == 0
            or converged
            or iteration == config.max_iterations
        ):
            _print_progress(
                iteration=iteration,
                max_iterations=config.max_iterations,
                tolerance=config.tolerance,
                max_change=max_change,
                changes=parameter_changes,
                converged=converged,
            )
        previous = current
        if converged:
            break
    if config.progress:
        print()
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


def _parameter_snapshot(
    graph, parameter_nodes: tuple[str, ...]
) -> dict[str, np.ndarray]:
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
            np.concatenate(parts).astype(FLOAT_DTYPE)
            if parts
            else np.empty(0, dtype=FLOAT_DTYPE)
        )
    return snapshot


def _snapshot_changes(
    previous: dict[str, np.ndarray],
    current: dict[str, np.ndarray],
    absolute_tolerance: float,
) -> dict[str, np.float32]:
    changes: dict[str, np.float32] = {}
    for key in sorted(set(previous) | set(current)):
        old = previous.get(key)
        new = current.get(key)
        if old is None or new is None or old.shape != new.shape:
            changes[key] = np.asarray(np.inf, dtype=FLOAT_DTYPE)
            continue
        if old.size == 0:
            changes[key] = np.asarray(0.0, dtype=FLOAT_DTYPE)
            continue
        denominator = np.maximum(
            np.maximum(np.abs(old), np.abs(new)),
            np.asarray(absolute_tolerance, dtype=FLOAT_DTYPE),
        )
        change = np.max(np.abs(new - old) / denominator)
        changes[key] = np.asarray(change, dtype=FLOAT_DTYPE)
    return changes


def _max_snapshot_change(changes: dict[str, np.float32]) -> np.float32:
    if not changes:
        return np.asarray(0.0, dtype=FLOAT_DTYPE)
    return np.asarray(
        max(float(value) for value in changes.values()),
        dtype=FLOAT_DTYPE,
    )


def _print_progress(
    *,
    iteration: int,
    max_iterations: int,
    tolerance: float,
    max_change: np.float32,
    changes: dict[str, np.float32],
    converged: bool,
) -> None:
    width = 24
    filled = int(round(width * iteration / max_iterations))
    bar = "#" * filled + "." * (width - filled)
    pending = [
        (name, change)
        for name, change in changes.items()
        if float(change) > float(tolerance)
    ]
    pending = sorted(pending, key=lambda item: float(item[1]), reverse=True)
    pending_text = "all parameter nodes converged"
    if pending:
        pending_text = "pending " + ", ".join(
            f"{name}={float(change):.2g}" for name, change in pending[:4]
        )
        if len(pending) > 4:
            pending_text += f", +{len(pending) - 4} more"
    status = "converged" if converged else "running"
    print(
        f"CAVI [{bar}] {iteration}/{max_iterations} {status}; "
        f"max change={float(max_change):.3g}; {pending_text}",
        end="\r",
        flush=True,
    )
