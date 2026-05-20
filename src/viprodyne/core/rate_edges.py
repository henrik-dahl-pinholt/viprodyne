"""State-transition edge metadata for column-sum-zero CTMC generators.

Generators use the convention ``Q[to_state, from_state]`` and columns sum to
zero. Transition-rate parameters are ordered by counting off-diagonal matrix
entries row by row. For a three-state promoter this gives
``[[-, 0, 1], [2, -, 3], [4, 5, -]]``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def ordered_transition_index(n_states: int, to_state: int, from_state: int) -> int:
    """Return the row-major off-diagonal index for ``Q[to_state, from_state]``."""
    if n_states < 2:
        raise ValueError("n_states must be at least 2.")
    if to_state == from_state:
        raise ValueError("self transitions do not have transition-rate indices.")
    if not 0 <= to_state < n_states:
        raise ValueError("to_state is outside [0, n_states).")
    if not 0 <= from_state < n_states:
        raise ValueError("from_state is outside [0, n_states).")
    return to_state * (n_states - 1) + from_state - int(from_state > to_state)


def transition_states(n_states: int, transition_index: int) -> tuple[int, int]:
    """Invert :func:`ordered_transition_index`.

    Returns ``(to_state, from_state)``.
    """
    n_edges = n_states * (n_states - 1)
    if not 0 <= transition_index < n_edges:
        raise ValueError("transition_index is outside [0, n_states * (n_states - 1)).")
    to_state, within_row = divmod(int(transition_index), n_states - 1)
    from_state = within_row if within_row < to_state else within_row + 1
    return to_state, from_state


def wrap_column_generator(offdiag_rates: np.ndarray, n_states: int | None = None) -> np.ndarray:
    """Fill a column-sum-zero generator from row-major off-diagonal rates."""
    rates = np.asarray(offdiag_rates, dtype=float)
    if rates.ndim != 1:
        raise ValueError("offdiag_rates must be one-dimensional.")
    if np.any(rates < 0):
        raise ValueError("offdiag_rates must be non-negative.")
    if n_states is None:
        n_states = int(0.5 * (1.0 + np.sqrt(1.0 + 4.0 * rates.size)))
    if rates.size != n_states * (n_states - 1):
        raise ValueError("offdiag_rates length must equal n_states * (n_states - 1).")
    generator = np.zeros((n_states, n_states), dtype=float)
    for index, rate in enumerate(rates):
        to_state, from_state = transition_states(n_states, index)
        generator[to_state, from_state] = rate
    np.fill_diagonal(generator, 0.0)
    generator -= np.diag(np.sum(generator, axis=0))
    return generator


def unwrap_column_generator(generator: np.ndarray) -> np.ndarray:
    """Return row-major off-diagonal entries from a column-sum-zero generator."""
    generator = np.asarray(generator, dtype=float)
    if generator.ndim != 2 or generator.shape[0] != generator.shape[1]:
        raise ValueError("generator must be a square matrix.")
    n_states = generator.shape[0]
    return np.asarray(
        [
            generator[to_state, from_state]
            for to_state in range(n_states)
            for from_state in range(n_states)
            if to_state != from_state
        ],
        dtype=float,
    )


def validate_column_generator(generator: np.ndarray, atol: float = 1e-10) -> np.ndarray:
    """Validate and return a column-sum-zero CTMC generator."""
    generator = np.asarray(generator, dtype=float)
    if generator.ndim != 2 or generator.shape[0] != generator.shape[1]:
        raise ValueError("generator must be a square matrix.")
    offdiag = generator.copy()
    np.fill_diagonal(offdiag, 0.0)
    if np.any(offdiag < -atol):
        raise ValueError("off-diagonal generator entries must be non-negative.")
    if not np.allclose(np.sum(generator, axis=0), 0.0, atol=atol):
        raise ValueError("generator columns must sum to zero.")
    return generator


@dataclass(frozen=True)
class RateEdge:
    """A promoter transition edge supplied to a promoter-state variational node.

    The transition is from ``from_state`` to ``to_state`` and occupies matrix
    entry ``Q[to_state, from_state]``.
    """

    n_states: int
    to_state: int
    from_state: int
    rate_node: str
    drive_node: str | None = None
    transition_index: int | None = None

    def __post_init__(self) -> None:
        expected = ordered_transition_index(self.n_states, self.to_state, self.from_state)
        if self.transition_index is None:
            object.__setattr__(self, "transition_index", expected)
        elif self.transition_index != expected:
            raise ValueError(
                "transition_index does not match row-major off-diagonal ordering: "
                f"expected {expected}, got {self.transition_index}."
            )

    @property
    def is_driven(self) -> bool:
        """Whether this transition is modulated by a time-varying drive."""
        return self.drive_node is not None
