"""State-transition edge metadata.

Transition-rate parameters are ordered by counting off-diagonal entries row by row.
For a three-state promoter this gives [[-, 0, 1], [2, -, 3], [4, 5, -]].
"""

from __future__ import annotations

from dataclasses import dataclass


def ordered_transition_index(n_states: int, source_state: int, target_state: int) -> int:
    """Return the row-major off-diagonal index for a transition."""
    if n_states < 2:
        raise ValueError("n_states must be at least 2.")
    if source_state == target_state:
        raise ValueError("self transitions do not have transition-rate indices.")
    if not 0 <= source_state < n_states:
        raise ValueError("source_state is outside [0, n_states).")
    if not 0 <= target_state < n_states:
        raise ValueError("target_state is outside [0, n_states).")
    return source_state * (n_states - 1) + target_state - int(target_state > source_state)


def transition_states(n_states: int, transition_index: int) -> tuple[int, int]:
    """Invert :func:`ordered_transition_index`."""
    n_edges = n_states * (n_states - 1)
    if not 0 <= transition_index < n_edges:
        raise ValueError("transition_index is outside [0, n_states * (n_states - 1)).")
    source_state, within_row = divmod(int(transition_index), n_states - 1)
    target_state = within_row if within_row < source_state else within_row + 1
    return source_state, target_state


@dataclass(frozen=True)
class RateEdge:
    """A promoter transition edge supplied to a promoter-state variational node."""

    n_states: int
    source_state: int
    target_state: int
    rate_node: str
    drive_node: str | None = None
    transition_index: int | None = None

    def __post_init__(self) -> None:
        expected = ordered_transition_index(self.n_states, self.source_state, self.target_state)
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

