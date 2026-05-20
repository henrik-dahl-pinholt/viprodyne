import pytest

import numpy as np

from viprodyne.core.rate_edges import (
    RateEdge,
    ordered_transition_index,
    transition_states,
    unwrap_column_generator,
    validate_column_generator,
    wrap_column_generator,
)


def test_three_state_transition_order_matches_design_note():
    matrix = [[None for _ in range(3)] for _ in range(3)]
    for idx in range(6):
        source, target = transition_states(3, idx)
        matrix[source][target] = idx

    assert matrix == [[None, 0, 1], [2, None, 3], [4, 5, None]]


def test_ordered_transition_index_round_trip():
    for n_states in [2, 3, 5]:
        for to_state in range(n_states):
            for from_state in range(n_states):
                if to_state == from_state:
                    continue
                idx = ordered_transition_index(n_states, to_state, from_state)
                assert transition_states(n_states, idx) == (to_state, from_state)


def test_rate_edge_validates_supplied_index():
    edge = RateEdge(n_states=3, to_state=1, from_state=2, rate_node="R_12")
    assert edge.transition_index == 3  # Q[1, 2]
    assert not edge.is_driven

    driven = RateEdge(n_states=3, to_state=1, from_state=2, rate_node="R_12", drive_node="rc")
    assert driven.is_driven

    with pytest.raises(ValueError, match="does not match"):
        RateEdge(n_states=3, to_state=1, from_state=2, rate_node="R_12", transition_index=0)


def test_wrap_column_generator_uses_column_sum_zero_convention():
    generator = wrap_column_generator(np.arange(6.0), n_states=3)

    expected = np.array([[-6.0, 0.0, 1.0], [2.0, -5.0, 3.0], [4.0, 5.0, -4.0]])
    assert generator.dtype == np.float32
    np.testing.assert_allclose(generator, expected)
    np.testing.assert_allclose(np.sum(generator, axis=0), np.zeros(3))
    unwrapped = unwrap_column_generator(generator)
    assert unwrapped.dtype == np.float32
    np.testing.assert_allclose(unwrapped, np.arange(6.0))
    validate_column_generator(generator)
