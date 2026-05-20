import pytest

from viprodyne.core.rate_edges import RateEdge, ordered_transition_index, transition_states


def test_three_state_transition_order_matches_design_note():
    matrix = [[None for _ in range(3)] for _ in range(3)]
    for idx in range(6):
        source, target = transition_states(3, idx)
        matrix[source][target] = idx

    assert matrix == [[None, 0, 1], [2, None, 3], [4, 5, None]]


def test_ordered_transition_index_round_trip():
    for n_states in [2, 3, 5]:
        for source in range(n_states):
            for target in range(n_states):
                if source == target:
                    continue
                idx = ordered_transition_index(n_states, source, target)
                assert transition_states(n_states, idx) == (source, target)


def test_rate_edge_validates_supplied_index():
    edge = RateEdge(n_states=3, source_state=1, target_state=2, rate_node="R_12")
    assert edge.transition_index == 3
    assert not edge.is_driven

    driven = RateEdge(n_states=3, source_state=1, target_state=2, rate_node="R_12", drive_node="rc")
    assert driven.is_driven

    with pytest.raises(ValueError, match="does not match"):
        RateEdge(n_states=3, source_state=1, target_state=2, rate_node="R_12", transition_index=0)

