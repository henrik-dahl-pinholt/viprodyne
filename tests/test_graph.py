import numpy as np

from viprodyne.variational.base import UpdateContext, VariationalGraph, VariationalNode
from viprodyne.variational.distributions import DeltaNode


class RecordingNode(VariationalNode):
    def __init__(self, name: str):
        super().__init__(name)
        self.records = []

    def moments(self):
        return {"mean": np.array(1.0)}

    def update(self, context: UpdateContext) -> None:
        self.records.append(
            {
                "parents": context.parent_names,
                "children": context.child_names,
                "blanket": context.blanket_names,
                "parent_moments": context.parent_moments(),
                "child_moments": context.child_moments(),
                "blanket_moments": context.blanket_moments(),
                "rho": context.rho,
            }
        )

    def entropy(self) -> float:
        return 0.0

    def sample(self, rng=None, size=None):
        return 1.0


def test_graph_owns_connectivity_and_passes_context_to_nodes():
    graph = VariationalGraph()
    parent = DeltaNode("rate", 2.0)
    co_parent = DeltaNode("noise", 0.5)
    child = DeltaNode("observed", 5.0)
    node = RecordingNode("promoter")

    graph.add_node(parent)
    graph.add_node(co_parent)
    graph.add_node(node)
    graph.add_node(child)
    graph.add_edge("rate", "promoter")
    graph.add_edge("noise", "observed")
    graph.add_edge("promoter", "observed")

    graph.run_schedule(["promoter"], rho=0.25)

    assert graph.parents_of("promoter") == ("rate",)
    assert graph.children_of("promoter") == ("observed",)
    assert graph.markov_blanket("promoter") == ("noise", "observed", "rate")
    assert node.records[0]["parents"] == ("rate",)
    assert node.records[0]["children"] == ("observed",)
    assert node.records[0]["blanket"] == ("noise", "observed", "rate")
    assert node.records[0]["parent_moments"]["rate"]["mean"] == 2.0
    assert node.records[0]["child_moments"]["observed"]["mean"] == 5.0
    assert node.records[0]["blanket_moments"]["noise"]["mean"] == 0.5
    assert node.records[0]["rho"] == 0.25
