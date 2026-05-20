"""Common variational-node and graph plumbing."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass
from typing import Mapping, MutableMapping

import numpy as np

MomentDict = dict[str, object]


class MomentStore:
    """Small message bus for node moments."""

    def __init__(self) -> None:
        self._moments: dict[str, MomentDict] = {}

    def publish(self, node_name: str, moments: Mapping[str, object]) -> None:
        self._moments[node_name] = dict(moments)

    def get(self, node_name: str) -> MomentDict:
        try:
            return self._moments[node_name]
        except KeyError as exc:
            raise KeyError(f"No moments have been published for node {node_name!r}.") from exc

    def as_dict(self) -> dict[str, MomentDict]:
        return dict(self._moments)


@dataclass(frozen=True)
class UpdateContext:
    """Connectivity-aware context passed from a graph to a node update."""

    moments: MomentStore
    parent_names: tuple[str, ...]
    child_names: tuple[str, ...]
    blanket_names: tuple[str, ...] = ()
    rho: float = 1.0

    def parent_moments(self) -> dict[str, MomentDict]:
        return {name: self.moments.get(name) for name in self.parent_names}

    def child_moments(self) -> dict[str, MomentDict]:
        return {name: self.moments.get(name) for name in self.child_names}

    def blanket_moments(self) -> dict[str, MomentDict]:
        return {name: self.moments.get(name) for name in self.blanket_names}


class VariationalNode(ABC):
    """Base class for nodes in the mean-field variational graph."""

    def __init__(self, name: str) -> None:
        if not name:
            raise ValueError("node name must be non-empty.")
        self.name = name

    @abstractmethod
    def moments(self) -> MomentDict:
        """Return moments emitted by the variational distribution."""

    def initialize(self, moments: MomentStore) -> None:
        """Publish initial moments before a CAVI schedule is run."""
        moments.publish(self.name, self.moments())

    def update(self, context: UpdateContext) -> None:
        """Update the node. Subclasses may override."""

    @abstractmethod
    def entropy(self) -> float:
        """Return the entropy contribution from this node."""

    def elbo_contribution(self) -> float:
        """Return this node's local ELBO contribution when available."""
        return self.entropy()

    @abstractmethod
    def sample(self, rng: np.random.Generator | None = None, size=None):
        """Sample from the node distribution."""


class VariationalGraph:
    """Owns node connectivity and update scheduling."""

    def __init__(self) -> None:
        self.nodes: MutableMapping[str, VariationalNode] = OrderedDict()
        self._parents: dict[str, set[str]] = {}
        self._children: dict[str, set[str]] = {}
        self.moments = MomentStore()

    def add_node(self, node: VariationalNode) -> None:
        if node.name in self.nodes:
            raise ValueError(f"Duplicate node name {node.name!r}.")
        self.nodes[node.name] = node
        self._parents[node.name] = set()
        self._children[node.name] = set()
        node.initialize(self.moments)

    def add_edge(self, parent: str, child: str) -> None:
        self._require_node(parent)
        self._require_node(child)
        self._children[parent].add(child)
        self._parents[child].add(parent)

    def parents_of(self, node_name: str) -> tuple[str, ...]:
        self._require_node(node_name)
        return tuple(sorted(self._parents[node_name]))

    def children_of(self, node_name: str) -> tuple[str, ...]:
        self._require_node(node_name)
        return tuple(sorted(self._children[node_name]))

    def markov_blanket(self, node_name: str) -> tuple[str, ...]:
        """Return graph neighbors needed for a node-local variational update."""
        self._require_node(node_name)
        blanket = set(self._parents[node_name]) | set(self._children[node_name])
        for child in self._children[node_name]:
            blanket.update(self._parents[child])
        blanket.discard(node_name)
        return tuple(sorted(blanket))

    def run_schedule(self, schedule: list[str] | tuple[str, ...] | None = None, rho: float = 1.0) -> None:
        if not 0 < rho <= 1:
            raise ValueError("rho must be in (0, 1].")
        schedule = tuple(self.nodes) if schedule is None else tuple(schedule)
        for node_name in schedule:
            self._require_node(node_name)
            node = self.nodes[node_name]
            context = UpdateContext(
                moments=self.moments,
                parent_names=self.parents_of(node_name),
                child_names=self.children_of(node_name),
                blanket_names=self.markov_blanket(node_name),
                rho=float(rho),
            )
            node.update(context)
            self.moments.publish(node.name, node.moments())

    def _require_node(self, node_name: str) -> None:
        if node_name not in self.nodes:
            raise KeyError(f"Unknown node {node_name!r}.")
