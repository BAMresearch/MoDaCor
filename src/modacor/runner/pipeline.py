# src/modacor/runner/pipeline.py
# # -*- coding: utf-8 -*-
from __future__ import annotations

from graphlib import TopologicalSorter
from pathlib import Path

from attrs import define, field
from attrs import validators as v

from ..dataclasses.process_step import ProcessStep

__all__ = ["Pipeline"]


@define
class Pipeline(TopologicalSorter):
    """
    Pipeline nodes are assumed to be of type ProcessStep
    """

    name: str = field(default="Unnamed Pipeline")
    graph: dict[ProcessStep] = field(factory=dict)

    def __attrs_post_init__(self):
        super().__init__(graph=self.graph)

    @classmethod
    def from_json(cls, path_to_json: Path):
        # functionality postponed
        return cls(name="dummy")

    @classmethod
    def from_dict(cls, graph_dict: dict, name=""):
        return cls(name=name, graph=graph_dict)

    def add_incoming_branch(self, branch: Self, branching_node):
        """
        Add a pipeline as a branch whose outcome shall be combined the existing pipeline.

        This assumes that the branch to be added has a single exit point.

        """
        pipeline_to_add = Pipeline(graph=branch.graph)
        pipeline_to_add_ordered = [*pipeline_to_add.static_order()]
        # add the last node of the incoming as a predecessor to the connection point
        self.graph[branching_node].update({pipeline_to_add_ordered[-1]})
        # add the rest of the graph
        self.graph = self.graph | branch.graph
        # reinitialize the TopologicalSorter
        super().__init__(graph=self.graph)

    def add_outgoing_branch(self, branch: Self, branching_node):
        """
        Add a pipeline as a branch whose input is based on the existing pipeline.

        This assumes that the branch to be added has a single entry point.

        """
        pipeline_to_add = Pipeline(graph=branch.graph)
        pipeline_to_add_ordered = [*pipeline_to_add.static_order()]
        # add the connection node as a predecessor to the first node of the outgoing branch
        branch.graph[pipeline_to_add_ordered[0]].update({branching_node})
        # add the rest of the graph
        self.graph = self.graph | branch.graph
        # reinitialize the TopologicalSorter
        super().__init__(graph=self.graph)

    def run(self, data: DataBundle, **kwargs):
        """
        run pipeline. to be extended for different schedulers
        """
        self.prepare()
        while self.is_active():
            for node in self.get_ready():
                node.execute(data, **kwargs)
                self.done(node)
