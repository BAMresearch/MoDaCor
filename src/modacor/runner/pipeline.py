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
    name: str = field(default="Unnamed Pipeline")
    graph: dict = field(factory=dict)

    def __attrs_post_init__(self):
        super().__init__(graph=self.graph)

    @classmethod
    def from_json(cls, path_to_json: Path):
        # functionality postponed
        return cls(name="dummy")

    @classmethod
    def from_dict(cls, graph_dict: dict, name=""):
        return cls(name=name, graph=graph_dict)

    def add_branch(self, branch_graph: dict, branching_node):
        """
        add a pipeline as a branch on an existing pipeline, using the inherited add method

        """
        pipeline_to_add = Pipeline(graph=branch_graph)
        pipeline_to_add_ordered = [*pipeline_to_add.static_order()]
        self.add(branching_node, *pipeline_to_add_ordered)
        self.graph = {**self.graph, **branch_graph}
        self.graph[branching_node].update({pipeline_to_add_ordered[-1]})

