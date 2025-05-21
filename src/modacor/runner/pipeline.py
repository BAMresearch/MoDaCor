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
