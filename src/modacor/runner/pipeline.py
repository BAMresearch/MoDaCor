# src/modacor/runner/pipeline.py
# # -*- coding: utf-8 -*-
from __future__ import annotations

import importlib
import re
from graphlib import TopologicalSorter
from pathlib import Path
from typing import Iterable, Mapping, Self

import yaml
from attrs import define, field
from attrs import validators as v

from ..dataclasses.process_step import ProcessStep
from ..io.io_sources import IoSources

__all__ = ["Pipeline"]


def _find_python_module(modacor_module_name):
    "convert from PascalCase to snake_case"
    submodule = re.sub(r"(?<!^)([A-Z])", r"_\1", modacor_module_name)
    return submodule.lower()


@define
class Pipeline(TopologicalSorter):
    """
    Pipeline nodes are assumed to be of type ProcessStep
    """

    graph: dict[ProcessStep, set[ProcessStep]] = field(factory=dict)
    name: str = field(default="Unnamed Pipeline")

    def __attrs_post_init__(self):
        super().__init__(graph=self.graph)

    @classmethod
    def from_yaml_file(cls, yaml_file: Path):
        """
        Instantiate a Pipeline from a yaml configuration file.
        """
        with open(yaml_file, "r", encoding="utf-8") as f:
            yaml_string = f.read()
        return cls.from_yaml(yaml_string)

    @classmethod
    def from_yaml(cls, yaml_string: str):
        """
        Instantiate a Pipeline from a yaml configuration file.
        """

        yaml_obj = yaml.safe_load(yaml_string)

        steps_cfg = yaml_obj.get("steps", {}) or {}

        process_step_instances: dict[str, ProcessStep] = {}
        dependency_ids: dict[str, set[str]] = {}

        # Let's make sure we have what we need.
        all_defined_ids = set(process_step_instances.keys())

        for step_id, deps in dependency_ids.items():
            missing = deps - all_defined_ids
            if missing:
                missing_str = ", ".join(map(str, sorted(missing)))
                raise ValueError(
                    f"Step {step_id!r} requires unknown steps {missing_str}. "
                    "Check `step_id` and `requires_steps` in the YAML; "
                    "you may have duplicate step names or a typo."
                )

        # check complete - now instantiate all steps
        for _step_name, module_data in steps_cfg.items():
            step_id = module_data["step_id"]
            module_ref = module_data["module"]
            configuration = module_data.get("configuration") or {}
            requires_steps = module_data.get("requires_steps") or []

            module = importlib.import_module(f"modacor.modules.base_modules.{_find_python_module(module_ref)}")
            step_cls = getattr(module, module_ref)
            step_instance: ProcessStep = step_cls(io_sources=None, step_id=step_id)
            step_instance.modify_config_by_dict(configuration)

            process_step_instances[step_id] = step_instance
            dependency_ids[step_id] = set(requires_steps)

        # Translate step_id graph into ProcessStep graph
        graph: dict[ProcessStep, set[ProcessStep]] = {}
        for step_id, deps in dependency_ids.items():
            graph[process_step_instances[step_id]] = {process_step_instances[dep_id] for dep_id in deps}

        name = yaml_obj.get("name", "Unnamed Pipeline")
        return cls(name=name, graph=graph)

    @classmethod
    def from_dict(
        cls,
        graph_dict: Mapping[ProcessStep, Iterable[ProcessStep]],
        name: str = "",
    ) -> "Pipeline":
        """
        Instantiate a Pipeline from a mapping.

        The mapping must be of the form: node -> iterable of prerequisite nodes.
        """
        graph: dict[ProcessStep, set[ProcessStep]] = {node: set(deps) for node, deps in graph_dict.items()}
        return cls(name=name or "Unnamed Pipeline", graph=graph)

    def _reinitialize(self) -> None:
        """Recreate the underlying TopologicalSorter with the current graph."""
        super().__init__(graph=self.graph)

    def add_incoming_branch(self, branch: Self, branching_node):
        """
        Add a pipeline as a branch whose outcome shall be combined the existing pipeline at branching_node.

        This assumes that the branch to be added has a single exit point.

        """
        ordered_branch = list(branch.static_order())
        if not ordered_branch:
            return self

        last_node = ordered_branch[-1]
        self.graph.setdefault(branching_node, set()).add(last_node)
        # Add the rest of the graph
        self.graph |= branch.graph
        self._reinitialize()
        return self

    def add_outgoing_branch(self, branch: Self, branching_node):
        """
        Add a pipeline as a branch whose input is based on the existing pipeline.

        This assumes that the branch to be added has a single entry point.

        """
        ordered_branch = list(branch.static_order())
        if not ordered_branch:
            return self

        first_node = ordered_branch[0]
        branch.graph.setdefault(first_node, set()).add(branching_node)
        self.graph |= branch.graph
        self._reinitialize()
        return self

    def run(self, **kwargs):
        """
        run pipeline with simple scheduling.
        """
        self.prepare()
        while self.is_active():
            for node in self.get_ready():
                node.execute(**kwargs)
                self.done(node)
