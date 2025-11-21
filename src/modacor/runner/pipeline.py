# src/modacor/runner/pipeline.py
# -*- coding: utf-8 -*-
from __future__ import annotations

from graphlib import TopologicalSorter
from pathlib import Path
from typing import Iterable, Mapping, Self

import yaml
from attrs import define, field

from ..dataclasses.process_step import ProcessStep
from ..io.io_sources import IoSources  # noqa: F401  # reserved for future use
from .process_step_registry import DEFAULT_PROCESS_STEP_REGISTRY, ProcessStepRegistry

__all__ = ["Pipeline"]


@define
class Pipeline(TopologicalSorter):
    """
    Pipeline nodes are assumed to be of type ProcessStep.

    The underlying `graph` maps each node to the set of prerequisite nodes
    that must complete before it can run.
    """

    graph: dict[ProcessStep, set[ProcessStep]] = field(factory=dict)
    name: str = field(default="Unnamed Pipeline")

    def __attrs_post_init__(self) -> None:
        super().__init__(graph=self.graph)

    # --------------------------------------------------------------------- #
    # Construction helpers
    # --------------------------------------------------------------------- #

    @classmethod
    def from_yaml_file(
        cls,
        yaml_file: Path | str,
        registry: ProcessStepRegistry | None = None,
    ) -> "Pipeline":
        """
        Instantiate a Pipeline from a YAML configuration file.

        Parameters
        ----------
        yaml_file:
            Path to the YAML file.
        registry:
            Optional ProcessStepRegistry. If omitted, the global
            DEFAULT_PROCESS_STEP_REGISTRY is used.
        """
        yaml_path = Path(yaml_file)
        yaml_string = yaml_path.read_text(encoding="utf-8")
        return cls.from_yaml(yaml_string, registry=registry)

    @classmethod
    def from_yaml(
        cls,
        yaml_string: str,
        registry: ProcessStepRegistry | None = None,
    ) -> "Pipeline":
        """
        Instantiate a Pipeline from a YAML configuration string.

        Expected YAML schema (keyed by step_id):

        ```yaml
        name: my_pipeline
        steps:
          1:
            module: PoissonUncertainties
            requires_steps: []
            configuration: {...}

          "pu":
            module: PoissonUncertainties
            requires_steps: [1]        # may be int or string
            configuration: {...}
        ```

        Notes
        -----
        * The keys under `steps` (`1`, `"pu"`, etc.) are treated as the
          canonical `step_id`s and are normalized to `str`.
        * `requires_steps` entries can be ints or strings; they are also
          normalized to `str`.
        * If a `step_id` field is present inside a step, it must match
          the outer key (after string conversion), otherwise an error
          is raised to avoid silent mismatches.
        """
        yaml_obj = yaml.safe_load(yaml_string) or {}
        steps_cfg = yaml_obj.get("steps", {}) or {}

        registry = registry or DEFAULT_PROCESS_STEP_REGISTRY

        process_step_instances: dict[str, ProcessStep] = {}
        dependency_ids: dict[str, set[str]] = {}

        # First pass: instantiate steps and collect dependency ids
        for raw_step_key, module_data in steps_cfg.items():
            # Normalize outer key to string (allows numeric or string keys)
            step_id = str(raw_step_key)

            if not isinstance(module_data, dict):
                raise ValueError(
                    f"Step {step_id!r} must map to a mapping with 'module' and "
                    "'configuration' / 'requires_steps' fields."
                )

            # Optional inner step_id sanity check
            inner_step_id = module_data.get("step_id")
            if inner_step_id is not None and str(inner_step_id) != step_id:
                raise ValueError(
                    f"Step {step_id!r} has inner 'step_id' {inner_step_id!r} "
                    "which does not match the outer key. "
                    "Either omit the inner 'step_id' or make them identical."
                )

            try:
                module_ref = module_data["module"]
            except KeyError as exc:
                raise ValueError(f"Step {step_id!r} is missing required field 'module'.") from exc

            configuration = module_data.get("configuration") or {}
            requires_raw = module_data.get("requires_steps") or []

            # Normalize dependencies to strings as well
            requires_steps = {str(dep) for dep in requires_raw}

            # Resolve ProcessStep class via registry
            step_cls = registry.get(module_ref)

            # Pass the normalized string step_id into the ProcessStep
            step_instance: ProcessStep = step_cls(io_sources=None, step_id=step_id)
            step_instance.modify_config_by_dict(configuration)

            process_step_instances[step_id] = step_instance
            dependency_ids[step_id] = requires_steps

        # Second pass: validate dependencies
        all_defined_ids = set(process_step_instances.keys())
        for step_id, deps in dependency_ids.items():
            missing = deps - all_defined_ids
            if missing:
                missing_str = ", ".join(sorted(missing))
                raise ValueError(
                    f"Step {step_id!r} requires unknown steps {missing_str}. "
                    "Check `steps` keys and `requires_steps` in the YAML."
                )

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

        Parameters
        ----------
        graph_dict:
            Mapping of node -> iterable of prerequisite nodes.

        Notes
        -----
        This is a low-level constructor mainly intended for internal use or
        tests. Normal users should prefer `from_yaml_file` or `from_yaml`.
        """
        graph: dict[ProcessStep, set[ProcessStep]] = {node: set(deps) for node, deps in graph_dict.items()}
        return cls(name=name or "Unnamed Pipeline", graph=graph)

    # --------------------------------------------------------------------- #
    # Graph mutation helpers
    # --------------------------------------------------------------------- #

    def _reinitialize(self) -> None:
        """Recreate the underlying TopologicalSorter with the current graph."""
        super().__init__(graph=self.graph)

    def add_incoming_branch(self, branch: Self, branching_node: ProcessStep) -> Self:
        """
        Add a pipeline as a branch whose outcome shall be combined with
        the existing pipeline at `branching_node`.

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

    def add_outgoing_branch(self, branch: Self, branching_node: ProcessStep) -> Self:
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

    # --------------------------------------------------------------------- #
    # Execution
    # --------------------------------------------------------------------- #

    def run(self, **kwargs) -> None:
        """
        Run pipeline with simple topological scheduling.

        Any keyword arguments are passed through to `ProcessStep.execute`.
        """
        self.prepare()
        while self.is_active():
            for node in self.get_ready():
                node.execute(**kwargs)
                self.done(node)
