# SPDX-License-Identifier: BSD-3-Clause
# /usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

__coding__ = "utf-8"
__authors__ = ["Anja Hörmann", "Brian R. Pauw"]  # add names to the list as appropriate
__copyright__ = "Copyright 2025, The MoDaCor team"
__date__ = "22/11/2025"
__status__ = "Development"  # "Development", "Production"
# end of header and standard imports

from graphlib import TopologicalSorter
from pathlib import Path
from typing import Any, Iterable, Mapping, Self

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

    # importer for future use of web tools for graphing pipelines:
    @classmethod
    def from_spec(
        cls,
        spec: dict,
        registry: ProcessStepRegistry | None = None,
    ) -> "Pipeline":
        """
        Build a Pipeline from a graph spec of the shape produced by `to_spec`.

        Expected shape:
        {
        "name": "...",
        "nodes": [
            {"id": "...", "module": "...", "config": {...}},
            ...
        ],
        "edges": [
            {"from": "...", "to": "..."},
            ...
        ],
        }
        """
        registry = registry or DEFAULT_PROCESS_STEP_REGISTRY

        # 1) Build ProcessStep instances
        process_step_instances: dict[str, ProcessStep] = {}
        for node in spec.get("nodes", []):
            step_id = str(node["id"])
            module_name = node["module"]
            config = node.get("config", {}) or {}

            step_cls = registry.get(module_name)
            step = step_cls(io_sources=None, step_id=step_id)
            step.modify_config_by_dict(config)

            process_step_instances[step_id] = step

        # 2) Build prerequisite sets from edges
        #    edges are from -> to, but TopologicalSorter wants node -> prerequisites
        prereqs: dict[str, set[str]] = {sid: set() for sid in process_step_instances}
        for edge in spec.get("edges", []):
            src = str(edge["from"])
            dst = str(edge["to"])
            if src not in prereqs or dst not in prereqs:
                raise ValueError(f"Edge refers to unknown node: {src!r} -> {dst!r}")
            prereqs[dst].add(src)

        # 3) Convert to ProcessStep graph
        graph: dict[ProcessStep, set[ProcessStep]] = {}
        for sid, deps in prereqs.items():
            graph[process_step_instances[sid]] = {process_step_instances[dep_id] for dep_id in deps}

        name = spec.get("name", "Unnamed Pipeline")
        return cls(name=name, graph=graph)

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

    # --------------------------------------------------------------------- #
    # Introspection / visualization helpers
    # --------------------------------------------------------------------- #

    def to_spec(self) -> dict[str, Any]:
        """
        Export the pipeline to a JSON-serializable graph spec.

        Returns
        -------
        dict with structure:
        {
          "name": "<pipeline_name>",
          "nodes": [
            {
              "id": "<step_id>",
              "label": "<human readable label>",
              "module": "<ProcessStep class name>",
              "module_path": "<path to module>" or "",
              "version": "<module version>" or "",
              "config": {...}  # current configuration dict
            },
            ...
          ],
          "edges": [
            {"from": "<source_step_id>", "to": "<target_step_id>"},
            ...
          ],
        }
        """
        nodes: list[dict[str, Any]] = []
        edges: list[dict[str, str]] = []

        # map ProcessStep instance -> its step_id (as string)
        id_by_node: dict[ProcessStep, str] = {}

        for node in self.graph.keys():
            sid = str(node.step_id)
            id_by_node[node] = sid

            # Build a nice label using ProcessStepDescriber if available
            doc = getattr(node, "documentation", None)
            if doc is not None and getattr(doc, "calling_name", None):
                display_label = doc.calling_name
            else:
                display_label = type(node).__name__

            node_spec: dict[str, Any] = {
                "id": sid,
                "label": display_label,
                "module": type(node).__name__,
                "config": dict(getattr(node, "configuration", {})),
            }

            if doc is not None:
                module_path = getattr(doc, "calling_module_path", None)
                node_spec["module_path"] = str(module_path) if module_path is not None else ""
                node_spec["version"] = getattr(doc, "calling_version", "") or ""

            nodes.append(node_spec)

        # Edges: self.graph maps node -> set(prerequisite nodes),
        # but visually we want edges prereq -> node.
        for node, prereqs in self.graph.items():
            target_id = id_by_node[node]
            for pre in prereqs:
                edges.append(
                    {
                        "from": id_by_node[pre],
                        "to": target_id,
                    }
                )

        return {
            "name": self.name,
            "nodes": nodes,
            "edges": edges,
        }

    def to_dot(self) -> str:
        """
        Export the pipeline as a Graphviz DOT string for visualization.

        Nodes are labeled with "<step_id>: <calling_name/module_name>".
        """
        spec = self.to_spec()
        lines: list[str] = [
            f'digraph "{spec["name"]}" {{',
            "  rankdir=LR;",  # left-to-right layout; change to TB for top-to-bottom
        ]

        # Nodes
        for node in spec["nodes"]:
            nid = node["id"]
            # Show both id and label so it's easy to match YAML <-> graph
            label = f'{node["id"]}: {node["label"]}'
            esc_label = label.replace('"', '\\"')
            lines.append(f'  "{nid}" [label="{esc_label}"];')  # noqa: E702, E231

        # Edges
        for edge in spec["edges"]:
            lines.append(f'  "{edge["from"]}" -> "{edge["to"]}";')  # noqa: E702, E231

        lines.append("}")
        return "\n".join(lines)

    def to_mermaid(self, direction: str = "LR") -> str:
        """
        Export the pipeline as a Mermaid flowchart definition.

        Parameters
        ----------
        direction:
            Mermaid direction: "LR" (left-right), "TB" (top-bottom), etc.
        """
        spec = self.to_spec()

        # Mermaid node IDs must be simple identifiers (no spaces, quotes, etc.).
        # We'll generate safe IDs but keep the original step_id visible in the label.
        def sanitize(node_id: str) -> str:
            return "".join(c if (c.isalnum() or c == "_") else "_" for c in node_id)

        id_map: dict[str, str] = {}
        for node in spec["nodes"]:
            raw = str(node["id"])
            id_map[node["id"]] = sanitize(raw)

        lines: list[str] = [f"flowchart {direction}"]

        # Nodes
        for node in spec["nodes"]:
            nid = id_map[node["id"]]
            label = f'{node["id"]}: {node["label"]}'
            esc_label = label.replace('"', '\\"')
            lines.append(f'    {nid}["{esc_label}"]')

        # Edges
        for edge in spec["edges"]:
            src = id_map[edge["from"]]
            dst = id_map[edge["to"]]
            lines.append(f"    {src} --> {dst}")

        return "\n".join(lines)

    # in case we used to and from spec to modify the pipeline, we can
    # store the new pipeline back to yaml
    def to_yaml(self) -> str:
        """
        Export the pipeline to a YAML string using the same schema
        that `from_yaml` expects (keyed by step_id).

        The result looks like:

        ```yaml
        name: my_pipeline
        steps:
          1:
            module: SomeStep
            requires_steps: []
            configuration: {...}
          "pu":
            module: OtherStep
            requires_steps: [1]
            configuration: {...}
        ```
        """
        spec = self.to_spec()

        # Build steps mapping keyed by step_id
        steps: dict[str, dict[str, Any]] = {}

        # Pre-compute requires_steps per node from edges
        requires_map: dict[str, list[str]] = {n["id"]: [] for n in spec["nodes"]}
        for edge in spec["edges"]:
            src = str(edge["from"])
            dst = str(edge["to"])
            # edge: src -> dst  =>  dst.requires_steps includes src
            if dst in requires_map:
                requires_map[dst].append(src)
            else:
                requires_map[dst] = [src]

        for node in spec["nodes"]:
            sid = str(node["id"])
            module_name = node["module"]
            cfg = node.get("config", {}) or {}
            requires = requires_map.get(sid, [])

            step_dict: dict[str, Any] = {
                "module": module_name,
            }
            if requires:
                step_dict["requires_steps"] = requires
            if cfg:
                step_dict["configuration"] = cfg

            steps[sid] = step_dict

        yaml_obj = {
            "name": spec.get("name", self.name or "Unnamed Pipeline"),
            "steps": steps,
        }

        # sort_keys=False keeps insertion order, which follows node order in spec
        return yaml.safe_dump(yaml_obj, sort_keys=False)
