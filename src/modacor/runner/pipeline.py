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

import json
from graphlib import TopologicalSorter

# quick hash at node-level (UI can show "config changed" without reading trace events)
from hashlib import sha256
from pathlib import Path
from typing import Any, Iterable, Mapping, Self

import yaml
from attrs import define, field

from modacor.debug.pipeline_tracer import tracer_event_to_datasets_payload

from ..dataclasses.process_step import ProcessStep
from ..dataclasses.trace_event import TraceEvent
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
    # Optional trace events collected during a run (step_id -> list of events)
    trace_events: dict[str, list[TraceEvent]] = field(factory=dict, repr=False)

    def __attrs_post_init__(self) -> None:
        super().__init__(graph=self.graph)

    # trace helpers, this helps to debug pipelines by storing trace events per step:
    def add_trace_event(self, event: TraceEvent) -> None:
        self.trace_events.setdefault(str(event.step_id), []).append(event)

    def clear_trace_events(self) -> None:
        self.trace_events.clear()

    # --------------------------------------------------------------------- #
    # Pipeline construction helpers
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
            short_title = module_data.get("short_title")

            # Normalize dependencies to strings as well
            requires_steps = {str(dep) for dep in requires_raw}

            # Resolve ProcessStep class via registry
            step_cls = registry.get(module_ref)

            # Pass the normalized string step_id into the ProcessStep
            step_instance: ProcessStep = step_cls(io_sources=None, io_sinks=None, step_id=step_id)
            step_instance.modify_config_by_dict(configuration)
            if short_title is not None:
                step_instance.short_title = str(short_title)

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
            step = step_cls(io_sources=None, io_sinks=None, step_id=step_id)
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

        e.g.:
        {
        "id": "FL",
        "label": "Divide by relative flux",
        "module": "Divide",
        "requires_steps": ["DC"],
        "config": {...},
        "trace_events": [
            {
            "step_id": "FL",
            "config_hash": "...",
            "datasets": {
                "sample.signal": {"diff": ["units","nan_signal"], "prev": {...}, "now": {...}}
            }
            }
        ]
        }

        Adds:
        - requires_steps per node (derived from graph prereqs)
        - optional trace events (if Pipeline.trace_events is populated)
        - config_hash per node (stable)
        """
        nodes: list[dict[str, Any]] = []
        edges: list[dict[str, str]] = []

        # Build a stable node set (keys + prereqs, just in case)
        all_nodes: set[ProcessStep] = set(self.graph.keys())
        for prereqs in self.graph.values():
            all_nodes |= set(prereqs)

        # map ProcessStep instance -> its step_id (as string)
        id_by_node: dict[ProcessStep, str] = {node: str(node.step_id) for node in all_nodes}

        # For stable output: sort by step_id
        def _node_sort_key(n: ProcessStep) -> str:
            return str(n.step_id)

        for node in sorted(all_nodes, key=_node_sort_key):
            sid = id_by_node[node]

            # Human label
            doc = getattr(node, "documentation", None)
            if doc is not None and getattr(doc, "calling_name", None):
                display_label = doc.calling_name
            else:
                display_label = type(node).__name__

            # prereqs list (sorted for spec stability)
            prereq_ids = sorted(id_by_node[p] for p in self.graph.get(node, set()))

            cfg = dict(getattr(node, "configuration", {}))

            node_spec: dict[str, Any] = {
                "id": sid,
                "label": display_label,
                "module": type(node).__name__,
                "config": cfg,
                "requires_steps": prereq_ids,
                "produced_outputs": sorted(getattr(node, "produced_outputs", {}).keys()),
            }
            if getattr(node, "short_title", None):
                node_spec["short_title"] = node.short_title

            cfg_json = json.dumps(node_spec["config"], sort_keys=True, default=str).encode("utf-8")
            node_spec["config_hash"] = sha256(cfg_json).hexdigest()

            if doc is not None:
                module_path = getattr(doc, "calling_module_path", None)
                node_spec["module_path"] = str(module_path) if module_path is not None else ""
                node_spec["version"] = getattr(doc, "calling_version", "") or ""
                calling_id = getattr(doc, "calling_id", None)
                if calling_id:
                    node_spec["module_id"] = calling_id

            # Attach trace events if present (kept lightweight)
            if sid in self.trace_events and self.trace_events[sid]:
                node_spec["trace_events"] = [ev.to_dict() for ev in self.trace_events[sid]]
            else:
                node_spec["trace_events"] = []

            nodes.append(node_spec)

        # Edges: self.graph maps node -> set(prerequisite nodes),
        # but visually we want edges prereq -> node.
        for node, prereqs in self.graph.items():
            target_id = id_by_node[node]
            for pre in prereqs:
                edges.append({"from": id_by_node[pre], "to": target_id})

        return {"name": self.name, "nodes": nodes, "edges": edges}

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
            label = f'{node["id"]}: {node["module"]}'
            short_title = node.get("short_title")
            if short_title:
                label = f"{label}\\n{short_title}"
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
            label = f'{node["id"]}: {node["module"]}'
            short_title = node.get("short_title")
            if short_title:
                label = f"{label}<br/>{short_title}"
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
            short_title = node.get("short_title")

            step_dict: dict[str, Any] = {
                "module": module_name,
            }
            if requires:
                step_dict["requires_steps"] = requires
            if cfg:
                step_dict["configuration"] = cfg
            if short_title:
                step_dict["short_title"] = short_title

            steps[sid] = step_dict

        yaml_obj = {
            "name": spec.get("name", self.name or "Unnamed Pipeline"),
            "steps": steps,
        }

        # sort_keys=False keeps insertion order, which follows node order in spec
        return yaml.safe_dump(yaml_obj, sort_keys=False)

    # --------------------------------------------------------------------- #
    # Trace events / run-time introspection
    # --------------------------------------------------------------------- #
    #
    # Summary
    # -------
    # Pipelines can optionally collect per-step TraceEvent records during execution.
    #
    # Design goals:
    # - Keep Pipeline execution fast and lightweight (no arrays stored).
    # - Keep trace events strictly step-local (each event describes only one executed node).
    # - Support UI rendering without requiring access to live ProcessingData.
    #
    # What is stored:
    # - Always: module metadata, prerequisite step_ids, and the step configuration used.
    # - Optionally: dataset "diff" payloads produced by PipelineTracer (units/dimensionality/NaNs/etc).
    # - Optionally: rendered, UI-ready snippets (HTML/Markdown/plain) for trace + config.
    #
    # What is NOT stored:
    # - No signal arrays, maps, or large objects (TraceEvent must stay JSON-friendly).
    # - No global or cross-step state (events can be attached/serialized independently).
    #
    # Integration pattern (typical runner / notebook):
    #   node(processing_data)
    #   tracer.after_step(node, processing_data)
    #   pipeline.attach_tracer_event(node, tracer,
    #                               include_rendered_trace=True,
    #                               include_rendered_config=True)
    #   pipeline.done(node)
    #
    # Export:
    # - Pipeline.to_spec() includes node-level config + optional trace_events per node,
    #   enabling graph viewers to show "what changed" as expandable panels.

    def attach_tracer_event(
        self,
        node: ProcessStep,
        tracer: Any | None,
        *,
        include_rendered_trace: bool = False,
        include_rendered_config: bool = False,
        rendered_format: str = "text/html",
    ) -> TraceEvent:
        """
        Create & attach a TraceEvent for `node`, using `tracer.events` if available.

        - Always attaches a TraceEvent so the graph UI can show config/module info.
        - Adds datasets diffs only if a matching tracer event for this step_id exists.
        """
        step_id = str(node.step_id)
        doc = getattr(node, "documentation", None)

        label = getattr(doc, "calling_name", "") if doc is not None else ""
        module_path = getattr(doc, "calling_module_path", "") if doc is not None else ""
        version = getattr(doc, "calling_version", "") if doc is not None else ""

        prereqs = tuple(sorted(str(p.step_id) for p in self.graph.get(node, set())))
        cfg = dict(getattr(node, "configuration", {}) or {})

        datasets: dict[str, Any] = {}

        def _html_escape(s: str) -> str:
            return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

        matched_ev: dict[str, Any] | None = None

        # Try to find the most recent tracer event for this step_id
        if tracer is not None:
            events = getattr(tracer, "events", None)
            if isinstance(events, list) and events:
                for ev in reversed(events):
                    if str(ev.get("step_id")) == step_id:
                        matched_ev = ev
                        datasets = tracer_event_to_datasets_payload(ev)
                        break

        duration_s: float | None = None
        if matched_ev is not None:
            d = matched_ev.get("duration_s", None)
            if isinstance(d, (int, float)):
                duration_s = float(d)

        messages: list[dict[str, Any]] = []

        # --- Rendered trace (STRICTLY step-local) ---
        if include_rendered_trace and matched_ev is not None:
            try:
                from modacor.debug.pipeline_tracer import (  # noqa: WPS433
                    MarkdownCssRenderer,
                    PlainUnicodeRenderer,
                    render_tracer_event,
                )

                if rendered_format in {"text/html", "text/markdown"}:
                    renderer = MarkdownCssRenderer(wrap_in_markdown_codeblock=False)
                    content = render_tracer_event(matched_ev, renderer=renderer)
                    fmt = "text/html"
                else:
                    renderer = PlainUnicodeRenderer(wrap_in_markdown_codeblock=False)
                    content = render_tracer_event(matched_ev, renderer=renderer)
                    fmt = "text/plain"

                messages.append(
                    {
                        "kind": "rendered_trace",
                        "title": "Trace",
                        "format": fmt,
                        "content": content,
                    }
                )
            except Exception as exc:
                messages.append(
                    {
                        "kind": "rendered_trace_error",
                        "title": "Trace",
                        "format": "text/plain",
                        "content": f"{exc!r}",
                    }
                )

        # --- Rendered config (STRICTLY step-local) ---
        if include_rendered_config:
            try:
                cfg_yaml = yaml.safe_dump(cfg, sort_keys=False)

                if rendered_format in {"text/html", "text/markdown"}:
                    # keep styling consistent with your CSS classes
                    # (don’t rely on MarkdownCssRenderer.codewrap here; we want to escape YAML)
                    content = "<pre class='mdc-pre mdc-config'>\n" + _html_escape(cfg_yaml) + "\n</pre>"
                    fmt = "text/html"
                else:
                    content = "Configuration:\n" + cfg_yaml
                    fmt = "text/plain"

                messages.append(
                    {
                        "kind": "rendered_config",
                        "title": "Configuration",
                        "format": fmt,
                        "content": content,
                    }
                )
            except Exception as exc:
                messages.append(
                    {
                        "kind": "rendered_config_error",
                        "title": "Configuration",
                        "format": "text/plain",
                        "content": f"{exc!r}",
                    }
                )

        event = TraceEvent(
            step_id=step_id,
            module=type(node).__name__,
            label=str(label or ""),
            module_path=str(module_path or ""),
            version=str(version or ""),
            requires_steps=prereqs,
            config=cfg,
            datasets=datasets,
            duration_s=duration_s,
            messages=messages,
        )

        self.add_trace_event(event)
        return event
