"""Render flat pipeline specs with optional repeated-block grouping."""

from __future__ import annotations

__all__ = ["render_pipeline_dot", "render_pipeline_mermaid"]

from collections import defaultdict
from typing import Any, Iterable


def _node_label(node: dict[str, Any], *, line_break: str) -> str:
    label = f'{node["id"]}: {node["module"]}'
    short_title = node.get("short_title")
    if short_title:
        label = f"{label}{line_break}{short_title}"
    return label


def _origin(node: dict[str, Any]) -> dict[str, Any] | None:
    origin = node.get("origin")
    if not isinstance(origin, dict):
        return None
    required = {"block", "item", "local_step", "block_index", "item_index", "local_step_index"}
    if not required.issubset(origin):
        return None
    return origin


def _group_nodes(
    nodes: Iterable[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[tuple[str, list[tuple[str, list[dict[str, Any]]]]]]]:
    ordinary: list[dict[str, Any]] = []
    grouped: dict[tuple[int, str], dict[tuple[int, str], list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for node in nodes:
        origin = _origin(node)
        if origin is None:
            ordinary.append(node)
            continue
        block_key = (int(origin["block_index"]), str(origin["block"]))
        item_key = (int(origin["item_index"]), str(origin["item"]))
        grouped[block_key][item_key].append(node)

    blocks: list[tuple[str, list[tuple[str, list[dict[str, Any]]]]]] = []
    for (_, block_name), items in sorted(grouped.items()):
        rendered_items: list[tuple[str, list[dict[str, Any]]]] = []
        for (_, item_name), item_nodes in sorted(items.items()):

            def local_step_order(node: dict[str, Any]) -> tuple[int, str]:
                origin = _origin(node)
                if origin is None:  # pragma: no cover - guaranteed by grouping above
                    raise RuntimeError("Grouped pipeline node lost its origin metadata.")
                return int(origin["local_step_index"]), str(node["id"])

            item_nodes.sort(key=local_step_order)
            rendered_items.append((item_name, item_nodes))
        blocks.append((block_name, rendered_items))
    return ordinary, blocks


def _dot_escape(value: Any) -> str:
    return str(value).replace("\\", "\\\\").replace('"', '\\"')


def _dot_node(node: dict[str, Any], *, indent: str) -> str:
    node_id = _dot_escape(node["id"])
    label = _dot_escape(_node_label(node, line_break="\n")).replace("\n", "\\n")
    return f'{indent}"{node_id}" [label="{label}"];'


def render_pipeline_dot(
    spec: dict[str, Any],
    *,
    direction: str = "LR",
    group_step_blocks: bool = True,
) -> str:
    """Render a pipeline spec as Graphviz DOT."""

    ordinary, blocks = _group_nodes(spec["nodes"]) if group_step_blocks else (spec["nodes"], [])
    lines = [
        f'digraph "{_dot_escape(spec["name"])}" {{',
        f"  rankdir={direction};",
    ]
    if blocks:
        # Required for rank constraints spanning nested item clusters.
        lines.append("  newrank=true;")

    for node in ordinary:
        lines.append(_dot_node(node, indent="  "))

    for block_index, (block_name, items) in enumerate(blocks):
        lines.extend(
            [
                f'  subgraph "cluster_block_{block_index}" {{',
                f'    label="{_dot_escape(block_name)} (for_each)";',
            ]
        )
        stages: dict[str, list[str]] = defaultdict(list)
        for item_index, (item_name, item_nodes) in enumerate(items):
            lines.extend(
                [
                    f'    subgraph "cluster_block_{block_index}_item_{item_index}" {{',
                    f'      label="{_dot_escape(item_name)}";',
                ]
            )
            for node in item_nodes:
                lines.append(_dot_node(node, indent="      "))
                origin = _origin(node)
                if origin is not None:
                    stages[str(origin["local_step"])].append(str(node["id"]))
            lines.append("    }")
        for stage_nodes in stages.values():
            if len(stage_nodes) > 1:
                quoted_ids = "; ".join(f'"{_dot_escape(node_id)}"' for node_id in stage_nodes)
                lines.append(f"    {{ rank=same; {quoted_ids}; }}")
        lines.append("  }")

    for edge in spec["edges"]:
        source = _dot_escape(edge["from"])
        target = _dot_escape(edge["to"])
        lines.append(f'  "{source}" -> "{target}";')

    lines.append("}")
    return "\n".join(lines)


def _mermaid_escape(value: Any) -> str:
    return str(value).replace('"', "&quot;")


def _mermaid_node(node: dict[str, Any], renderer_id: str, *, indent: str) -> str:
    label = _mermaid_escape(_node_label(node, line_break="<br/>"))
    return f'{indent}{renderer_id}["{label}"]'


def _mermaid_direction(direction: str) -> str:
    # Mermaid accepts TD as a top-level alias in some versions, but nested
    # subgraphs consistently require the canonical TB spelling.
    return "TB" if direction.upper() == "TD" else direction.upper()


def render_pipeline_mermaid(
    spec: dict[str, Any],
    *,
    direction: str = "LR",
    group_step_blocks: bool = True,
) -> str:
    """Render a pipeline spec as a Mermaid flowchart."""

    direction = _mermaid_direction(direction)
    nodes = spec["nodes"]
    id_map = {node["id"]: f"node_{index}" for index, node in enumerate(nodes)}
    lines = [f"flowchart {direction}"]
    ordinary, blocks = _group_nodes(nodes) if group_step_blocks else (nodes, [])

    for node in ordinary:
        lines.append(_mermaid_node(node, id_map[node["id"]], indent="    "))

    lane_direction = "TB" if direction in {"LR", "RL"} else "LR"
    for block_index, (block_name, items) in enumerate(blocks):
        lines.append(f'    subgraph block_{block_index}["{_mermaid_escape(block_name)} (for_each)"]')
        lines.append(f"        direction {lane_direction}")
        for item_index, (item_name, item_nodes) in enumerate(items):
            lines.append(f'        subgraph block_{block_index}_item_{item_index}["{_mermaid_escape(item_name)}"]')
            lines.append(f"            direction {direction}")
            for node in item_nodes:
                lines.append(_mermaid_node(node, id_map[node["id"]], indent="            "))
            lines.append("        end")
        lines.append("    end")

    for edge in spec["edges"]:
        lines.append(f'    {id_map[edge["from"]]} --> {id_map[edge["to"]]}')

    return "\n".join(lines)
