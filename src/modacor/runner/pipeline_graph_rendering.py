"""Render flat pipeline specs with optional repeated-block grouping."""

from __future__ import annotations

__all__ = ["render_pipeline_dot", "render_pipeline_drawio", "render_pipeline_mermaid"]

import html
import json
import subprocess
import xml.etree.ElementTree as ET
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


def _graphviz_layout_json(dot_source: str, *, dot_executable: str) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            [dot_executable, "-Tjson"],
            input=dot_source,
            text=True,
            capture_output=True,
            check=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"Graphviz executable {dot_executable!r} was not found; install Graphviz to export draw.io XML."
        ) from exc
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "").strip()
        suffix = f": {detail}" if detail else "."
        raise RuntimeError(f"Graphviz failed while laying out the pipeline{suffix}") from exc
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("Graphviz returned invalid JSON layout data.") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("Graphviz returned an invalid pipeline layout.")
    return payload


def _graphviz_box(value: Any, *, label: str) -> tuple[float, float, float, float]:
    try:
        coordinates = tuple(float(part) for part in str(value).split(","))
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"Graphviz returned an invalid {label} bounding box: {value!r}.") from exc
    if len(coordinates) != 4:
        raise RuntimeError(f"Graphviz returned an invalid {label} bounding box: {value!r}.")
    return coordinates


def _drawio_geometry(
    graphviz_object: dict[str, Any],
    *,
    graph_box: tuple[float, float, float, float],
    scale: float,
    margin: float,
) -> dict[str, float]:
    graph_x0, _, _, graph_y1 = graph_box
    if "bb" in graphviz_object:
        x0, y0, x1, y1 = _graphviz_box(graphviz_object["bb"], label=str(graphviz_object.get("name", "cluster")))
        return {
            "x": margin + (x0 - graph_x0) * scale,
            "y": margin + (graph_y1 - y1) * scale,
            "width": (x1 - x0) * scale,
            "height": (y1 - y0) * scale,
        }

    try:
        center_x, center_y = (float(part) for part in str(graphviz_object["pos"]).split(",")[:2])
        width = float(graphviz_object["width"]) * 72.0 * scale
        height = float(graphviz_object["height"]) * 72.0 * scale
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(f"Graphviz returned invalid node geometry for {graphviz_object.get('name')!r}.") from exc
    return {
        "x": margin + (center_x - graph_x0) * scale - width / 2.0,
        "y": margin + (graph_y1 - center_y) * scale - height / 2.0,
        "width": width,
        "height": height,
    }


def _relative_geometry(geometry: dict[str, float], parent: dict[str, float]) -> dict[str, float]:
    return {
        "x": geometry["x"] - parent["x"],
        "y": geometry["y"] - parent["y"],
        "width": geometry["width"],
        "height": geometry["height"],
    }


def _append_mx_geometry(cell: ET.Element, geometry: dict[str, float], *, relative: bool = False) -> None:
    attributes = {"as": "geometry"}
    if relative:
        attributes["relative"] = "1"
    else:
        attributes.update(
            {
                "x": f'{geometry["x"]:.2f}',
                "y": f'{geometry["y"]:.2f}',
                "width": f'{geometry["width"]:.2f}',
                "height": f'{geometry["height"]:.2f}',
            }
        )
    ET.SubElement(cell, "mxGeometry", attributes)


def _drawio_node_value(node: dict[str, Any]) -> str:
    step_id = html.escape(str(node["id"]))
    module = html.escape(str(node["module"]))
    value = f"<b>{step_id}</b><br>{module}"
    if node.get("short_title"):
        value = f'{value}<br><i>{html.escape(str(node["short_title"]))}</i>'
    return value


def render_pipeline_drawio(
    spec: dict[str, Any],
    *,
    direction: str = "LR",
    group_step_blocks: bool = True,
    dot_executable: str = "dot",
    pixels_per_inch: float = 96.0,
) -> str:
    """Render a pipeline as editable, uncompressed draw.io XML.

    Graphviz supplies node and cluster geometry. Expanded ``step_blocks`` are
    represented as nested editable containers, while every pipeline node and
    dependency remains an individual draw.io cell.
    """

    if pixels_per_inch <= 0:
        raise ValueError("pixels_per_inch must be positive.")
    dot_lines = render_pipeline_dot(
        spec,
        direction=direction,
        group_step_blocks=group_step_blocks,
    ).splitlines()
    dot_lines.insert(1, '  node [shape=box, style="rounded"];')
    layout = _graphviz_layout_json("\n".join(dot_lines), dot_executable=dot_executable)
    graph_box = _graphviz_box(layout.get("bb"), label="graph")
    scale = float(pixels_per_inch) / 72.0
    margin = 24.0
    graph_width = (graph_box[2] - graph_box[0]) * scale
    graph_height = (graph_box[3] - graph_box[1]) * scale
    objects = {
        str(item["name"]): item for item in layout.get("objects", []) if isinstance(item, dict) and "name" in item
    }

    node_geometry: dict[str, dict[str, float]] = {}
    for node in spec["nodes"]:
        step_id = str(node["id"])
        if step_id not in objects:
            raise RuntimeError(f"Graphviz layout is missing pipeline node {step_id!r}.")
        node_geometry[step_id] = _drawio_geometry(objects[step_id], graph_box=graph_box, scale=scale, margin=margin)

    mxfile = ET.Element(
        "mxfile",
        {
            "host": "app.diagrams.net",
            "agent": "MoDaCor pipeline draw.io exporter",
            "compressed": "false",
        },
    )
    diagram = ET.SubElement(mxfile, "diagram", {"id": "modacor-pipeline", "name": str(spec["name"])})
    model = ET.SubElement(
        diagram,
        "mxGraphModel",
        {
            "dx": str(int(graph_width + 2 * margin)),
            "dy": str(int(graph_height + 2 * margin)),
            "grid": "1",
            "gridSize": "10",
            "guides": "1",
            "tooltips": "1",
            "connect": "1",
            "arrows": "1",
            "fold": "1",
            "page": "1",
            "pageScale": "1",
            "pageWidth": str(int(graph_width + 2 * margin)),
            "pageHeight": str(int(graph_height + 2 * margin)),
            "math": "1",
            "shadow": "0",
        },
    )
    root = ET.SubElement(model, "root")
    ET.SubElement(root, "mxCell", {"id": "0"})
    ET.SubElement(root, "mxCell", {"id": "1", "parent": "0"})

    grouped_parent: dict[str, tuple[str, dict[str, float]]] = {}
    if group_step_blocks:
        _, blocks = _group_nodes(spec["nodes"])
        block_style = (
            "rounded=1;container=1;collapsible=0;recursiveResize=0;whiteSpace=wrap;html=1;"
            "verticalAlign=top;spacingTop=5;fontStyle=1;fontSize=14;"
            "fillColor=#f5f5f5;strokeColor=#666666;strokeWidth=1.5;"
        )
        item_style = (
            "rounded=1;container=1;collapsible=0;recursiveResize=0;whiteSpace=wrap;html=1;"
            "verticalAlign=top;spacingTop=4;fontStyle=1;fontSize=12;"
            "fillColor=#ffffff;strokeColor=#999999;strokeWidth=1.2;"
        )
        for block_index, (block_name, items) in enumerate(blocks):
            block_object_name = f"cluster_block_{block_index}"
            if block_object_name not in objects:
                raise RuntimeError(f"Graphviz layout is missing block container {block_name!r}.")
            block_geometry = _drawio_geometry(
                objects[block_object_name], graph_box=graph_box, scale=scale, margin=margin
            )
            block_cell_id = f"block-{block_index}"
            block_cell = ET.SubElement(
                root,
                "mxCell",
                {
                    "id": block_cell_id,
                    "value": f"{html.escape(block_name)} (for_each)",
                    "style": block_style,
                    "vertex": "1",
                    "parent": "1",
                },
            )
            _append_mx_geometry(block_cell, block_geometry)

            for item_index, (item_name, item_nodes) in enumerate(items):
                item_object_name = f"cluster_block_{block_index}_item_{item_index}"
                if item_object_name not in objects:
                    raise RuntimeError(f"Graphviz layout is missing item container {block_name}.{item_name!s}.")
                item_geometry = _drawio_geometry(
                    objects[item_object_name], graph_box=graph_box, scale=scale, margin=margin
                )
                item_cell_id = f"block-{block_index}-item-{item_index}"
                item_cell = ET.SubElement(
                    root,
                    "mxCell",
                    {
                        "id": item_cell_id,
                        "value": html.escape(item_name),
                        "style": item_style,
                        "vertex": "1",
                        "parent": block_cell_id,
                    },
                )
                _append_mx_geometry(item_cell, _relative_geometry(item_geometry, block_geometry))
                for node in item_nodes:
                    grouped_parent[str(node["id"])] = (item_cell_id, item_geometry)

    node_cell_ids: dict[str, str] = {}
    node_style = (
        "rounded=1;whiteSpace=wrap;html=1;arcSize=12;fillColor=#ffffff;"
        "strokeColor=#333333;strokeWidth=1.3;fontFamily=Helvetica;fontSize=12;"
        "align=center;verticalAlign=middle;"
    )
    for index, node in enumerate(spec["nodes"], start=1):
        step_id = str(node["id"])
        cell_id = f"node-{index}"
        node_cell_ids[step_id] = cell_id
        parent_id, parent_geometry = grouped_parent.get(step_id, ("1", None))
        geometry = node_geometry[step_id]
        if parent_geometry is not None:
            geometry = _relative_geometry(geometry, parent_geometry)
        attributes = {
            "id": cell_id,
            "value": _drawio_node_value(node),
            "style": node_style,
            "vertex": "1",
            "parent": parent_id,
            "modacorStepId": step_id,
            "modacorModule": str(node["module"]),
        }
        origin = _origin(node)
        if origin is not None:
            attributes["modacorBlock"] = str(origin["block"])
            attributes["modacorItem"] = str(origin["item"])
        cell = ET.SubElement(root, "mxCell", attributes)
        _append_mx_geometry(cell, geometry)

    edge_style = (
        "edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;"
        "strokeColor=#666666;strokeWidth=1.2;endArrow=block;endFill=1;"
    )
    for index, edge in enumerate(spec["edges"], start=1):
        source = str(edge["from"])
        target = str(edge["to"])
        cell = ET.SubElement(
            root,
            "mxCell",
            {
                "id": f"edge-{index}",
                "style": edge_style,
                "edge": "1",
                "parent": "1",
                "source": node_cell_ids[source],
                "target": node_cell_ids[target],
            },
        )
        _append_mx_geometry(cell, {}, relative=True)

    ET.indent(mxfile, space="  ")
    return ET.tostring(mxfile, encoding="unicode") + "\n"


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
