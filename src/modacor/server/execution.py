# SPDX-License-Identifier: BSD-3-Clause
# /usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from collections import defaultdict, deque
from graphlib import TopologicalSorter
from typing import Any

from modacor.runner.pipeline import Pipeline

__all__ = ["find_dirty_step_ids"]


def _collect_source_refs(value: Any, *, key_name: str | None, out: set[str]) -> None:
    if isinstance(value, str):
        if "::" in value:
            ref = value.split("::", 1)[0].strip()
            if ref:
                out.add(ref)
            return
        if key_name == "source_identifier":
            ref = value.strip()
            if ref:
                out.add(ref)
        return

    if isinstance(value, dict):
        for key, item in value.items():
            _collect_source_refs(item, key_name=str(key), out=out)
        return

    if isinstance(value, (list, tuple, set)):
        for item in value:
            _collect_source_refs(item, key_name=key_name, out=out)


def _step_source_refs(config: dict[str, Any]) -> set[str]:
    refs: set[str] = set()
    _collect_source_refs(config or {}, key_name=None, out=refs)
    return refs


def find_dirty_step_ids(pipeline: Pipeline, changed_sources: list[str] | set[str] | tuple[str, ...]) -> set[str]:
    """
    Determine dirty steps for a partial rerun.

    Dirty set = seed steps that reference any changed source + all descendants.
    """
    changed = {str(item) for item in changed_sources if str(item).strip()}
    if not changed:
        return set()

    id_by_node = {node: str(node.step_id) for node in pipeline.graph}

    seed_ids: set[str] = set()
    for node, sid in id_by_node.items():
        cfg = dict(getattr(node, "configuration", {}) or {})
        refs = _step_source_refs(cfg)
        if refs & changed:
            seed_ids.add(sid)

    if not seed_ids:
        return set()

    # Build reverse adjacency for descendant traversal.
    dependents: dict[str, set[str]] = defaultdict(set)
    for node, prereqs in pipeline.graph.items():
        node_id = id_by_node[node]
        for pre in prereqs:
            pre_id = id_by_node.get(pre)
            if pre_id is not None:
                dependents[pre_id].add(node_id)

    dirty_ids: set[str] = set(seed_ids)
    queue: deque[str] = deque(seed_ids)
    while queue:
        current = queue.popleft()
        for dep in dependents.get(current, set()):
            if dep not in dirty_ids:
                dirty_ids.add(dep)
                queue.append(dep)

    # Keep only existing ids (defensive), in topological order set form.
    topo_nodes = list(TopologicalSorter(pipeline.graph).static_order())
    ordered_dirty = [str(node.step_id) for node in topo_nodes if str(node.step_id) in dirty_ids]
    return set(ordered_dirty)
