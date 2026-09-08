# SPDX-License-Identifier: BSD-3-Clause
# /usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from collections import defaultdict, deque
from typing import Any

from modacor.dataclasses.process_step import ProcessStepDependencies, matches_processing_pattern
from modacor.runner.pipeline import Pipeline

__all__ = ["find_dirty_step_ids"]


def _node_id(node: Any) -> str:
    return str(getattr(node, "step_id", node))


def _node_dependency_contract(node: Any) -> ProcessStepDependencies:
    dependency_contract = getattr(node, "dependency_contract", None)
    if callable(dependency_contract):
        return dependency_contract()
    return ProcessStepDependencies(processing_reads={"*"}, processing_writes={"*"})


def find_dirty_step_ids(
    pipeline: Pipeline,
    changed_sources: list[str] | set[str] | tuple[str, ...] | None = None,
    changed_keys: list[str] | set[str] | tuple[str, ...] | None = None,
) -> set[str]:
    """
    Determine dirty steps for a partial rerun.

    Dirty set = seed steps whose dependency contract references any changed
    source or changed processing key + all descendants.
    """
    changed_source_set = {str(item).strip() for item in changed_sources or [] if str(item).strip()}
    changed_key_set = {str(item).strip() for item in changed_keys or [] if str(item).strip()}
    if not changed_source_set and not changed_key_set:
        return set()

    all_nodes = set(pipeline.graph.keys())
    for prereqs in pipeline.graph.values():
        all_nodes.update(prereqs)
    id_by_node = {node: _node_id(node) for node in all_nodes}

    seed_ids: set[str] = set()
    for node, sid in id_by_node.items():
        contract = _node_dependency_contract(node)
        processing_patterns = contract.processing_reads | contract.processing_writes

        source_match = bool(contract.source_refs & changed_source_set)
        key_match = any(matches_processing_pattern(changed_key, processing_patterns) for changed_key in changed_key_set)
        if source_match or key_match:
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
    topo_nodes = list(pipeline.static_order())
    ordered_dirty = [str(node.step_id) for node in topo_nodes if str(node.step_id) in dirty_ids]
    return set(ordered_dirty)
