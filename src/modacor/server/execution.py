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


def _normalize_processing_keys(value: Any) -> list[str]:
    if value is None:
        return ["*"]
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value]
    return ["*"]


def _step_processing_key_patterns(config: dict[str, Any]) -> tuple[set[str], set[str]]:
    """
    Return (reads, writes) key patterns like:
      - sample.*           (bundle-level pattern)
      - sample.signal      (specific basedata key)
      - *                  (conservative wildcard)
    """
    cfg = config or {}
    reads: set[str] = set()
    writes: set[str] = set()

    with_keys = _normalize_processing_keys(cfg.get("with_processing_keys"))
    for key in with_keys:
        reads.add(f"{key}.*" if key != "*" else "*")
        writes.add(f"{key}.*" if key != "*" else "*")

    # AppendProcessingData-style config
    processing_key = cfg.get("processing_key")
    databundle_output_key = cfg.get("databundle_output_key")
    if isinstance(processing_key, str) and processing_key.strip():
        pkey = processing_key.strip()
        if isinstance(databundle_output_key, str) and databundle_output_key.strip():
            writes.add(f"{pkey}.{databundle_output_key.strip()}")
        else:
            writes.add(f"{pkey}.*")

    # Explicit output processing key (for modules that write to a dedicated key)
    output_processing_key = cfg.get("output_processing_key")
    if isinstance(output_processing_key, str) and output_processing_key.strip():
        writes.add(f"{output_processing_key.strip()}.*")

    # If no signal at all, keep conservative wildcard so we do not miss dependencies.
    if not reads:
        reads.add("*")
    if not writes:
        writes.add("*")

    return reads, writes


def _match_changed_key(changed_key: str, patterns: set[str]) -> bool:
    if "*" in patterns:
        return True
    for pattern in patterns:
        if pattern.endswith(".*"):
            prefix = pattern[:-2]
            if changed_key == prefix or changed_key.startswith(prefix + "."):
                return True
        if changed_key == pattern:
            return True
    return False


def find_dirty_step_ids(
    pipeline: Pipeline,
    changed_sources: list[str] | set[str] | tuple[str, ...] | None = None,
    changed_keys: list[str] | set[str] | tuple[str, ...] | None = None,
) -> set[str]:
    """
    Determine dirty steps for a partial rerun.

    Dirty set = seed steps that reference any changed source or read any changed
    processing key + all descendants.
    """
    changed_source_set = {str(item) for item in changed_sources or [] if str(item).strip()}
    changed_key_set = {str(item) for item in changed_keys or [] if str(item).strip()}
    if not changed_source_set and not changed_key_set:
        return set()

    id_by_node = {node: str(node.step_id) for node in pipeline.graph}

    seed_ids: set[str] = set()
    for node, sid in id_by_node.items():
        cfg = dict(getattr(node, "configuration", {}) or {})
        refs = _step_source_refs(cfg)
        reads, _writes = _step_processing_key_patterns(cfg)

        source_match = bool(refs & changed_source_set)
        key_match = any(_match_changed_key(changed_key, reads) for changed_key in changed_key_set)
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
    topo_nodes = list(TopologicalSorter(pipeline.graph).static_order())
    ordered_dirty = [str(node.step_id) for node in topo_nodes if str(node.step_id) in dirty_ids]
    return set(ordered_dirty)
