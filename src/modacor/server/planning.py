# SPDX-License-Identifier: BSD-3-Clause
# /usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from graphlib import TopologicalSorter
from typing import Any

from modacor.runner.pipeline import Pipeline

from .execution import find_dirty_step_ids
from .session_manager import PipelineSession

__all__ = [
    "build_dry_run_plan",
    "missing_required_source_refs",
    "ordered_step_ids",
    "resolve_effective_mode",
]


def resolve_effective_mode(requested_mode: str) -> tuple[str, str | None]:
    if requested_mode == "partial":
        return "partial", None
    if requested_mode == "auto":
        return "partial", "auto mode: partial first, full fallback on failure"
    return "full", None


def ordered_step_ids(pipeline: Pipeline) -> list[str]:
    return [str(node.step_id) for node in TopologicalSorter(pipeline.graph).static_order()]


def missing_required_source_refs(session: PipelineSession) -> list[str]:
    if not session.required_source_refs:
        return []
    present_refs = set(session.sources.keys())
    return [ref for ref in session.required_source_refs if ref not in present_refs]


def build_dry_run_plan(
    session: PipelineSession,
    *,
    mode: str,
    changed_sources: list[str],
    changed_keys: list[str],
) -> dict[str, Any]:
    pipeline = Pipeline.from_yaml(session.pipeline_yaml or "")
    topo_ids = ordered_step_ids(pipeline)

    effective_mode, mode_note = resolve_effective_mode(mode)
    warnings: list[str] = []
    if mode_note:
        warnings.append(mode_note)

    if effective_mode == "partial" and session.processing_data is None:
        effective_mode = "full"
        warnings.append("No previous ProcessingData snapshot available; dry-run assumes full rerun.")
    if mode == "auto" and not changed_sources and not changed_keys:
        effective_mode = "full"
        warnings.append("Auto mode without changed_sources/changed_keys defaults to full rerun.")

    missing_refs = missing_required_source_refs(session)
    if missing_refs:
        warnings.append(f"Missing required sources for profile '{session.source_profile}': {', '.join(missing_refs)}.")

    if effective_mode == "partial":
        dirty_ids = find_dirty_step_ids(
            pipeline,
            changed_sources=changed_sources,
            changed_keys=changed_keys,
        )
        dirty_steps = [step_id for step_id in topo_ids if step_id in dirty_ids]
        skipped_steps = [step_id for step_id in topo_ids if step_id not in dirty_ids]
    else:
        dirty_steps = list(topo_ids)
        skipped_steps = []

    boundary_step = dirty_steps[0] if dirty_steps else None

    return {
        "session_id": session.session_id,
        "mode": mode,
        "effective_mode": effective_mode,
        "changed_sources": list(changed_sources),
        "changed_keys": list(changed_keys),
        "topological_steps": topo_ids,
        "dirty_steps": dirty_steps,
        "skipped_steps": skipped_steps,
        "checkpoint_boundary_step": boundary_step,
        "fallback_strategy": "full_on_partial_failure" if mode == "auto" else None,
        "missing_required_sources": missing_refs,
        "warnings": warnings,
        "can_process": len(missing_refs) == 0,
    }
