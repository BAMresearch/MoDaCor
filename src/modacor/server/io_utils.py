# SPDX-License-Identifier: BSD-3-Clause
# /usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Any

from modacor.io.io_sinks import IoSinks
from modacor.io.io_sources import IoSources
from modacor.io.runtime_support import build_sinks_from_specs, build_sources_from_specs, write_processing_data_hdf

from .session_manager import PipelineSession

__all__ = ["build_sinks_from_session", "build_sources_from_session", "write_hdf_output"]


def _runtime_metadata_flags(value: Any) -> dict[str, bool]:
    if isinstance(value, bool):
        return {
            "pipeline_yaml": value,
            "pipeline_spec": value,
            "trace_events": value,
        }
    if isinstance(value, dict):
        return {str(key): bool(flag) for key, flag in value.items()}
    return {}


def _flatten_trace_events(pipeline: Any | None) -> list[Any]:
    trace_events = getattr(pipeline, "trace_events", None)
    if not isinstance(trace_events, dict):
        return []

    flattened: list[Any] = []
    for step_events in trace_events.values():
        if isinstance(step_events, list):
            flattened.extend(step_events)
    return flattened


def _sink_kwargs_with_runtime_metadata(
    kwargs: dict[str, Any],
    *,
    session: PipelineSession,
    pipeline: Any | None,
) -> dict[str, Any]:
    include_runtime_metadata = kwargs.pop("include_runtime_metadata", False)
    flags = _runtime_metadata_flags(include_runtime_metadata)
    if not flags:
        return kwargs

    method_kwargs = dict(kwargs.get("iosink_method_kwargs", kwargs))
    if flags.get("pipeline_yaml"):
        method_kwargs.setdefault("pipeline_yaml", session.pipeline_yaml or "")
    if flags.get("pipeline_spec") and pipeline is not None:
        method_kwargs.setdefault("pipeline_spec", pipeline.to_spec())
    if flags.get("trace_events"):
        trace_events = _flatten_trace_events(pipeline)
        if trace_events:
            method_kwargs.setdefault("trace_events", trace_events)

    if "iosink_method_kwargs" in kwargs:
        kwargs["iosink_method_kwargs"] = method_kwargs
        return kwargs
    return method_kwargs


def build_sources_from_session(session: PipelineSession) -> IoSources:
    specs: list[dict[str, Any]] = []
    for ref in sorted(session.sources.keys()):
        reg = session.sources[ref]
        specs.append(
            {
                "ref": ref,
                "type": reg["type"],
                "location": reg["location"],
                "kwargs": dict(reg.get("kwargs", {}) or {}),
            }
        )
    return build_sources_from_specs(specs)


def build_sinks_from_session(session: PipelineSession, *, pipeline: Any | None = None) -> IoSinks:
    specs: list[dict[str, Any]] = []
    for ref in sorted(session.sinks.keys()):
        reg = session.sinks[ref]
        sink_type = str(reg["type"]).strip().lower()
        kwargs = dict(reg.get("kwargs", {}) or {})
        if sink_type in {"hdf", "hdf_processing"}:
            kwargs = _sink_kwargs_with_runtime_metadata(
                kwargs,
                session=session,
                pipeline=pipeline,
            )
        specs.append(
            {
                "ref": ref,
                "type": reg["type"],
                "location": reg["location"],
                "kwargs": kwargs,
            }
        )
    return build_sinks_from_specs(specs)


def write_hdf_output(
    write_hdf: dict[str, Any] | None,
    *,
    run_name: str,
    result: Any,
    pipeline_yaml: str,
) -> str | None:
    return write_processing_data_hdf(
        write_hdf,
        run_name=run_name,
        result=result,
        pipeline_yaml=pipeline_yaml,
    )
