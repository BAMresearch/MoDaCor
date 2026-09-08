# SPDX-License-Identifier: BSD-3-Clause
# /usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
from typing import Any

from modacor.io.buffer.runtime_buffer_store import RuntimeBufferStore
from modacor.io.io_sinks import IoSinks
from modacor.io.io_sources import IoSources
from modacor.io.runtime_support import build_sink_from_spec, build_source_from_spec, write_processing_data_hdf

from .runtime_policy import RuntimePolicy
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


def _source_spec_from_registration(ref: str, reg: dict[str, Any]) -> dict[str, Any]:
    return {
        "ref": ref,
        "type": reg["type"],
        "location": reg["location"],
        "kwargs": dict(reg.get("kwargs", {}) or {}),
    }


def _cache_key_value(value: Any) -> Any:
    if isinstance(value, dict):
        return tuple(sorted((str(key), _cache_key_value(val)) for key, val in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_cache_key_value(item) for item in value)
    if isinstance(value, Path):
        return str(value)
    try:
        hash(value)
    except TypeError:
        return repr(value)
    return value


def _hdf_file_fingerprint(location: str) -> tuple[Any, ...]:
    path = Path(str(location)).expanduser()
    try:
        stat = path.stat()
    except OSError:
        return ("missing", str(path))
    return ("file", str(path), stat.st_mtime_ns, stat.st_size)


def _source_cache_fingerprint(spec: dict[str, Any]) -> tuple[Any, ...] | None:
    source_type = str(spec["type"]).strip().lower()
    if source_type != "hdf":
        return None

    return (
        str(spec["ref"]).strip(),
        source_type,
        str(spec["location"]),
        _cache_key_value(spec.get("kwargs", {}) or {}),
        _hdf_file_fingerprint(str(spec["location"])),
    )


def _source_from_session_cache(
    session: PipelineSession,
    spec: dict[str, Any],
    *,
    buffer_store: RuntimeBufferStore | None = None,
    runtime_policy: RuntimePolicy | None = None,
) -> Any:
    policy = runtime_policy or RuntimePolicy.trusted()
    policy.validate_source_registration(spec)
    ref = str(spec["ref"]).strip()
    fingerprint = _source_cache_fingerprint(spec)
    if fingerprint is None:
        return build_source_from_spec(
            spec,
            buffer_store=buffer_store,
            session_id=session.session_id,
            **policy.source_builder_kwargs(),
        )

    cached = session.source_cache.get(ref)
    if cached is not None and cached.get("fingerprint") == fingerprint:
        return cached["source"]

    source = build_source_from_spec(
        spec,
        buffer_store=buffer_store,
        session_id=session.session_id,
        **policy.source_builder_kwargs(),
    )
    session.source_cache[ref] = {"fingerprint": fingerprint, "source": source}
    return source


def build_sources_from_session(
    session: PipelineSession,
    *,
    buffer_store: RuntimeBufferStore | None = None,
    runtime_policy: RuntimePolicy | None = None,
) -> IoSources:
    sources = IoSources()
    for ref in sorted(session.sources.keys()):
        reg = session.sources[ref]
        source = _source_from_session_cache(
            session,
            _source_spec_from_registration(ref, reg),
            buffer_store=buffer_store,
            runtime_policy=runtime_policy,
        )
        sources.register_source(source)
    return sources


def build_sinks_from_session(
    session: PipelineSession,
    *,
    pipeline: Any | None = None,
    buffer_store: RuntimeBufferStore | None = None,
    runtime_policy: RuntimePolicy | None = None,
) -> IoSinks:
    policy = runtime_policy or RuntimePolicy.trusted()
    sinks = IoSinks()
    for ref in sorted(session.sinks.keys()):
        reg = session.sinks[ref]
        policy.validate_sink_registration({"ref": ref, **reg})
        sink_type = str(reg["type"]).strip().lower()
        kwargs = dict(reg.get("kwargs", {}) or {})
        if sink_type in {"hdf", "hdf_processing"}:
            kwargs = _sink_kwargs_with_runtime_metadata(
                kwargs,
                session=session,
                pipeline=pipeline,
            )
        sink = build_sink_from_spec(
            {
                "ref": ref,
                "type": reg["type"],
                "location": reg["location"],
                "kwargs": kwargs,
            },
            buffer_store=buffer_store,
            session_id=session.session_id,
            **policy.sink_builder_kwargs(),
        )
        sinks.register_sink(sink)
    return sinks


def write_hdf_output(
    write_hdf: dict[str, Any] | None,
    *,
    run_name: str,
    result: Any,
    pipeline_yaml: str,
    runtime_policy: RuntimePolicy | None = None,
) -> str | None:
    policy = runtime_policy or RuntimePolicy.trusted()
    if write_hdf and write_hdf.get("path"):
        policy.validate_write_hdf_path(write_hdf["path"])
    return write_processing_data_hdf(
        write_hdf,
        run_name=run_name,
        result=result,
        pipeline_yaml=pipeline_yaml,
    )
