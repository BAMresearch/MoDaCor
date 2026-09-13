# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Any, Mapping
from uuid import uuid4

from modacor.dataclasses.processing_data import ProcessingData
from modacor.io.chunk_planning import materialize_chunk_specs, resolve_provisional_chunk_plan
from modacor.io.chunking import (
    ChunkInputPlan,
    ChunkOutputStatus,
    ChunkPlan,
    ChunkSpec,
    ChunkWriteResult,
    ProvisionalChunkPlan,
    ProvisionalChunkSpec,
)
from modacor.io.io_sink import IoSink
from modacor.io.runtime_support import build_sink_from_spec

from .runtime_policy import RuntimePolicy

__all__ = ["ChunkedOutputManager", "ChunkedOutputResource"]


def _utc_now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _resolved_pilot_trace_events(
    trace_events: Any | None,
    chunk: ChunkSpec,
    provisional_hash: str,
) -> Any | None:
    if not isinstance(trace_events, list):
        return trace_events
    identity = {**chunk.identity_dict(), "provisional_hash": provisional_hash}
    resolved: list[Any] = []
    for event in trace_events:
        if hasattr(event, "to_dict"):
            event = event.to_dict()
        if isinstance(event, Mapping):
            event = deepcopy(dict(event))
            event["chunk_identity"] = identity
        resolved.append(event)
    return resolved


@dataclass(slots=True)
class ChunkedOutputResource:
    """Immutable destination registration plus mutable lifecycle synchronization."""

    output_id: str
    sink_spec: dict[str, Any]
    target_key: tuple[str, str]
    subpath: str
    plan: ChunkPlan | None
    sink: IoSink = field(repr=False)
    lock: RLock = field(repr=False)
    provisional_plan: ProvisionalChunkPlan | None = None
    input_plan: ChunkInputPlan | None = None
    chunk_specs: tuple[ChunkSpec, ...] = ()
    created_utc: str = field(default_factory=_utc_now_iso)
    pipeline_spec: dict[str, Any] | None = field(default=None, repr=False)
    pipeline_yaml: str | None = field(default=None, repr=False)


class ChunkedOutputManager:
    """Own server-level chunked outputs independently of pipeline sessions."""

    def __init__(self, *, policy: RuntimePolicy) -> None:
        self._policy = policy
        self._resources: dict[str, ChunkedOutputResource] = {}
        self._target_locks: dict[tuple[str, str], RLock] = {}
        self._lock = RLock()

    @staticmethod
    def _normalize_sink_spec(spec: Mapping[str, Any]) -> dict[str, Any]:
        missing = [name for name in ("ref", "type", "location") if name not in spec]
        if missing:
            raise ValueError(f"Sink registration missing required field(s): {', '.join(missing)}.")
        kwargs = spec.get("kwargs", {}) or {}
        if not isinstance(kwargs, Mapping):
            raise ValueError("Sink registration kwargs must be an object.")
        normalized = {
            "ref": str(spec["ref"]).strip(),
            "type": str(spec["type"]).strip(),
            "location": str(spec["location"]).strip(),
            "kwargs": deepcopy(dict(kwargs)),
        }
        if not all(normalized[name] for name in ("ref", "type", "location")):
            raise ValueError("Sink registration ref, type, and location must be non-empty.")
        return normalized

    def _prepare_sink(self, sink_spec: Mapping[str, Any]) -> tuple[dict[str, Any], IoSink, tuple[str, str], RLock]:
        normalized = self._normalize_sink_spec(sink_spec)
        self._policy.validate_sink_registration(normalized)
        sink = build_sink_from_spec(normalized, **self._policy.sink_builder_kwargs())
        if not isinstance(sink, IoSink) or not sink.supports_chunked_writes:
            raise ValueError(f"Sink type {normalized['type']!r} does not support chunked writes.")

        sink_type = normalized["type"].lower()
        location = normalized["location"]
        if sink_type in {"csv", "hdf", "hdf_chunked", "hdf_processing"}:
            location = str(Path(location).expanduser().resolve(strict=False))
        target_key = (sink_type, location)
        with self._lock:
            target_lock = self._target_locks.setdefault(target_key, RLock())
        return normalized, sink, target_key, target_lock

    def initialize(
        self,
        *,
        sink_spec: Mapping[str, Any],
        subpath: str,
        plan: ChunkPlan,
        collision: str = "error",
    ) -> tuple[ChunkedOutputResource, ChunkWriteResult]:
        normalized, sink, target_key, target_lock = self._prepare_sink(sink_spec)

        with target_lock:
            result = sink.initialize_chunked(str(subpath), plan, collision=collision)
            resource = ChunkedOutputResource(
                output_id=f"out-{uuid4().hex[:12]}",
                sink_spec=normalized,
                target_key=target_key,
                subpath=str(subpath),
                plan=plan,
                sink=sink,
                lock=target_lock,
                chunk_specs=(),
            )
            with self._lock:
                if collision == "replace":
                    obsolete = [
                        output_id
                        for output_id, existing in self._resources.items()
                        if existing.target_key == target_key and existing.subpath == str(subpath)
                    ]
                    for output_id in obsolete:
                        del self._resources[output_id]
                self._resources[resource.output_id] = resource
        return resource, result

    def initialize_provisional(
        self,
        *,
        sink_spec: Mapping[str, Any],
        subpath: str,
        provisional_plan: ProvisionalChunkPlan,
        input_plan: ChunkInputPlan,
        collision: str = "error",
    ) -> tuple[ChunkedOutputResource, ChunkWriteResult]:
        normalized, sink, target_key, target_lock = self._prepare_sink(sink_spec)
        with target_lock:
            result = sink.initialize_provisional_chunked(
                str(subpath),
                provisional_plan,
                input_plan=input_plan,
                collision=collision,
            )
            resource = ChunkedOutputResource(
                output_id=f"out-{uuid4().hex[:12]}",
                sink_spec=normalized,
                target_key=target_key,
                subpath=str(subpath),
                plan=None,
                sink=sink,
                lock=target_lock,
                provisional_plan=provisional_plan,
                input_plan=input_plan,
            )
            with self._lock:
                if collision == "replace":
                    obsolete = [
                        output_id
                        for output_id, existing in self._resources.items()
                        if existing.target_key == target_key and existing.subpath == str(subpath)
                    ]
                    for output_id in obsolete:
                        del self._resources[output_id]
                self._resources[resource.output_id] = resource
        return resource, result

    def reopen(
        self,
        *,
        sink_spec: Mapping[str, Any],
        plan_id: str,
        plan_hash: str | None = None,
    ) -> tuple[ChunkedOutputResource, ChunkOutputStatus]:
        """Reconstruct a server mapping from an authoritative sink manifest."""

        normalized, sink, target_key, target_lock = self._prepare_sink(sink_spec)
        with target_lock:
            provisional_plan: ProvisionalChunkPlan | None = None
            input_plan: ChunkInputPlan | None = None
            try:
                subpath, provisional_plan, input_plan = sink.load_provisional_chunked(plan_id)
            except (KeyError, ValueError):
                subpath, plan = sink.load_chunked_plan(plan_id)
                ancestry = sink.load_chunked_resolution(plan_id)
                if ancestry is not None:
                    provisional_plan, input_plan = ancestry
            else:
                plan = None
            active_hash = provisional_plan.provisional_hash if plan is None else plan.plan_hash
            if plan_hash is not None and str(plan_hash) != active_hash:
                raise ValueError("plan_hash does not match the persisted chunk plan.")
            if plan is None:
                assert provisional_plan is not None and input_plan is not None
                sink.initialize_provisional_chunked(
                    subpath,
                    provisional_plan,
                    input_plan=input_plan,
                    collision="resume",
                )
            else:
                sink.initialize_chunked(subpath, plan, collision="resume")
            resource = ChunkedOutputResource(
                output_id=f"out-{uuid4().hex[:12]}",
                sink_spec=normalized,
                target_key=target_key,
                subpath=subpath,
                plan=plan,
                sink=sink,
                lock=target_lock,
                provisional_plan=provisional_plan,
                input_plan=input_plan,
                chunk_specs=() if plan is None or input_plan is None else materialize_chunk_specs(plan, input_plan),
            )
            with self._lock:
                self._resources[resource.output_id] = resource
            status = (
                sink.inspect_provisional_chunked(subpath, plan=provisional_plan, input_plan=input_plan)
                if plan is None
                else sink.inspect_chunked(subpath, plan=plan)
            )
        return resource, status

    def get(self, output_id: str) -> ChunkedOutputResource:
        with self._lock:
            try:
                return self._resources[str(output_id)]
            except KeyError as exc:
                raise KeyError(f"Chunked output {output_id!r} not found.") from exc

    def inspect(self, output_id: str, *, offset: int = 0, limit: int | None = None) -> ChunkOutputStatus:
        resource = self.get(output_id)
        with resource.lock:
            if resource.plan is None:
                assert resource.provisional_plan is not None and resource.input_plan is not None
                return resource.sink.inspect_provisional_chunked(
                    resource.subpath,
                    plan=resource.provisional_plan,
                    input_plan=resource.input_plan,
                    offset=offset,
                    limit=limit,
                )
            return resource.sink.inspect_chunked(
                resource.subpath,
                plan=resource.plan,
                offset=offset,
                limit=limit,
            )

    def write_chunk(
        self,
        output_id: str,
        processing_data: ProcessingData,
        *,
        chunk: ChunkSpec | ProvisionalChunkSpec,
        execution_metadata: dict[str, Any] | None = None,
        pipeline_spec: dict[str, Any] | None = None,
        pipeline_yaml: str | None = None,
        trace_events: Any | None = None,
    ) -> ChunkWriteResult:
        resource = self.get(output_id)
        with resource.lock:
            if resource.plan is None:
                provisional = resource.provisional_plan
                input_plan = resource.input_plan
                assert provisional is not None and input_plan is not None
                expected = input_plan.chunk(chunk.chunk_id)
                if not isinstance(chunk, ProvisionalChunkSpec) or chunk != expected:
                    raise ValueError("Pilot work item does not match the provisional input plan.")
                plan, specs = resolve_provisional_chunk_plan(
                    provisional,
                    input_plan,
                    processing_data,
                    chunk.chunk_id,
                )
                resolved_chunk = specs[chunk.ordinal]
                resolved_metadata = dict(execution_metadata or {})
                resolved_metadata.update(resolved_chunk.identity_dict())
                resolved_metadata["provisional_hash"] = provisional.provisional_hash
                try:
                    result = resource.sink.resolve_provisional_chunked(
                        resource.subpath,
                        processing_data,
                        provisional_plan=provisional,
                        input_plan=input_plan,
                        plan=plan,
                        chunk=resolved_chunk,
                        execution_metadata=resolved_metadata,
                        pipeline_spec=pipeline_spec,
                        pipeline_yaml=pipeline_yaml,
                        trace_events=_resolved_pilot_trace_events(
                            trace_events,
                            resolved_chunk,
                            provisional.provisional_hash,
                        ),
                    )
                except Exception:
                    try:
                        resource.sink.inspect_chunked(resource.subpath, plan=plan, offset=0, limit=1)
                    except Exception:
                        pass
                    else:
                        resource.plan = plan
                        resource.chunk_specs = specs
                    raise
                resource.plan = plan
                resource.chunk_specs = specs
                if pipeline_spec is not None:
                    resource.pipeline_spec = deepcopy(pipeline_spec)
                if pipeline_yaml is not None:
                    resource.pipeline_yaml = str(pipeline_yaml)
                return result

            if isinstance(chunk, ProvisionalChunkSpec):
                if not resource.chunk_specs:
                    raise ValueError("Resolved chunk specifications are unavailable.")
                chunk = resource.chunk_specs[chunk.ordinal]
            chunk.validate_for_plan(resource.plan)
            result = resource.sink.write_chunk(
                resource.subpath,
                processing_data,
                plan=resource.plan,
                chunk=chunk,
                execution_metadata=execution_metadata,
                pipeline_spec=pipeline_spec,
                pipeline_yaml=pipeline_yaml,
                trace_events=trace_events,
            )
            if pipeline_spec is not None:
                resource.pipeline_spec = deepcopy(pipeline_spec)
            if pipeline_yaml is not None:
                resource.pipeline_yaml = str(pipeline_yaml)
            return result

    def execution_chunk(self, output_id: str, chunk_id: str) -> ChunkSpec | ProvisionalChunkSpec:
        resource = self.get(output_id)
        with resource.lock:
            if resource.plan is None:
                assert resource.input_plan is not None
                return resource.input_plan.chunk(chunk_id)
            if resource.chunk_specs:
                for chunk in resource.chunk_specs:
                    if chunk.chunk_id == chunk_id:
                        return chunk
            raise KeyError(f"Chunk {chunk_id!r} is not managed by output {output_id!r}.")

    def recover(self, output_id: str, *, action: str) -> ChunkOutputStatus:
        resource = self.get(output_id)
        with resource.lock:
            if resource.plan is None:
                raise ValueError("An output awaiting schema can only be detached or replaced.")
            return resource.sink.recover_chunked(resource.subpath, plan=resource.plan, action=action)

    def detach(self, output_id: str) -> bool:
        """Forget one opaque mapping without deleting its persistent output."""

        with self._lock:
            return self._resources.pop(str(output_id), None) is not None

    def finalize(self, output_id: str, *, plan_hash: str) -> ChunkWriteResult:
        resource = self.get(output_id)
        if resource.plan is None:
            raise ValueError("Chunked output is awaiting pilot schema resolution and cannot be finalized.")
        if str(plan_hash) != resource.plan.plan_hash:
            raise ValueError("plan_hash does not match the initialized chunked output.")
        with resource.lock:
            return resource.sink.finalize_chunked(
                resource.subpath,
                plan=resource.plan,
                pipeline_spec=resource.pipeline_spec,
                pipeline_yaml=resource.pipeline_yaml,
            )
