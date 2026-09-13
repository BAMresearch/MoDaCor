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
from modacor.io.chunking import ChunkOutputStatus, ChunkPlan, ChunkSpec, ChunkWriteResult
from modacor.io.io_sink import IoSink
from modacor.io.runtime_support import build_sink_from_spec

from .runtime_policy import RuntimePolicy

__all__ = ["ChunkedOutputManager", "ChunkedOutputResource"]


def _utc_now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


@dataclass(slots=True)
class ChunkedOutputResource:
    """Immutable destination registration plus mutable lifecycle synchronization."""

    output_id: str
    sink_spec: dict[str, Any]
    target_key: tuple[str, str]
    subpath: str
    plan: ChunkPlan
    sink: IoSink = field(repr=False)
    lock: RLock = field(repr=False)
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
            subpath, plan = sink.load_chunked_plan(plan_id)
            if plan_hash is not None and str(plan_hash) != plan.plan_hash:
                raise ValueError("plan_hash does not match the persisted chunk plan.")
            sink.initialize_chunked(subpath, plan, collision="resume")
            resource = ChunkedOutputResource(
                output_id=f"out-{uuid4().hex[:12]}",
                sink_spec=normalized,
                target_key=target_key,
                subpath=subpath,
                plan=plan,
                sink=sink,
                lock=target_lock,
            )
            with self._lock:
                self._resources[resource.output_id] = resource
            status = sink.inspect_chunked(subpath, plan=plan)
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
        chunk: ChunkSpec,
        execution_metadata: dict[str, Any] | None = None,
        pipeline_spec: dict[str, Any] | None = None,
        pipeline_yaml: str | None = None,
        trace_events: Any | None = None,
    ) -> ChunkWriteResult:
        resource = self.get(output_id)
        chunk.validate_for_plan(resource.plan)
        with resource.lock:
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

    def recover(self, output_id: str, *, action: str) -> ChunkOutputStatus:
        resource = self.get(output_id)
        with resource.lock:
            return resource.sink.recover_chunked(resource.subpath, plan=resource.plan, action=action)

    def detach(self, output_id: str) -> bool:
        """Forget one opaque mapping without deleting its persistent output."""

        with self._lock:
            return self._resources.pop(str(output_id), None) is not None

    def finalize(self, output_id: str, *, plan_hash: str) -> ChunkWriteResult:
        resource = self.get(output_id)
        if str(plan_hash) != resource.plan.plan_hash:
            raise ValueError("plan_hash does not match the initialized chunked output.")
        with resource.lock:
            return resource.sink.finalize_chunked(
                resource.subpath,
                plan=resource.plan,
                pipeline_spec=resource.pipeline_spec,
                pipeline_yaml=resource.pipeline_yaml,
            )
