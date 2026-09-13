# SPDX-License-Identifier: BSD-3-Clause
# /usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import traceback
from copy import deepcopy
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any

from modacor.dataclasses.processing_data import ProcessingData
from modacor.debug.pipeline_tracer import PlainUnicodeRenderer
from modacor.io.buffer.codec import decode_npy, encode_npy
from modacor.io.chunk_planning import resolve_chunk_input_plan
from modacor.io.chunking import ChunkPlan, ChunkSpec, ProvisionalChunkPlan, ProvisionalChunkSpec
from modacor.io.io_sinks import IoSinks
from modacor.io.io_sources import IoSources
from modacor.runner import run_pipeline_job
from modacor.runner.pipeline import Pipeline
from modacor.runner.pipeline_runner import PipelineRunError, RunResult
from modacor.runner.process_step_registry import ProcessStepRegistry

from .chunked_outputs import ChunkedOutputManager
from .errors import ApiError
from .execution import find_dirty_step_ids
from .io_utils import bind_chunk_source_slices, build_sinks_from_session, build_sources_from_session, write_hdf_output
from .planning import build_dry_run_plan, missing_required_source_refs, ordered_step_ids, resolve_effective_mode
from .runtime_policy import RuntimePolicy
from .session_manager import PipelineSession, SessionManager
from .source_profiles import get_source_profile, list_source_profiles

__all__ = ["RuntimeService"]

TRACE_REPORT_LINES = 500


def _source_provenance(session: PipelineSession) -> list[dict[str, str]]:
    """Record source identity without persisting constructor kwargs or credentials."""

    return [
        {name: str(registration[name]) for name in ("ref", "type", "location")}
        for registration in session.sources.values()
    ]


def _pipeline_trace_events(pipeline: Pipeline) -> list[Any]:
    """Flatten lightweight, array-free step events in execution order."""

    return [event for step_events in pipeline.trace_events.values() for event in step_events]


def _pipeline_provenance_spec(pipeline: Pipeline) -> dict[str, Any]:
    """Keep run-specific trace events out of the plan-level pipeline record."""

    pipeline_spec = pipeline.to_spec()
    for node in pipeline_spec.get("nodes", []):
        node.pop("trace_events", None)
    return pipeline_spec


@dataclass(frozen=True, slots=True)
class ChunkPublishRequest:
    output_id: str
    chunk_spec: ChunkSpec | ProvisionalChunkSpec


@dataclass(slots=True)
class ProcessRequest:
    """Normalized process request payload used by runtime service methods."""

    mode: str
    changed_sources: list[str]
    changed_keys: list[str]
    write_hdf: dict[str, Any] | None = None
    run_name: str | None = None
    rollback_snapshot: bool = True
    chunk_output: ChunkPublishRequest | None = None
    resolved_source_slices: list[dict[str, Any]] = field(default_factory=list)


class _ChunkPublicationError(RuntimeError):
    def __init__(
        self,
        output_id: str,
        chunk_spec: ChunkSpec | ProvisionalChunkSpec,
        original_exception: Exception,
    ) -> None:
        super().__init__(str(original_exception))
        self.output_id = output_id
        self.chunk_spec = chunk_spec
        self.original_exception = original_exception


@dataclass(slots=True)
class ProcessPreparation:
    """Execution preparation details for a process request."""

    selected_step_ids: set[str] | None = None
    snapshot_before_partial: ProcessingData | None = None
    dirty_step_ids_ordered: list[str] = field(default_factory=list)
    boundary_step_id: str | None = None
    early_response: dict[str, Any] | None = None


@dataclass(slots=True)
class RuntimeService:
    """Application service for the MoDaCor runtime API."""

    manager: SessionManager
    policy: RuntimePolicy = field(default_factory=RuntimePolicy.trusted)
    process_step_registry: ProcessStepRegistry = field(init=False)
    chunked_outputs: ChunkedOutputManager = field(init=False)

    def __post_init__(self) -> None:
        self.process_step_registry = self.policy.create_process_step_registry()
        self.chunked_outputs = ChunkedOutputManager(policy=self.policy)

    def _chunk_sink_registration(self, payload: dict[str, Any]) -> dict[str, Any]:
        sink_raw = payload.get("sink")
        if sink_raw is None:
            session_id = str(payload.get("session_id", "")).strip()
            sink_ref = str(payload.get("sink_ref", "")).strip()
            if not session_id or not sink_ref:
                raise ApiError(status_code=422, detail="Provide sink or both session_id and sink_ref.")
            session = self._require_session(session_id)
            try:
                sink_raw = session.sinks[sink_ref]
            except KeyError as exc:
                raise ApiError(status_code=404, detail=f"Sink {sink_ref!r} is not registered in the session.") from exc
        if not isinstance(sink_raw, dict):
            raise ApiError(status_code=422, detail="sink must be an I/O registration object.")
        return sink_raw

    def initialize_chunked_output(self, payload: dict[str, Any]) -> dict[str, Any]:
        plan_raw = payload.get("plan")
        provisional_raw = payload.get("provisional_plan")
        if isinstance(plan_raw, dict) == isinstance(provisional_raw, dict):
            raise ApiError(status_code=422, detail="Provide exactly one of plan or provisional_plan.")
        sink_raw = self._chunk_sink_registration(payload)
        collision = str(payload.get("collision", "error"))

        if isinstance(provisional_raw, dict):
            try:
                provisional = ProvisionalChunkPlan.from_dict(provisional_raw)
                session_id = str(payload.get("session_id", "")).strip()
                if not session_id:
                    raise ValueError("session_id is required for provisional source discovery.")
                session = self._require_session(session_id)
                driver_reference = IoSources.normalize_data_reference(str(provisional.driver["source"]))
                source_ref, data_key = driver_reference.split("::", 1)
                try:
                    registration = session.sources[source_ref]
                except KeyError as exc:
                    raise ValueError(
                        f"Provisional driver source {source_ref!r} is not registered in the session."
                    ) from exc
                source_type = str(registration["type"]).strip().lower()
                declared_shape = provisional.driver.get("full_shape")
                if source_type == "buffer" and declared_shape is None:
                    raise ValueError("BufferSource provisional plans require driver.full_shape.")
                if source_type == "buffer":
                    source_shape = tuple(int(size) for size in declared_shape)
                    source_dtype = provisional.driver.get("dtype")
                else:
                    sources = build_sources_from_session(
                        session,
                        buffer_store=self.manager.buffer_store,
                        runtime_policy=self.policy,
                    )
                    source = sources.get_source(source_ref)
                    source_shape = tuple(source.get_data_shape(data_key))
                    source_dtype = source.get_data_dtype(data_key)
                    if not source_shape and declared_shape is not None:
                        source_shape = tuple(int(size) for size in declared_shape)
                input_plan = resolve_chunk_input_plan(provisional, source_shape, source_dtype)
                resource, result = self.chunked_outputs.initialize_provisional(
                    sink_spec=sink_raw,
                    subpath=str(payload.get("subpath", "")),
                    provisional_plan=provisional,
                    input_plan=input_plan,
                    collision=collision,
                )
            except ApiError:
                raise
            except FileExistsError as exc:
                raise ApiError(status_code=409, detail=str(exc)) from exc
            except (KeyError, TypeError, ValueError) as exc:
                raise ApiError(status_code=422, detail=f"Invalid provisional chunk plan: {exc}") from exc
            progress = self.chunked_outputs.inspect(resource.output_id, offset=0, limit=1).to_dict()
            progress.pop("chunks", None)
            return {
                "output_id": resource.output_id,
                "created_utc": resource.created_utc,
                "sink_type": resource.sink_spec["type"],
                "subpath": resource.subpath,
                "provisional_hash": provisional.provisional_hash,
                "resolution_hash": input_plan.resolution_hash,
                "pilot_chunk_id": input_plan.chunks[0].chunk_id,
                **result.to_dict(),
                **progress,
            }

        assert isinstance(plan_raw, dict)
        try:
            plan = ChunkPlan.from_dict(plan_raw)
            resource, result = self.chunked_outputs.initialize(
                sink_spec=sink_raw,
                subpath=str(payload.get("subpath", "")),
                plan=plan,
                collision=collision,
            )
        except FileExistsError as exc:
            raise ApiError(status_code=409, detail=str(exc)) from exc
        except (KeyError, TypeError, ValueError) as exc:
            raise ApiError(status_code=422, detail=f"Invalid complete ChunkPlan: {exc}") from exc
        progress = self.chunked_outputs.inspect(resource.output_id, offset=0, limit=1).to_dict()
        progress.pop("chunks", None)
        return {
            "output_id": resource.output_id,
            "created_utc": resource.created_utc,
            "sink_type": resource.sink_spec["type"],
            "subpath": resource.subpath,
            **result.to_dict(),
            **progress,
        }

    def reopen_chunked_output(self, payload: dict[str, Any]) -> dict[str, Any]:
        plan_id = str(payload.get("plan_id", "")).strip()
        if not plan_id:
            raise ApiError(status_code=422, detail="plan_id is required.")
        sink_raw = self._chunk_sink_registration(payload)
        plan_hash_raw = payload.get("plan_hash")
        plan_hash = None if plan_hash_raw is None else str(plan_hash_raw)
        try:
            resource, status = self.chunked_outputs.reopen(
                sink_spec=sink_raw,
                plan_id=plan_id,
                plan_hash=plan_hash,
            )
        except KeyError as exc:
            raise ApiError(status_code=404, detail=str(exc)) from exc
        except (TypeError, ValueError) as exc:
            raise ApiError(status_code=422, detail=str(exc)) from exc
        return {
            "output_id": resource.output_id,
            "created_utc": resource.created_utc,
            "sink_type": resource.sink_spec["type"],
            "subpath": resource.subpath,
            "reopened": True,
            **status.to_dict(),
        }

    def inspect_chunked_output(self, output_id: str, *, offset: int = 0, limit: int = 100) -> dict[str, Any]:
        try:
            resource = self.chunked_outputs.get(output_id)
            status = self.chunked_outputs.inspect(output_id, offset=offset, limit=limit)
        except KeyError as exc:
            raise ApiError(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise ApiError(status_code=422, detail=str(exc)) from exc
        return {
            "output_id": resource.output_id,
            "created_utc": resource.created_utc,
            "sink_type": resource.sink_spec["type"],
            "subpath": resource.subpath,
            "offset": offset,
            "limit": limit,
            **status.to_dict(),
        }

    def finalize_chunked_output(self, output_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        plan_hash = str(payload.get("plan_hash", "")).strip()
        if not plan_hash:
            raise ApiError(status_code=422, detail="plan_hash is required.")
        try:
            result = self.chunked_outputs.finalize(output_id, plan_hash=plan_hash)
        except KeyError as exc:
            raise ApiError(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise ApiError(status_code=409, detail=str(exc)) from exc
        except RuntimeError as exc:
            try:
                progress = self.inspect_chunked_output(output_id)
            except ApiError:
                progress = None
            raise ApiError(
                status_code=409,
                detail={"code": "CHUNK_FINALIZE_CONFLICT", "message": str(exc), "progress": progress},
            ) from exc
        return {"output_id": output_id, **result.to_dict()}

    def recover_chunked_output(self, output_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        action = str(payload.get("action", "")).strip()
        if action not in {"reconcile", "abandon", "resume"}:
            raise ApiError(status_code=422, detail="action must be one of: reconcile, abandon, resume.")
        try:
            status = self.chunked_outputs.recover(output_id, action=action)
        except KeyError as exc:
            raise ApiError(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise ApiError(status_code=422, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise ApiError(status_code=409, detail=str(exc)) from exc
        return {"output_id": output_id, "action": action, **status.to_dict()}

    def detach_chunked_output(self, output_id: str) -> None:
        if not self.chunked_outputs.detach(output_id):
            raise ApiError(status_code=404, detail="Chunked output not found.")

    def health(self) -> dict[str, str]:
        """Return a lightweight liveness payload for process health checks."""

        return {"status": "ok"}

    def readiness(self) -> dict[str, Any]:
        """
        Return runtime readiness and high-level session metrics.

        The service stays `ready` as long as it can accept new requests. Session
        failures downgrade the status to `degraded` without making the service
        unavailable.
        """

        sessions = self.manager.list_sessions()
        error_session_ids = [session.session_id for session in sessions if session.state.startswith("error_")]
        last_updated_utc = max((session.updated_utc for session in sessions), default=None)

        return {
            "status": "degraded" if error_session_ids else "ready",
            "ready": True,
            "metrics": {
                "session_count": len(sessions),
                "active_run_count": sum(1 for session in sessions if session.active_run_id is not None),
                "error_session_count": len(error_session_ids),
                "error_session_ids": error_session_ids,
                "last_updated_utc": last_updated_utc,
                "runtime_policy": self.policy_summary(),
            },
        }

    def policy_summary(self) -> dict[str, Any]:
        return {
            "allow_pipeline_yaml_path": self.policy.allow_pipeline_yaml_path,
            "allow_process_step_filesystem_discovery": self.policy.allow_process_step_filesystem_discovery,
            "allow_custom_io_class_path": self.policy.allow_custom_io_class_path,
            "require_io_path_roots": self.policy.require_io_path_roots,
            "source_read_roots": [str(path) for path in self.policy.source_read_roots],
            "sink_write_roots": [str(path) for path in self.policy.sink_write_roots],
            "max_sessions": self.policy.max_sessions,
            "max_pipeline_yaml_bytes": self.policy.max_pipeline_yaml_bytes,
            "max_buffer_upload_bytes": self.policy.max_buffer_upload_bytes,
        }

    def latest_error(self, session_id: str) -> dict[str, Any]:
        """Return the current and latest recorded error diagnostics for a session."""

        session = self._require_session(session_id)
        latest_failed_run = self._latest_failed_run(session)
        latest_error = None if latest_failed_run is None else latest_failed_run.get("error")
        return {
            "session_id": session.session_id,
            "state": session.state,
            "active_run_id": session.active_run_id,
            "updated_utc": session.updated_utc,
            "current_error": session.last_error,
            "latest_error": latest_error,
            "latest_failed_run": latest_failed_run,
        }

    def session_summary(self, session: PipelineSession) -> dict[str, Any]:
        return {
            "session_id": session.session_id,
            "name": session.name,
            "state": session.state,
            "active_run_id": session.active_run_id,
            "updated_utc": session.updated_utc,
        }

    def session_detail(self, session: PipelineSession) -> dict[str, Any]:
        out = self.session_summary(session)
        out.update(
            {
                "sources": list(session.sources.values()),
                "sinks": list(session.sinks.values()),
                "trace": {
                    "enabled": session.trace_enabled,
                    "watch": session.trace_watch,
                    "record_only_on_change": True,
                    "snapshot_processing_data": session.trace_snapshot_processing_data,
                    "snapshot_step_ids": list(session.trace_snapshot_step_ids),
                },
                "last_run": session.run_history[-1] if session.run_history else None,
                "source_profile": session.source_profile,
                "required_source_refs": list(session.required_source_refs),
                "runtime_policy": self.policy_summary(),
            }
        )
        return out

    def source_templates(self) -> dict[str, Any]:
        return {"templates": list_source_profiles()}

    def list_sessions(self) -> dict[str, Any]:
        return {"sessions": [self.session_summary(session) for session in self.manager.list_sessions()]}

    def create_session(self, payload: dict[str, Any]) -> dict[str, Any]:
        session_id = str(payload.get("session_id", "")).strip()
        if not session_id:
            raise ApiError(status_code=422, detail="session_id is required.")
        if self.manager.get_session(session_id) is None:
            try:
                self.policy.validate_session_capacity(len(self.manager.list_sessions()))
            except RuntimeError as exc:
                raise ApiError(status_code=429, detail=str(exc)) from exc

        pipeline = payload.get("pipeline", {}) or {}
        yaml_text = pipeline.get("yaml_text")
        yaml_path = pipeline.get("yaml_path")
        if bool(yaml_text) == bool(yaml_path):
            raise ApiError(status_code=422, detail="Exactly one of pipeline.yaml_text or pipeline.yaml_path required.")

        if yaml_path:
            try:
                pipeline_yaml = self.policy.validate_pipeline_yaml_path(yaml_path).read_text(encoding="utf-8")
                self.policy.validate_pipeline_yaml_text(pipeline_yaml)
            except Exception as exc:
                raise ApiError(status_code=422, detail=f"Failed to read pipeline yaml_path: {exc}") from exc
        else:
            pipeline_yaml = str(yaml_text)
            try:
                self.policy.validate_pipeline_yaml_text(pipeline_yaml)
            except ValueError as exc:
                raise ApiError(status_code=413, detail=str(exc)) from exc

        source_profile_name = payload.get("source_profile")
        required_source_refs: list[str] = []
        normalized_profile: str | None = None
        if source_profile_name is not None:
            profile = get_source_profile(str(source_profile_name))
            if profile is None:
                raise ApiError(status_code=422, detail=f"Unknown source_profile: {source_profile_name!r}")
            normalized_profile = str(source_profile_name).strip().lower()
            required_source_refs = [str(item["ref"]) for item in profile.get("required_sources", [])]

        trace = payload.get("trace", {}) or {}
        trace_snapshot_step_ids = [str(step_id) for step_id in trace.get("snapshot_step_ids", []) or []]
        trace_snapshot_processing_data = bool(trace.get("snapshot_processing_data", False) or trace_snapshot_step_ids)
        try:
            session = self.manager.create_session(
                session_id=session_id,
                name=payload.get("name"),
                pipeline_yaml=pipeline_yaml,
                trace_enabled=bool(trace.get("enabled", False) or trace_snapshot_processing_data),
                trace_watch=dict(trace.get("watch", {}) or {}),
                trace_snapshot_processing_data=trace_snapshot_processing_data,
                trace_snapshot_step_ids=trace_snapshot_step_ids,
                auto_full_reset_on_partial_error=bool(payload.get("auto_full_reset_on_partial_error", True)),
                source_profile=normalized_profile,
                required_source_refs=required_source_refs,
            )
        except ValueError as exc:
            raise ApiError(status_code=409, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise ApiError(status_code=429, detail=str(exc)) from exc
        return self.session_detail(session)

    def get_session(self, session_id: str) -> dict[str, Any]:
        return self.session_detail(self._require_session(session_id))

    def delete_session(self, session_id: str) -> None:
        if not self.manager.delete_session(session_id):
            raise ApiError(status_code=404, detail="Session not found.")

    def upsert_sources(self, session_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        sources = payload.get("sources", [])
        if not isinstance(sources, list):
            raise ApiError(status_code=422, detail="'sources' must be a list.")
        try:
            for source in sources:
                self.policy.validate_source_registration(source)
        except ValueError as exc:
            raise ApiError(status_code=422, detail=str(exc)) from exc
        try:
            session = self.manager.upsert_sources(session_id, sources=sources)
        except KeyError as exc:
            raise ApiError(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise ApiError(status_code=422, detail=str(exc)) from exc
        return {"session_id": session.session_id, "sources": list(session.sources.values())}

    def upsert_sinks(self, session_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        sinks = payload.get("sinks", [])
        if not isinstance(sinks, list):
            raise ApiError(status_code=422, detail="'sinks' must be a list.")
        try:
            for sink in sinks:
                self.policy.validate_sink_registration(sink)
        except ValueError as exc:
            raise ApiError(status_code=422, detail=str(exc)) from exc
        try:
            session = self.manager.upsert_sinks(session_id, sinks=sinks)
        except KeyError as exc:
            raise ApiError(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise ApiError(status_code=422, detail=str(exc)) from exc
        return {"session_id": session.session_id, "sinks": list(session.sinks.values())}

    def patch_source(self, session_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        ref = str(payload.get("ref", "")).strip()
        source_type = str(payload.get("type", "")).strip()
        location = str(payload.get("location", "")).strip()
        kwargs = payload.get("kwargs", {}) or {}

        if not ref:
            raise ApiError(status_code=422, detail="'ref' is required.")
        if not source_type:
            raise ApiError(status_code=422, detail="'type' is required.")
        if not location:
            raise ApiError(status_code=422, detail="'location' is required.")
        if not isinstance(kwargs, dict):
            raise ApiError(status_code=422, detail="'kwargs' must be an object when provided.")

        source_registration = {"ref": ref, "type": source_type, "location": location, "kwargs": kwargs}
        try:
            self.policy.validate_source_registration(source_registration)
        except ValueError as exc:
            raise ApiError(status_code=422, detail=str(exc)) from exc

        try:
            session = self.manager.upsert_sources(
                session_id,
                sources=[source_registration],
            )
        except KeyError as exc:
            raise ApiError(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise ApiError(status_code=422, detail=str(exc)) from exc

        return {
            "session_id": session.session_id,
            "source": session.sources.get(ref),
            "sources": list(session.sources.values()),
        }

    def patch_sink(self, session_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        ref = str(payload.get("ref", "")).strip()
        sink_type = str(payload.get("type", "")).strip()
        location = str(payload.get("location", "")).strip()
        kwargs = payload.get("kwargs", {}) or {}

        if not ref:
            raise ApiError(status_code=422, detail="'ref' is required.")
        if not sink_type:
            raise ApiError(status_code=422, detail="'type' is required.")
        if not location:
            raise ApiError(status_code=422, detail="'location' is required.")
        if not isinstance(kwargs, dict):
            raise ApiError(status_code=422, detail="'kwargs' must be an object when provided.")

        sink_registration = {"ref": ref, "type": sink_type, "location": location, "kwargs": kwargs}
        try:
            self.policy.validate_sink_registration(sink_registration)
        except ValueError as exc:
            raise ApiError(status_code=422, detail=str(exc)) from exc

        try:
            session = self.manager.upsert_sinks(
                session_id,
                sinks=[sink_registration],
            )
        except KeyError as exc:
            raise ApiError(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise ApiError(status_code=422, detail=str(exc)) from exc

        return {
            "session_id": session.session_id,
            "sink": session.sinks.get(ref),
            "sinks": list(session.sinks.values()),
        }

    def set_sample_source(self, session_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        location = str(payload.get("location", "")).strip()
        source_type = str(payload.get("type", "hdf")).strip()
        kwargs = payload.get("kwargs", {}) or {}

        if not location:
            raise ApiError(status_code=422, detail="'location' is required.")
        if not source_type:
            raise ApiError(status_code=422, detail="'type' must be non-empty.")
        if not isinstance(kwargs, dict):
            raise ApiError(status_code=422, detail="'kwargs' must be an object when provided.")

        source_registration = {"ref": "sample", "type": source_type, "location": location, "kwargs": kwargs}
        try:
            self.policy.validate_source_registration(source_registration)
        except ValueError as exc:
            raise ApiError(status_code=422, detail=str(exc)) from exc

        try:
            session = self.manager.upsert_sources(
                session_id,
                sources=[source_registration],
            )
        except KeyError as exc:
            raise ApiError(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise ApiError(status_code=422, detail=str(exc)) from exc

        return {
            "session_id": session.session_id,
            "source": session.sources.get("sample"),
            "sources": list(session.sources.values()),
        }

    def delete_source(self, session_id: str, ref: str) -> None:
        try:
            existed = self.manager.delete_source(session_id, ref)
        except KeyError as exc:
            raise ApiError(status_code=404, detail=str(exc)) from exc
        if not existed:
            raise ApiError(status_code=404, detail=f"Source '{ref}' not found.")

    def delete_sink(self, session_id: str, ref: str) -> None:
        try:
            existed = self.manager.delete_sink(session_id, ref)
        except KeyError as exc:
            raise ApiError(status_code=404, detail=str(exc)) from exc
        if not existed:
            raise ApiError(status_code=404, detail=f"Sink '{ref}' not found.")

    def put_buffer_source_array(
        self,
        session_id: str,
        source_ref: str,
        data_key: str,
        payload: bytes,
    ) -> dict[str, Any]:
        self._require_session(session_id)
        try:
            self.policy.validate_buffer_upload(payload)
        except ValueError as exc:
            raise ApiError(status_code=413, detail=str(exc)) from exc
        try:
            array = decode_npy(payload)
            self.manager.buffer_store.put_array(session_id, "source", source_ref, data_key, array)
        except Exception as exc:
            raise ApiError(status_code=422, detail=f"Invalid .npy buffer payload: {exc}") from exc
        return {
            "session_id": session_id,
            "kind": "source",
            "ref": source_ref,
            "data_key": data_key.strip("/"),
            "shape": list(array.shape),
            "dtype": str(array.dtype),
        }

    def put_buffer_source_attrs(
        self,
        session_id: str,
        source_ref: str,
        data_key: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        self._require_session(session_id)
        if not isinstance(payload, dict):
            raise ApiError(status_code=422, detail="Buffer attrs payload must be an object.")
        self.manager.buffer_store.put_attrs(session_id, "source", source_ref, data_key, payload)
        return {
            "session_id": session_id,
            "kind": "source",
            "ref": source_ref,
            "data_key": data_key.strip("/"),
            "attrs": dict(payload),
        }

    def put_buffer_source_metadata(
        self,
        session_id: str,
        source_ref: str,
        data_key: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        self._require_session(session_id)
        if not isinstance(payload, dict) or "value" not in payload:
            raise ApiError(status_code=422, detail="Buffer metadata payload must be an object containing 'value'.")
        self.manager.buffer_store.put_metadata(session_id, "source", source_ref, data_key, payload["value"])
        return {
            "session_id": session_id,
            "kind": "source",
            "ref": source_ref,
            "data_key": data_key.strip("/"),
            "value": payload["value"],
        }

    def get_buffer_sink_array(self, session_id: str, sink_ref: str, data_key: str) -> bytes:
        self._require_session(session_id)
        try:
            array = self.manager.buffer_store.get_array(session_id, "sink", sink_ref, data_key)
        except KeyError as exc:
            raise ApiError(status_code=404, detail=str(exc)) from exc
        return encode_npy(array)

    def get_buffer_manifest(self, session_id: str, kind: str, ref: str) -> dict[str, Any]:
        self._require_session(session_id)
        try:
            return self.manager.buffer_store.manifest(session_id, kind, ref)
        except ValueError as exc:
            raise ApiError(status_code=422, detail=str(exc)) from exc

    def get_plot_payload(self, session_id: str, sink_ref: str, plot_id: str) -> dict[str, Any]:
        session = self._require_session(session_id)
        try:
            plot = self.manager.buffer_store.get_metadata(session_id, "sink", sink_ref, plot_id)
        except KeyError as exc:
            raise ApiError(status_code=404, detail=str(exc)) from exc
        if not isinstance(plot, dict):
            raise ApiError(status_code=500, detail=f"Stored plot payload {sink_ref!r}/{plot_id!r} is not an object.")
        return {
            "session_id": session_id,
            "sink_ref": sink_ref,
            "plot_id": plot_id,
            "state": session.state,
            "updated_utc": session.updated_utc,
            "latest_run": session.run_history[-1] if session.run_history else None,
            "plot": plot,
        }

    def clear_buffers(
        self,
        session_id: str,
        *,
        kind: str | None = None,
        ref: str | None = None,
        data_key: str | None = None,
    ) -> dict[str, Any]:
        self._require_session(session_id)
        try:
            removed = self.manager.buffer_store.clear(session_id, kind=kind, ref=ref, data_key=data_key)
        except ValueError as exc:
            raise ApiError(status_code=422, detail=str(exc)) from exc
        return {"session_id": session_id, "removed": removed}

    def clear_buffer_sink(self, session_id: str, sink_ref: str) -> dict[str, Any]:
        return self.clear_buffers(session_id, kind="sink", ref=sink_ref)

    def clear_buffer_sink_array(self, session_id: str, sink_ref: str, data_key: str) -> dict[str, Any]:
        return self.clear_buffers(session_id, kind="sink", ref=sink_ref, data_key=data_key)

    def process(self, session_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        request = self._parse_process_request(payload)
        session = self._require_session(session_id)
        self._ensure_required_sources(session)

        effective_mode, mode_note = self._resolve_process_mode(session, request)
        run_id = self._enqueue_process_run(
            session_id,
            mode=request.mode,
            changed_sources=request.changed_sources,
            effective_mode=effective_mode,
        )

        preparation = ProcessPreparation()
        sources: IoSources | None = None
        sinks: IoSinks | None = None
        try:
            pipeline = Pipeline.from_yaml(session.pipeline_yaml or "", registry=self.process_step_registry)
            sources = build_sources_from_session(
                session,
                buffer_store=self.manager.buffer_store,
                runtime_policy=self.policy,
            )
            if request.chunk_output is not None:
                resource = self.chunked_outputs.get(request.chunk_output.output_id)
                active_plan = resource.plan or resource.provisional_plan
                assert active_plan is not None
                request.resolved_source_slices = bind_chunk_source_slices(
                    sources,
                    session,
                    active_plan,
                    request.chunk_output.chunk_spec,
                )
            sinks = build_sinks_from_session(
                session,
                pipeline=pipeline,
                buffer_store=self.manager.buffer_store,
                runtime_policy=self.policy,
            )
            preparation = self._prepare_process_execution(
                session_id=session_id,
                run_id=run_id,
                session=session,
                pipeline=pipeline,
                request=request,
                effective_mode=effective_mode,
            )
            if preparation.early_response is not None:
                return preparation.early_response

            result, elapsed_s = self._execute_process_run(
                session=session,
                pipeline=pipeline,
                sources=sources,
                sinks=sinks,
                effective_mode=effective_mode,
                preparation=preparation,
                chunk_spec=None if request.chunk_output is None else request.chunk_output.chunk_spec,
            )
            return self._finalize_process_run(
                session_id=session_id,
                run_id=run_id,
                session=session,
                pipeline=pipeline,
                request=request,
                effective_mode=effective_mode,
                mode_note=mode_note,
                preparation=preparation,
                result=result,
                elapsed_s=elapsed_s,
            )
        except Exception as exc:
            return self._handle_process_failure(
                session=session,
                session_id=session_id,
                request=request,
                effective_mode=effective_mode,
                run_id=run_id,
                preparation=preparation,
                sources=sources,
                sinks=sinks,
                exc=exc,
            )

    def process_dry_run(self, session_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        request = self._parse_process_request(payload)
        session = self._require_session(session_id)
        try:
            return build_dry_run_plan(
                session,
                mode=request.mode,
                changed_sources=request.changed_sources,
                changed_keys=request.changed_keys,
                registry=self.process_step_registry,
            )
        except Exception as exc:
            raise ApiError(
                status_code=500,
                detail={
                    "code": "DRY_RUN_FAILED",
                    "message": str(exc),
                    "details": {"session_id": session_id},
                },
            ) from exc

    def reset(self, session_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        mode = str(payload.get("mode", "")).strip()
        if mode not in {"partial", "full"}:
            raise ApiError(status_code=422, detail="mode must be one of: partial, full.")
        try:
            session = self.manager.reset_session(session_id, mode=mode)
        except KeyError as exc:
            raise ApiError(status_code=404, detail=str(exc)) from exc
        return {"session_id": session.session_id, "mode": mode, "state": session.state}

    def recover(self, session_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        strategy = str(payload.get("strategy", "")).strip()
        if strategy not in {"full_reset_then_process", "full_reset_only"}:
            raise ApiError(status_code=422, detail="Invalid recovery strategy.")
        try:
            self.manager.reset_session(session_id, mode="full")
            if strategy == "full_reset_only":
                session = self._require_session(session_id)
                return {"session_id": session_id, "state": session.state, "strategy": strategy}
        except KeyError as exc:
            raise ApiError(status_code=404, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise ApiError(status_code=409, detail=str(exc)) from exc

        process_payload: dict[str, Any] = {"mode": "full", "changed_sources": list(payload.get("changed_sources", []))}
        if "write_hdf" in payload:
            process_payload["write_hdf"] = payload["write_hdf"]
        if "run_name" in payload:
            process_payload["run_name"] = payload["run_name"]
        process_response = self.process(session_id, process_payload)
        process_response["strategy"] = strategy
        return process_response

    def list_runs(self, session_id: str) -> dict[str, Any]:
        session = self._require_session(session_id)
        return {"session_id": session_id, "runs": session.run_history}

    def get_run(self, session_id: str, run_id: str) -> dict[str, Any]:
        session = self._require_session(session_id)
        for run_meta in session.run_history:
            if run_meta.get("run_id") == run_id:
                return run_meta
        raise ApiError(status_code=404, detail="Run not found.")

    def session_state_event(self, session_id: str) -> dict[str, Any]:
        session = self._require_session(session_id)
        return {
            "event": "session_state_changed",
            "session_id": session_id,
            "payload": {"state": session.state, "active_run_id": session.active_run_id},
        }

    def _require_session(self, session_id: str) -> PipelineSession:
        session = self.manager.get_session(session_id)
        if session is None:
            raise ApiError(status_code=404, detail="Session not found.")
        return session

    def _latest_failed_run(self, session: PipelineSession) -> dict[str, Any] | None:
        for run_meta in reversed(session.run_history):
            if run_meta.get("status") == "failed":
                return dict(run_meta)
        return None

    def _ensure_required_sources(self, session: PipelineSession) -> None:
        missing_refs = missing_required_source_refs(session)
        if not missing_refs:
            return
        raise ApiError(
            status_code=422,
            detail={
                "code": "MISSING_REQUIRED_SOURCES",
                "message": "Session source profile requirements are not satisfied.",
                "details": {
                    "source_profile": session.source_profile,
                    "missing_refs": missing_refs,
                    "required_refs": session.required_source_refs,
                },
            },
        )

    def _parse_process_request(self, payload: dict[str, Any]) -> ProcessRequest:
        mode = str(payload.get("mode", "")).strip()
        if mode not in {"partial", "full", "auto"}:
            raise ApiError(status_code=422, detail="mode must be one of: partial, full, auto.")

        changed_sources = list(payload.get("changed_sources") or [])
        changed_keys = list(payload.get("changed_keys") or [])

        write_hdf_raw = payload.get("write_hdf")
        write_hdf = dict(write_hdf_raw) if isinstance(write_hdf_raw, dict) else None
        if write_hdf is not None and write_hdf.get("path"):
            try:
                self.policy.validate_write_hdf_path(write_hdf["path"])
            except ValueError as exc:
                raise ApiError(status_code=422, detail=str(exc)) from exc
        run_name_raw = payload.get("run_name")
        run_name = str(run_name_raw) if run_name_raw is not None else None
        chunk_output: ChunkPublishRequest | None = None
        chunk_output_raw = payload.get("chunk_output")
        if chunk_output_raw is not None:
            if not isinstance(chunk_output_raw, dict):
                raise ApiError(status_code=422, detail="chunk_output must be an object.")
            output_id = str(chunk_output_raw.get("output_id", "")).strip()
            spec_raw = chunk_output_raw.get("chunk_spec")
            chunk_id = str(chunk_output_raw.get("chunk_id", "")).strip()
            if not output_id or (isinstance(spec_raw, dict) == bool(chunk_id)):
                raise ApiError(
                    status_code=422,
                    detail="chunk_output requires output_id and exactly one of chunk_spec or chunk_id.",
                )
            try:
                resource = self.chunked_outputs.get(output_id)
                if isinstance(spec_raw, dict):
                    if resource.plan is None:
                        raise ValueError("An awaiting-schema output accepts chunk_id, not a complete chunk_spec.")
                    chunk_spec: ChunkSpec | ProvisionalChunkSpec = ChunkSpec.from_dict(spec_raw)
                    chunk_spec.validate_for_plan(resource.plan)
                else:
                    chunk_spec = self.chunked_outputs.execution_chunk(output_id, chunk_id)
            except KeyError as exc:
                raise ApiError(status_code=404, detail=str(exc)) from exc
            except (TypeError, ValueError) as exc:
                raise ApiError(status_code=422, detail=str(exc)) from exc
            chunk_output = ChunkPublishRequest(output_id=output_id, chunk_spec=chunk_spec)
            active_plan = resource.plan or resource.provisional_plan
            assert active_plan is not None
            for binding in active_plan.source_bindings:
                if binding.role != "static" and binding.source_ref not in changed_sources:
                    changed_sources.append(binding.source_ref)
            if not active_plan.source_bindings:
                driver_ref = str(active_plan.driver.get("source", "")).partition("::")[0].strip()
                if driver_ref and driver_ref not in changed_sources:
                    changed_sources.append(driver_ref)
        if mode == "partial" and not changed_sources and not changed_keys:
            raise ApiError(status_code=422, detail="partial mode requires changed_sources or changed_keys.")
        return ProcessRequest(
            mode=mode,
            changed_sources=changed_sources,
            changed_keys=changed_keys,
            write_hdf=write_hdf,
            run_name=run_name,
            rollback_snapshot=bool(payload.get("rollback_snapshot", True)),
            chunk_output=chunk_output,
        )

    def _resolve_process_mode(self, session: PipelineSession, request: ProcessRequest) -> tuple[str, str | None]:
        effective_mode, mode_note = resolve_effective_mode(request.mode)
        if effective_mode == "partial" and session.processing_data is None:
            effective_mode = "full"
            mode_note = "No previous ProcessingData snapshot available; executed full rerun."
        if request.mode == "auto" and not request.changed_sources and not request.changed_keys:
            effective_mode = "full"
            mode_note = "Auto mode without changed_sources/changed_keys defaults to full rerun."
        return effective_mode, mode_note

    def _enqueue_process_run(
        self,
        session_id: str,
        *,
        mode: str,
        changed_sources: list[str],
        effective_mode: str,
    ) -> str:
        try:
            run_meta = self.manager.enqueue_run(
                session_id,
                mode=mode,
                changed_sources=changed_sources,
                effective_mode=effective_mode,
            )
        except KeyError as exc:
            raise ApiError(status_code=404, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise ApiError(status_code=409, detail=str(exc)) from exc
        return str(run_meta["run_id"])

    def _prepare_process_execution(
        self,
        *,
        session_id: str,
        run_id: str,
        session: PipelineSession,
        pipeline: Pipeline,
        request: ProcessRequest,
        effective_mode: str,
    ) -> ProcessPreparation:
        if effective_mode != "partial":
            return ProcessPreparation()

        selected_step_ids = find_dirty_step_ids(
            pipeline,
            changed_sources=request.changed_sources,
            changed_keys=request.changed_keys,
        )
        topo_ids = ordered_step_ids(pipeline)
        dirty_step_ids_ordered = [step_id for step_id in topo_ids if step_id in selected_step_ids]
        boundary_step_id = dirty_step_ids_ordered[0] if dirty_step_ids_ordered else None

        if not selected_step_ids:
            if request.chunk_output is not None:
                raise RuntimeError("A chunk_output request cannot publish a partial run with no dirty steps.")
            return ProcessPreparation(
                early_response=self._mark_noop_process_run(
                    session_id=session_id,
                    run_id=run_id,
                    changed_keys=request.changed_keys,
                    topo_ids=topo_ids,
                )
            )

        snapshot_before_partial = (
            deepcopy(session.processing_data)
            if request.rollback_snapshot and session.processing_data is not None
            else None
        )
        return ProcessPreparation(
            selected_step_ids=selected_step_ids,
            snapshot_before_partial=snapshot_before_partial,
            dirty_step_ids_ordered=dirty_step_ids_ordered,
            boundary_step_id=boundary_step_id,
        )

    def _mark_noop_process_run(
        self,
        *,
        session_id: str,
        run_id: str,
        changed_keys: list[str],
        topo_ids: list[str],
    ) -> dict[str, Any]:
        run_meta = self.manager.mark_run_succeeded(
            session_id,
            run_id,
            details={
                "status": "succeeded",
                "executed_steps": [],
                "num_steps": 0,
                "note": "No pipeline steps matched changed_sources.",
                "changed_keys": changed_keys,
                "dirty_steps": [],
                "skipped_steps": topo_ids,
                "step_durations_s": {},
                "elapsed_s": 0.0,
            },
        )
        return {
            "session_id": session_id,
            "run_id": run_id,
            "state": "idle",
            "status": run_meta.get("status"),
            "effective_mode": run_meta.get("effective_mode"),
            "note": run_meta.get("note"),
        }

    def _execute_process_run(
        self,
        *,
        session: PipelineSession,
        pipeline: Pipeline,
        sources: IoSources,
        sinks: IoSinks,
        effective_mode: str,
        preparation: ProcessPreparation,
        chunk_spec: ChunkSpec | ProvisionalChunkSpec | None = None,
    ) -> tuple[RunResult, float]:
        reuse_processing_data = effective_mode == "partial" and session.processing_data is not None
        run_t0 = perf_counter()
        result = run_pipeline_job(
            pipeline,
            sources=sources,
            sinks=sinks,
            processing_data=session.processing_data if reuse_processing_data else None,
            trace=session.trace_enabled,
            trace_watch=session.trace_watch,
            tracer_kwargs={
                "snapshot_processing_data": session.trace_snapshot_processing_data,
                "snapshot_step_ids": set(session.trace_snapshot_step_ids),
            },
            selected_step_ids=preparation.selected_step_ids,
            capture_partial_on_error=True,
            chunk_spec=chunk_spec,
        )
        elapsed_s = perf_counter() - run_t0
        session.processing_data = result.processing_data
        return result, elapsed_s

    def _finalize_process_run(
        self,
        *,
        session_id: str,
        run_id: str,
        session: PipelineSession,
        pipeline: Pipeline,
        request: ProcessRequest,
        effective_mode: str,
        mode_note: str | None,
        preparation: ProcessPreparation,
        result: RunResult,
        elapsed_s: float,
    ) -> dict[str, Any]:
        hdf_out_path = write_hdf_output(
            request.write_hdf,
            run_name=request.run_name or run_id,
            result=result,
            pipeline_yaml=session.pipeline_yaml or "",
            runtime_policy=self.policy,
        )

        chunk_write = None
        published_chunk_spec: ChunkSpec | None = None
        if request.chunk_output is not None:
            try:
                chunk_write = self.chunked_outputs.write_chunk(
                    request.chunk_output.output_id,
                    result.processing_data,
                    chunk=request.chunk_output.chunk_spec,
                    execution_metadata={
                        "session_id": session_id,
                        "run_id": run_id,
                        "effective_mode": effective_mode,
                        "sources": _source_provenance(session),
                        "source_slices": request.resolved_source_slices,
                        **request.chunk_output.chunk_spec.identity_dict(),
                    },
                    pipeline_spec=_pipeline_provenance_spec(result.pipeline),
                    pipeline_yaml=session.pipeline_yaml or "",
                    trace_events=_pipeline_trace_events(result.pipeline) if session.trace_enabled else None,
                )
                if isinstance(request.chunk_output.chunk_spec, ProvisionalChunkSpec):
                    resolved = self.chunked_outputs.execution_chunk(
                        request.chunk_output.output_id,
                        request.chunk_output.chunk_spec.chunk_id,
                    )
                    assert isinstance(resolved, ChunkSpec)
                    published_chunk_spec = resolved
                else:
                    published_chunk_spec = request.chunk_output.chunk_spec
            except Exception as exc:
                raise _ChunkPublicationError(
                    request.chunk_output.output_id,
                    request.chunk_output.chunk_spec,
                    exc,
                ) from exc

        topo_ids = ordered_step_ids(pipeline)
        executed_set = set(result.executed_steps)
        skipped_steps = [step_id for step_id in topo_ids if step_id not in executed_set]
        details: dict[str, Any] = {
            "status": "succeeded",
            "executed_steps": result.executed_steps,
            "num_steps": len(result.executed_steps),
            "changed_sources": request.changed_sources,
            "changed_keys": request.changed_keys,
            "skipped_steps": skipped_steps,
            "step_durations_s": result.step_durations,
            "elapsed_s": elapsed_s,
            "rollback_snapshot": request.rollback_snapshot,
        }
        if mode_note:
            details["note"] = mode_note
        if preparation.dirty_step_ids_ordered:
            details["dirty_steps"] = preparation.dirty_step_ids_ordered
        if preparation.boundary_step_id is not None:
            details["checkpoint_boundary_step"] = preparation.boundary_step_id
        if hdf_out_path is not None:
            details["hdf_output"] = hdf_out_path
        if chunk_write is not None:
            assert published_chunk_spec is not None
            details["chunk_output"] = {
                "output_id": request.chunk_output.output_id,
                **published_chunk_spec.identity_dict(),
                **chunk_write.to_dict(),
            }
            if isinstance(request.chunk_output.chunk_spec, ProvisionalChunkSpec):
                details["chunk_output"]["provisional_hash"] = request.chunk_output.chunk_spec.provisional_hash
        if request.resolved_source_slices:
            details["source_slices"] = request.resolved_source_slices
        trace_report = self._trace_report(result)
        if trace_report is not None:
            details["trace_report"] = trace_report

        run_meta = self.manager.mark_run_succeeded(session_id, run_id, details=details)
        response = {
            "session_id": session_id,
            "run_id": run_id,
            "state": "idle",
            "status": run_meta.get("status"),
            "effective_mode": effective_mode,
            "note": run_meta.get("note"),
            "hdf_output": run_meta.get("hdf_output"),
        }
        if chunk_write is not None:
            response["chunk_output"] = run_meta["chunk_output"]
        return response

    def _handle_process_failure(
        self,
        *,
        session: PipelineSession,
        session_id: str,
        request: ProcessRequest,
        effective_mode: str,
        run_id: str,
        preparation: ProcessPreparation,
        sources: IoSources | None,
        sinks: IoSinks | None,
        exc: Exception,
    ) -> dict[str, Any]:
        publication_failure = isinstance(exc, _ChunkPublicationError)
        if not publication_failure and effective_mode == "partial" and preparation.snapshot_before_partial is not None:
            session.processing_data = preparation.snapshot_before_partial

        original_exception = (
            exc.original_exception if isinstance(exc, (PipelineRunError, _ChunkPublicationError)) else exc
        )
        error_code = (
            "CHUNK_WRITE_FAILED"
            if publication_failure
            else ("PARTIAL_RUN_FAILED" if effective_mode == "partial" else "RUN_FAILED")
        )
        error_details: dict[str, Any] = {
            "exception_type": type(original_exception).__name__,
            "traceback": traceback.format_exc(),
        }
        if isinstance(exc, PipelineRunError):
            if exc.failed_step_id is not None:
                error_details["failed_step_id"] = exc.failed_step_id
            trace_report = self._trace_report(exc.result)
            if trace_report is not None:
                error_details["trace_report"] = trace_report
        if isinstance(exc, _ChunkPublicationError):
            error_details.update(
                {
                    "processing_succeeded": True,
                    "output_id": exc.output_id,
                    "chunk_identity": exc.chunk_spec.identity_dict(),
                }
            )
        elif request.chunk_output is not None:
            error_details.update(
                {
                    "output_id": request.chunk_output.output_id,
                    "chunk_identity": request.chunk_output.chunk_spec.identity_dict(),
                }
            )

        self.manager.mark_run_failed(
            session_id,
            run_id,
            code=error_code,
            message=str(exc),
            details=error_details,
        )
        if (
            not publication_failure
            and request.mode == "auto"
            and effective_mode == "partial"
            and sources is not None
            and sinks is not None
        ):
            return self._run_auto_fallback(
                session=session,
                session_id=session_id,
                request=request,
                sources=sources,
                sinks=sinks,
                recovered_from_run_id=run_id,
                fallback_reason=exc,
            )
        raise ApiError(
            status_code=500,
            detail={
                "code": error_code,
                "message": str(exc),
                "details": {"session_id": session_id, "run_id": run_id},
            },
        ) from exc

    def _run_auto_fallback(
        self,
        *,
        session: PipelineSession,
        session_id: str,
        request: ProcessRequest,
        sources: IoSources,
        sinks: IoSinks,
        recovered_from_run_id: str,
        fallback_reason: Exception,
    ) -> dict[str, Any]:
        fallback_run = self.manager.enqueue_run(
            session_id,
            mode="full",
            changed_sources=request.changed_sources,
            effective_mode="full",
        )
        fallback_id = str(fallback_run["run_id"])
        try:
            fallback_pipeline = Pipeline.from_yaml(session.pipeline_yaml or "", registry=self.process_step_registry)
            fallback_t0 = perf_counter()
            fallback_result = run_pipeline_job(
                fallback_pipeline,
                sources=sources,
                sinks=sinks,
                processing_data=None,
                trace=session.trace_enabled,
                trace_watch=session.trace_watch,
                tracer_kwargs={
                    "snapshot_processing_data": session.trace_snapshot_processing_data,
                    "snapshot_step_ids": set(session.trace_snapshot_step_ids),
                },
                capture_partial_on_error=True,
                chunk_spec=None if request.chunk_output is None else request.chunk_output.chunk_spec,
            )
            fallback_elapsed = perf_counter() - fallback_t0
            session.processing_data = fallback_result.processing_data

            hdf_out_path = write_hdf_output(
                request.write_hdf,
                run_name=request.run_name or fallback_id,
                result=fallback_result,
                pipeline_yaml=session.pipeline_yaml or "",
                runtime_policy=self.policy,
            )

            chunk_write = None
            published_chunk_spec: ChunkSpec | None = None
            if request.chunk_output is not None:
                try:
                    chunk_write = self.chunked_outputs.write_chunk(
                        request.chunk_output.output_id,
                        fallback_result.processing_data,
                        chunk=request.chunk_output.chunk_spec,
                        execution_metadata={
                            "session_id": session_id,
                            "run_id": fallback_id,
                            "effective_mode": "full",
                            "sources": _source_provenance(session),
                            "source_slices": request.resolved_source_slices,
                            **request.chunk_output.chunk_spec.identity_dict(),
                        },
                        pipeline_spec=fallback_result.pipeline.to_spec(),
                        pipeline_yaml=session.pipeline_yaml or "",
                    )
                    if isinstance(request.chunk_output.chunk_spec, ProvisionalChunkSpec):
                        resolved = self.chunked_outputs.execution_chunk(
                            request.chunk_output.output_id,
                            request.chunk_output.chunk_spec.chunk_id,
                        )
                        assert isinstance(resolved, ChunkSpec)
                        published_chunk_spec = resolved
                    else:
                        published_chunk_spec = request.chunk_output.chunk_spec
                except Exception as exc:
                    raise _ChunkPublicationError(
                        request.chunk_output.output_id,
                        request.chunk_output.chunk_spec,
                        exc,
                    ) from exc

            fallback_topo_ids = ordered_step_ids(fallback_result.pipeline)
            fallback_executed_set = set(fallback_result.executed_steps)
            fallback_skipped = [step_id for step_id in fallback_topo_ids if step_id not in fallback_executed_set]
            trace_report = self._trace_report(fallback_result)
            details: dict[str, Any] = {
                "status": "succeeded",
                "executed_steps": fallback_result.executed_steps,
                "num_steps": len(fallback_result.executed_steps),
                "note": "Auto fallback succeeded after partial failure.",
                "recovered_from_run_id": recovered_from_run_id,
                "fallback_reason": str(fallback_reason),
                "changed_sources": request.changed_sources,
                "changed_keys": request.changed_keys,
                "skipped_steps": fallback_skipped,
                "step_durations_s": fallback_result.step_durations,
                "elapsed_s": fallback_elapsed,
            }
            if trace_report is not None:
                details["trace_report"] = trace_report
            if hdf_out_path:
                details["hdf_output"] = hdf_out_path
            if chunk_write is not None:
                assert published_chunk_spec is not None
                details["chunk_output"] = {
                    "output_id": request.chunk_output.output_id,
                    **published_chunk_spec.identity_dict(),
                    **chunk_write.to_dict(),
                }
                if isinstance(request.chunk_output.chunk_spec, ProvisionalChunkSpec):
                    details["chunk_output"]["provisional_hash"] = request.chunk_output.chunk_spec.provisional_hash
            if request.resolved_source_slices:
                details["source_slices"] = request.resolved_source_slices

            done = self.manager.mark_run_succeeded(
                session_id,
                fallback_id,
                details=details,
            )
            response = {
                "session_id": session_id,
                "run_id": fallback_id,
                "state": "idle",
                "status": done.get("status"),
                "effective_mode": done.get("effective_mode"),
                "note": done.get("note"),
                "recovered_from_run_id": recovered_from_run_id,
                "fallback_reason": done.get("fallback_reason"),
                "hdf_output": done.get("hdf_output"),
            }
            if chunk_write is not None:
                response["chunk_output"] = done["chunk_output"]
            return response
        except Exception as fallback_exc:
            original_exception = (
                fallback_exc.original_exception
                if isinstance(fallback_exc, (PipelineRunError, _ChunkPublicationError))
                else fallback_exc
            )
            error_details: dict[str, Any] = {
                "exception_type": type(original_exception).__name__,
                "traceback": traceback.format_exc(),
                "recovered_from_run_id": recovered_from_run_id,
            }
            if isinstance(fallback_exc, PipelineRunError):
                if fallback_exc.failed_step_id is not None:
                    error_details["failed_step_id"] = fallback_exc.failed_step_id
                trace_report = self._trace_report(fallback_exc.result)
                if trace_report is not None:
                    error_details["trace_report"] = trace_report

            publication_failure = isinstance(fallback_exc, _ChunkPublicationError)
            if publication_failure:
                error_details.update(
                    {
                        "processing_succeeded": True,
                        "output_id": fallback_exc.output_id,
                        "chunk_identity": fallback_exc.chunk_spec.identity_dict(),
                    }
                )
            failure_code = "CHUNK_WRITE_FAILED" if publication_failure else "FULL_RUN_FAILED"
            self.manager.mark_run_failed(
                session_id,
                fallback_id,
                code=failure_code,
                message=str(fallback_exc),
                details=error_details,
            )
            raise ApiError(
                status_code=500,
                detail={
                    "code": failure_code,
                    "message": str(fallback_exc),
                    "details": {"session_id": session_id, "run_id": fallback_id},
                },
            ) from fallback_exc

    def _trace_report(self, result: RunResult) -> str | None:
        if result.tracer is None:
            return None
        return result.tracer.last_report(
            TRACE_REPORT_LINES,
            renderer=PlainUnicodeRenderer(wrap_in_markdown_codeblock=False),
        )
