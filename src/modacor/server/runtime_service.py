# SPDX-License-Identifier: BSD-3-Clause
"""Runtime API facade and session lifecycle operations."""

from __future__ import annotations

__all__ = ["RuntimeService"]

import logging
from dataclasses import dataclass, field
from typing import Any

from modacor.runner.pipeline import Pipeline
from modacor.runner.process_step_registry import ProcessStepRegistry

from .buffer_service import BufferService
from .chunked_output_service import ChunkedOutputService
from .chunked_outputs import ChunkedOutputManager
from .errors import ApiError
from .process_failure import ProcessFailureHandler
from .process_service import ProcessService
from .runtime_policy import RuntimePolicy
from .session_manager import PipelineSession
from .session_resources import SessionResourceService
from .session_service import SessionService
from .source_profiles import get_source_profile, list_source_profiles

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class RuntimeService(SessionService):
    """Application service for the MoDaCor runtime API."""

    policy: RuntimePolicy = field(default_factory=RuntimePolicy.trusted)
    process_step_registry: ProcessStepRegistry = field(init=False)
    chunked_outputs: ChunkedOutputManager = field(init=False)

    _buffers: BufferService = field(init=False, repr=False)
    _chunked: ChunkedOutputService = field(init=False, repr=False)
    _resources: SessionResourceService = field(init=False, repr=False)
    _failures: ProcessFailureHandler = field(init=False, repr=False)
    _process: ProcessService = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.process_step_registry = self.policy.create_process_step_registry()
        self.chunked_outputs = ChunkedOutputManager(policy=self.policy)
        self._buffers = BufferService(
            manager=self.manager,
            policy=self.policy,
        )
        self._chunked = ChunkedOutputService(
            manager=self.manager,
            policy=self.policy,
            chunked_outputs=self.chunked_outputs,
        )
        self._resources = SessionResourceService(
            manager=self.manager,
            policy=self.policy,
        )
        self._failures = ProcessFailureHandler(
            manager=self.manager,
            policy=self.policy,
            process_step_registry=self.process_step_registry,
            chunked_outputs=self.chunked_outputs,
        )
        self._process = ProcessService(
            manager=self.manager,
            policy=self.policy,
            process_step_registry=self.process_step_registry,
            chunked_outputs=self.chunked_outputs,
            _failures=self._failures,
        )

    def initialize_chunked_output(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._chunked.initialize_chunked_output(payload)

    def reopen_chunked_output(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._chunked.reopen_chunked_output(payload)

    def inspect_chunked_output(self, output_id: str, *, offset: int = 0, limit: int = 100) -> dict[str, Any]:
        return self._chunked.inspect_chunked_output(output_id, offset=offset, limit=limit)

    def finalize_chunked_output(self, output_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        return self._chunked.finalize_chunked_output(output_id, payload)

    def recover_chunked_output(self, output_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        return self._chunked.recover_chunked_output(output_id, payload)

    def detach_chunked_output(self, output_id: str) -> None:
        return self._chunked.detach_chunked_output(output_id)

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
            "max_expanded_pipeline_steps": self.policy.max_expanded_pipeline_steps,
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

    def _latest_failed_run(self, session: PipelineSession) -> dict[str, Any] | None:
        for run_meta in reversed(session.run_history):
            if run_meta.get("status") == "failed":
                return dict(run_meta)
        return None

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
                # API boundary: every path/policy failure retains the existing 422 response.
                logger.warning("Failed to read pipeline yaml_path", exc_info=True, extra={"session_id": session_id})
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
            Pipeline.from_yaml(
                pipeline_yaml,
                registry=self.process_step_registry,
                max_expanded_steps=self.policy.max_expanded_pipeline_steps,
            )
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
        return self._resources.upsert_sources(session_id, payload)

    def upsert_sinks(self, session_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        return self._resources.upsert_sinks(session_id, payload)

    def patch_source(self, session_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        return self._resources.patch_source(session_id, payload)

    def patch_sink(self, session_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        return self._resources.patch_sink(session_id, payload)

    def set_sample_source(self, session_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        return self._resources.set_sample_source(session_id, payload)

    def delete_source(self, session_id: str, ref: str) -> None:
        return self._resources.delete_source(session_id, ref)

    def delete_sink(self, session_id: str, ref: str) -> None:
        return self._resources.delete_sink(session_id, ref)

    def put_buffer_source_array(
        self,
        session_id: str,
        source_ref: str,
        data_key: str,
        payload: bytes,
    ) -> dict[str, Any]:
        return self._buffers.put_buffer_source_array(session_id, source_ref, data_key, payload)

    def put_buffer_source_attrs(
        self,
        session_id: str,
        source_ref: str,
        data_key: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        return self._buffers.put_buffer_source_attrs(session_id, source_ref, data_key, payload)

    def put_buffer_source_metadata(
        self,
        session_id: str,
        source_ref: str,
        data_key: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        return self._buffers.put_buffer_source_metadata(session_id, source_ref, data_key, payload)

    def get_buffer_sink_array(self, session_id: str, sink_ref: str, data_key: str) -> bytes:
        return self._buffers.get_buffer_sink_array(session_id, sink_ref, data_key)

    def get_buffer_manifest(self, session_id: str, kind: str, ref: str) -> dict[str, Any]:
        return self._buffers.get_buffer_manifest(session_id, kind, ref)

    def get_plot_payload(self, session_id: str, sink_ref: str, plot_id: str) -> dict[str, Any]:
        return self._buffers.get_plot_payload(session_id, sink_ref, plot_id)

    def clear_buffers(
        self,
        session_id: str,
        *,
        kind: str | None = None,
        ref: str | None = None,
        data_key: str | None = None,
    ) -> dict[str, Any]:
        return self._buffers.clear_buffers(session_id, kind=kind, ref=ref, data_key=data_key)

    def clear_buffer_sink(self, session_id: str, sink_ref: str) -> dict[str, Any]:
        return self._buffers.clear_buffer_sink(session_id, sink_ref)

    def clear_buffer_sink_array(self, session_id: str, sink_ref: str, data_key: str) -> dict[str, Any]:
        return self._buffers.clear_buffer_sink_array(session_id, sink_ref, data_key)

    def process(self, session_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        return self._process.process(session_id, payload)

    def process_dry_run(self, session_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        return self._process.process_dry_run(session_id, payload)

    def reset(self, session_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        return self._process.reset(session_id, payload)

    def recover(self, session_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        return self._process.recover(session_id, payload)

    def list_runs(self, session_id: str) -> dict[str, Any]:
        return self._process.list_runs(session_id)

    def get_run(self, session_id: str, run_id: str) -> dict[str, Any]:
        return self._process.get_run(session_id, run_id)

    def session_state_event(self, session_id: str) -> dict[str, Any]:
        session = self._require_session(session_id)
        return {
            "event": "session_state_changed",
            "session_id": session_id,
            "payload": {"state": session.state, "active_run_id": session.active_run_id},
        }
