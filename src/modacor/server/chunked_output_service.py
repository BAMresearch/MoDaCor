# SPDX-License-Identifier: BSD-3-Clause
"""ChunkedOutputService implementation for the runtime API."""

from __future__ import annotations

__all__ = ["ChunkedOutputService"]

from dataclasses import dataclass
from typing import Any

from modacor.io.chunk_planning import resolve_chunk_input_plan
from modacor.io.chunking import ChunkPlan, ProvisionalChunkPlan
from modacor.io.io_sources import IoSources

from .chunked_outputs import ChunkedOutputManager
from .errors import ApiError, IOBuildError
from .io_utils import build_sources_from_session
from .runtime_policy import RuntimePolicy
from .session_service import SessionService


@dataclass(slots=True)
class ChunkedOutputService(SessionService):
    """ChunkedOutputService operations for the runtime facade."""

    policy: RuntimePolicy
    chunked_outputs: ChunkedOutputManager

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
                    try:
                        sources = build_sources_from_session(
                            session,
                            buffer_store=self.manager.buffer_store,
                            runtime_policy=self.policy,
                        )
                    except IOBuildError as exc:
                        # Preserve this endpoint's existing provisional-plan
                        # ApiError mappings; run failures map IOBuildError below.
                        raise exc.original_exception from exc
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
