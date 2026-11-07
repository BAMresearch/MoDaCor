# SPDX-License-Identifier: BSD-3-Clause
"""SessionResourceService implementation for the runtime API."""

from __future__ import annotations

__all__ = ["SessionResourceService"]

import logging
from dataclasses import dataclass
from typing import Any

from .errors import ApiError
from .io_diagnostics import check_file_location
from .runtime_policy import RuntimePolicy
from .session_service import SessionService

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class SessionResourceService(SessionService):
    """SessionResourceService operations for the runtime facade."""

    policy: RuntimePolicy

    def _log_io_registration(
        self, session_id: str, kind: str, registration: dict[str, Any], *, action: str = "registered"
    ) -> None:
        logger.info(
            "%s %s session_id=%s ref=%s type=%s location=%s",
            kind.capitalize(),
            action,
            session_id,
            registration["ref"],
            registration["type"],
            registration["location"],
            extra={
                "session_id": session_id,
                "ref": registration["ref"],
                "kind": kind,
                "io_type": registration["type"],
                "location": registration["location"],
            },
        )

    def _registration_warnings(
        self, session_id: str, kind: str, registrations: list[dict[str, Any]]
    ) -> list[dict[str, str]]:
        warnings = []
        for registration in registrations:
            try:
                check_file_location(kind=kind, io_type=registration["type"], location=registration["location"])
            except OSError as exc:
                warning = {
                    "ref": registration["ref"],
                    "kind": kind,
                    "location": registration["location"],
                    "problem": str(exc),
                }
                warnings.append(warning)
                logger.warning(
                    "I/O registration warning session_id=%s ref=%s kind=%s location=%s problem=%s",
                    session_id,
                    warning["ref"],
                    kind,
                    warning["location"],
                    warning["problem"],
                    extra={"session_id": session_id, **warning},
                )
        return warnings

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
        for source in sources:
            self._log_io_registration(session_id, "source", session.sources[str(source["ref"]).strip()])
        warnings = self._registration_warnings(
            session_id, "source", [session.sources[str(item["ref"]).strip()] for item in sources]
        )
        return {"session_id": session.session_id, "sources": list(session.sources.values()), "warnings": warnings}

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
        for sink in sinks:
            self._log_io_registration(session_id, "sink", session.sinks[str(sink["ref"]).strip()])
        warnings = self._registration_warnings(
            session_id, "sink", [session.sinks[str(item["ref"]).strip()] for item in sinks]
        )
        return {"session_id": session.session_id, "sinks": list(session.sinks.values()), "warnings": warnings}

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

        self._log_io_registration(session_id, "source", session.sources[ref], action="updated")
        warnings = self._registration_warnings(session_id, "source", [session.sources[ref]])
        return {
            "warnings": warnings,
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

        self._log_io_registration(session_id, "sink", session.sinks[ref], action="updated")
        warnings = self._registration_warnings(session_id, "sink", [session.sinks[ref]])
        return {
            "warnings": warnings,
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

        self._log_io_registration(session_id, "source", session.sources["sample"], action="updated")
        warnings = self._registration_warnings(session_id, "source", [session.sources["sample"]])
        return {
            "warnings": warnings,
            "session_id": session.session_id,
            "source": session.sources.get("sample"),
            "sources": list(session.sources.values()),
        }

    def delete_source(self, session_id: str, ref: str) -> None:
        session = self.manager.get_session(session_id)
        registration = None if session is None else session.sources.get(ref)
        try:
            existed = self.manager.delete_source(session_id, ref)
        except KeyError as exc:
            raise ApiError(status_code=404, detail=str(exc)) from exc
        if not existed:
            raise ApiError(status_code=404, detail=f"Source '{ref}' not found.")
        if registration is not None:
            self._log_io_registration(session_id, "source", registration, action="deleted")

    def delete_sink(self, session_id: str, ref: str) -> None:
        session = self.manager.get_session(session_id)
        registration = None if session is None else session.sinks.get(ref)
        try:
            existed = self.manager.delete_sink(session_id, ref)
        except KeyError as exc:
            raise ApiError(status_code=404, detail=str(exc)) from exc
        if not existed:
            raise ApiError(status_code=404, detail=f"Sink '{ref}' not found.")
        if registration is not None:
            self._log_io_registration(session_id, "sink", registration, action="deleted")
