# SPDX-License-Identifier: BSD-3-Clause
"""Shared session access for runtime services."""

# No postponed annotations: subclass constructors must see the SessionManager class, not a string.

__all__ = ["SessionService"]

from dataclasses import dataclass

from .errors import ApiError
from .session_manager import PipelineSession, SessionManager


@dataclass(slots=True)
class SessionService:
    """Shared session lookup for runtime services."""

    manager: SessionManager

    def _require_session(self, session_id: str) -> PipelineSession:
        session = self.manager.get_session(session_id)
        if session is None:
            raise ApiError(status_code=404, detail="Session not found.")
        return session
