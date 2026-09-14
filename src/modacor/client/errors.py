# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

__all__ = ["RuntimeAPIError"]


class RuntimeAPIError(RuntimeError):
    """A structured runtime-service or transport failure."""

    def __init__(
        self,
        *,
        status: int | None,
        code: str | None,
        message: str,
        details: Any = None,
        endpoint: str,
        method: str,
    ) -> None:
        self.status = status
        self.status_code = status
        self.code = code
        self.message = str(message)
        self.details = details
        self.endpoint = endpoint
        self.method = method.upper()
        status_text = "transport error" if status is None else f"HTTP {status}"
        code_text = f" [{code}]" if code else ""
        super().__init__(f"{self.method} {endpoint}: {status_text}{code_text}: {self.message}")


def error_fields(payload: Any, fallback: str) -> tuple[str | None, str, Any]:
    """Extract MoDaCor's structured error fields from an HTTP response body."""

    detail = payload.get("detail", payload) if isinstance(payload, Mapping) else payload
    if isinstance(detail, Mapping):
        code = detail.get("code")
        message = detail.get("message") or detail.get("detail") or fallback
        details = detail.get("details")
        if details is None:
            details = {key: value for key, value in detail.items() if key not in {"code", "message", "detail"}}
            if not details:
                details = None
        return None if code is None else str(code), str(message), details
    if detail not in (None, ""):
        return None, str(detail), None
    return None, fallback, None
