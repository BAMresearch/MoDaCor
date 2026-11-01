# SPDX-License-Identifier: BSD-3-Clause
# /usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = ["ApiError", "IOBuildError"]


@dataclass(slots=True)
class ApiError(Exception):
    """Framework-agnostic error payload used by the runtime service layer."""

    status_code: int
    detail: Any


class IOBuildError(RuntimeError):
    """Identify the registration whose runtime construction failed."""

    def __init__(self, *, ref: str, kind: str, io_type: str, location: str, cause: Exception) -> None:
        self.ref = ref
        self.kind = kind
        self.io_type = io_type
        self.location = location
        self.original_exception = cause
        super().__init__(
            f"{kind.capitalize()} '{ref}' ({io_type}, {location}) could not be opened: {type(cause).__name__}: {cause}"
        )

    def details(self) -> dict[str, str]:
        return {
            "ref": self.ref,
            "type": self.io_type,
            "location": self.location,
            "exception_type": type(self.original_exception).__name__,
        }
