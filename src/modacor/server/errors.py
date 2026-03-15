# SPDX-License-Identifier: BSD-3-Clause
# /usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = ["ApiError"]


@dataclass(slots=True)
class ApiError(Exception):
    """Framework-agnostic error payload used by the runtime service layer."""

    status_code: int
    detail: Any
