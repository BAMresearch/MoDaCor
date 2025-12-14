# SPDX-License-Identifier: BSD-3-Clause
# /usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

__coding__ = "utf-8"
__authors__ = ["Brian R. Pauw"]  # add names to the list as appropriate
__copyright__ = "Copyright 2025, The MoDaCor team"
__date__ = "13/12/2025"
__status__ = "Development"  # "Development", "Production"
__version__ = "20251213.1"

__all__ = ["TraceEvent"]

import json
from hashlib import sha256
from typing import Any

from attrs import define, field, validators


def _to_jsonable(value: Any) -> Any:
    """
    Convert arbitrary objects into a JSON-serializable structure.

    Rules:
    - dict keys become strings
    - tuples/sets become lists
    - unknown objects become str(value)
    """
    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}

    if isinstance(value, (list, tuple, set)):
        return [_to_jsonable(v) for v in value]

    # Common numpy-like scalars without importing numpy
    if hasattr(value, "item") and callable(getattr(value, "item")):
        try:
            return _to_jsonable(value.item())
        except Exception:
            pass

    return str(value)


def _stable_hash_dict(d: dict[str, Any]) -> str:
    """
    Stable content hash of a dict (order-independent).
    """
    canonical = json.dumps(_to_jsonable(d), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return sha256(canonical.encode("utf-8")).hexdigest()


@define(frozen=True, slots=True)
class TraceEvent:
    """
    A small, UI-friendly trace record for a single executed step.

    Intended to be embedded into Pipeline.to_spec() so graph viewers can show:
      - configuration used by the step
      - what changed (units/dimensionality/shape/NaNs/etc.)
      - optional human messages (later)

    Notes
    -----
    Keep this JSON-friendly and lightweight: no arrays, no heavy objects.
    """

    step_id: str
    module: str
    label: str = ""

    module_path: str = ""
    version: str = ""

    requires_steps: tuple[str, ...] = field(factory=tuple)

    # configuration as used for execution (JSON-friendly)
    config: dict[str, Any] = field(factory=dict)

    # computed stable hash of config
    config_hash: str = field(init=False)

    # dataset key -> { "diff": [...], "prev": {...} | None, "now": {...} }
    # Use a simple key like "sample.signal" or "sample_background.signal"
    datasets: dict[str, Any] = field(factory=dict)

    # reserved for later (MessageHandler, timing, etc.)
    messages: list[dict[str, Any]] = field(factory=list)

    # wall-clock runtime for this step execution (seconds)
    duration_s: float | None = field(default=None, validator=validators.optional(validators.instance_of(float)))

    def __attrs_post_init__(self) -> None:
        object.__setattr__(self, "config_hash", _stable_hash_dict(self.config))

    def to_dict(self) -> dict[str, Any]:
        """
        JSON-serializable representation suitable for Pipeline.to_spec().
        """
        return {
            "step_id": self.step_id,
            "module": self.module,
            "label": self.label,
            "module_path": self.module_path,
            "version": self.version,
            "requires_steps": list(self.requires_steps),
            "config": _to_jsonable(self.config),
            "config_hash": self.config_hash,
            "duration_s": self.duration_s,
            "datasets": _to_jsonable(self.datasets),
            "messages": _to_jsonable(self.messages),
        }
