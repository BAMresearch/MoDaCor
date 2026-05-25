# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from copy import deepcopy
from threading import RLock
from typing import Any, Literal

import numpy as np
from attrs import define, field

__all__ = ["BufferKind", "RuntimeBufferStore"]

BufferKind = Literal["source", "sink"]


def _normalise_kind(kind: str) -> BufferKind:
    value = str(kind).strip().lower()
    if value in {"source", "sources"}:
        return "source"
    if value in {"sink", "sinks"}:
        return "sink"
    raise ValueError("Buffer kind must be 'source' or 'sink'.")


def _normalise_key(data_key: str) -> str:
    key = str(data_key).strip()
    if not key:
        raise ValueError("Buffer data_key must be non-empty.")
    return key.strip("/")


@define
class RuntimeBufferStore:
    """Thread-safe in-memory storage for runtime buffer sources and sinks."""

    _arrays: dict[tuple[str, BufferKind, str, str], np.ndarray] = field(factory=dict, init=False)
    _attrs: dict[tuple[str, BufferKind, str, str], dict[str, Any]] = field(factory=dict, init=False)
    _metadata: dict[tuple[str, BufferKind, str, str], Any] = field(factory=dict, init=False)
    _lock: RLock = field(factory=RLock, init=False)

    def _entry_key(self, session_id: str, kind: str, ref: str, data_key: str) -> tuple[str, BufferKind, str, str]:
        session = str(session_id).strip()
        reference = str(ref).strip()
        if not session:
            raise ValueError("session_id must be non-empty.")
        if not reference:
            raise ValueError("Buffer ref must be non-empty.")
        return (session, _normalise_kind(kind), reference, _normalise_key(data_key))

    def put_array(self, session_id: str, kind: str, ref: str, data_key: str, array: np.ndarray) -> None:
        key = self._entry_key(session_id, kind, ref, data_key)
        with self._lock:
            self._arrays[key] = np.array(array, copy=True)

    def get_array(self, session_id: str, kind: str, ref: str, data_key: str) -> np.ndarray:
        key = self._entry_key(session_id, kind, ref, data_key)
        with self._lock:
            try:
                return np.array(self._arrays[key], copy=True)
            except KeyError as exc:
                raise KeyError(f"Buffer array not found for {key[1]} '{key[2]}' key '{key[3]}'.") from exc

    def has_array(self, session_id: str, kind: str, ref: str, data_key: str) -> bool:
        key = self._entry_key(session_id, kind, ref, data_key)
        with self._lock:
            return key in self._arrays

    def get_array_shape(self, session_id: str, kind: str, ref: str, data_key: str) -> tuple[int, ...]:
        key = self._entry_key(session_id, kind, ref, data_key)
        with self._lock:
            array = self._arrays.get(key)
            return tuple(array.shape) if array is not None else ()

    def get_array_dtype(self, session_id: str, kind: str, ref: str, data_key: str) -> np.dtype | None:
        key = self._entry_key(session_id, kind, ref, data_key)
        with self._lock:
            array = self._arrays.get(key)
            return None if array is None else array.dtype

    def put_attrs(self, session_id: str, kind: str, ref: str, data_key: str, attrs: dict[str, Any]) -> None:
        key = self._entry_key(session_id, kind, ref, data_key)
        with self._lock:
            self._attrs[key] = dict(attrs)

    def update_attrs(self, session_id: str, kind: str, ref: str, data_key: str, attrs: dict[str, Any]) -> None:
        key = self._entry_key(session_id, kind, ref, data_key)
        with self._lock:
            current = dict(self._attrs.get(key, {}))
            current.update(attrs)
            self._attrs[key] = current

    def get_attrs(self, session_id: str, kind: str, ref: str, data_key: str) -> dict[str, Any]:
        key = self._entry_key(session_id, kind, ref, data_key)
        with self._lock:
            return dict(self._attrs.get(key, {}))

    def put_metadata(self, session_id: str, kind: str, ref: str, data_key: str, value: Any) -> None:
        key = self._entry_key(session_id, kind, ref, data_key)
        with self._lock:
            self._metadata[key] = deepcopy(value)

    def get_metadata(self, session_id: str, kind: str, ref: str, data_key: str) -> Any:
        key = self._entry_key(session_id, kind, ref, data_key)
        with self._lock:
            if key not in self._metadata:
                raise KeyError(f"Buffer metadata not found for {key[1]} '{key[2]}' key '{key[3]}'.")
            return deepcopy(self._metadata[key])

    def manifest(self, session_id: str, kind: str, ref: str) -> dict[str, Any]:
        normal_kind = _normalise_kind(kind)
        session = str(session_id).strip()
        reference = str(ref).strip()
        with self._lock:
            arrays = sorted(
                key for sid, k, r, key in self._arrays if sid == session and k == normal_kind and r == reference
            )
            attrs = sorted(
                key for sid, k, r, key in self._attrs if sid == session and k == normal_kind and r == reference
            )
            metadata = sorted(
                key for sid, k, r, key in self._metadata if sid == session and k == normal_kind and r == reference
            )
            array_details = {}
            for data_key in arrays:
                array = self._arrays[(session, normal_kind, reference, data_key)]
                array_details[data_key] = {"shape": list(array.shape), "dtype": str(array.dtype)}
        return {
            "session_id": session,
            "kind": normal_kind,
            "ref": reference,
            "arrays": arrays,
            "attrs": attrs,
            "metadata": metadata,
            "array_details": array_details,
        }

    def clear(
        self,
        session_id: str,
        *,
        kind: str | None = None,
        ref: str | None = None,
        data_key: str | None = None,
    ) -> int:
        session = str(session_id).strip()
        normal_kind = _normalise_kind(kind) if kind is not None else None
        normal_ref = str(ref).strip() if ref is not None else None
        normal_data_key = _normalise_key(data_key) if data_key is not None else None

        def matches(entry: tuple[str, BufferKind, str, str]) -> bool:
            sid, entry_kind, entry_ref, entry_key = entry
            return (
                sid == session
                and (normal_kind is None or entry_kind == normal_kind)
                and (normal_ref is None or entry_ref == normal_ref)
                and (normal_data_key is None or entry_key == normal_data_key)
            )

        removed = 0
        with self._lock:
            for storage in (self._arrays, self._attrs, self._metadata):
                for key in [entry for entry in storage if matches(entry)]:
                    del storage[key]
                    removed += 1
        return removed
