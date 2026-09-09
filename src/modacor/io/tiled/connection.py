# SPDX-License-Identifier: BSD-3-Clause

"""Lazy Tiled connections shared by sources and sinks."""

from __future__ import annotations

from typing import Any


def connect_tiled(location: str | dict[str, Any] | None, kwargs: dict[str, Any]) -> Any:
    if location is None:
        return None
    if isinstance(location, dict):
        for key in ("client", "node"):
            if location.get(key) is not None:
                return location[key]
        connection_kwargs = {}
        for key in ("kwargs", "connection_kwargs"):
            extra = location.get(key, {})
            if not isinstance(extra, dict):
                raise TypeError(f"resource_location.{key} must be a dictionary.")
            connection_kwargs.update(extra)
        connection_kwargs.update(kwargs)
        for key in ("uri", "from_uri", "profile", "from_profile"):
            if key in location:
                descriptor = location[key]
                if not isinstance(descriptor, str) or not descriptor.strip():
                    raise ValueError(f"resource_location.{key} must be a non-empty string.")
                if key in {"profile", "from_profile"}:
                    descriptor = "profile:" + descriptor
                return connect_tiled(descriptor, connection_kwargs)
        raise ValueError("resource_location mapping did not contain a recognised connection descriptor.")
    if not isinstance(location, str):
        raise TypeError("resource_location must be a string, mapping, or None.")
    location = location.strip()
    if not location:
        raise ValueError("resource_location must not be empty.")

    try:
        from tiled.client import from_profile, from_uri
    except ImportError as exc:
        raise ImportError("Tiled I/O requires the optional extra: pip install 'modacor[tiled]'.") from exc

    for prefix in ("profile://", "profile:"):
        if location.startswith(prefix):
            return from_profile(location[len(prefix) :], **kwargs)
    return from_uri(location, **kwargs)
