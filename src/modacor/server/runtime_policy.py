# SPDX-License-Identifier: BSD-3-Clause
# /usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from modacor.runner.process_step_registry import ProcessStepRegistry

__all__ = ["RuntimePolicy"]


_FILE_SOURCE_TYPES = {"csv", "hdf", "yaml"}
_FILE_SINK_TYPES = {"csv", "hdf", "hdf_processing"}


def _normalise_roots(roots: tuple[Path | str, ...]) -> tuple[Path, ...]:
    return tuple(Path(root).expanduser().resolve(strict=False) for root in roots)


def _normalise_path(path: Path | str) -> Path:
    return Path(path).expanduser().resolve(strict=False)


def _is_relative_to(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def _format_roots(roots: tuple[Path, ...]) -> str:
    return ", ".join(str(root) for root in roots) or "<none>"


@dataclass(slots=True)
class RuntimePolicy:
    """
    Runtime API trust-boundary policy.

    Defaults preserve trusted/local behavior. Use ``RuntimePolicy.restricted()``
    for containerized or network-facing deployments.
    """

    allow_pipeline_yaml_path: bool = True
    allow_process_step_filesystem_discovery: bool = True
    allow_custom_io_class_path: bool = True
    require_io_path_roots: bool = False
    pipeline_yaml_read_roots: tuple[Path | str, ...] = field(default_factory=tuple)
    source_read_roots: tuple[Path | str, ...] = field(default_factory=tuple)
    sink_write_roots: tuple[Path | str, ...] = field(default_factory=tuple)
    custom_source_classes: Mapping[str, type] = field(default_factory=dict)
    custom_sink_classes: Mapping[str, type] = field(default_factory=dict)
    max_sessions: int | None = None
    max_pipeline_yaml_bytes: int | None = None
    max_buffer_upload_bytes: int | None = None

    def __post_init__(self) -> None:
        self.pipeline_yaml_read_roots = _normalise_roots(tuple(self.pipeline_yaml_read_roots))
        self.source_read_roots = _normalise_roots(tuple(self.source_read_roots))
        self.sink_write_roots = _normalise_roots(tuple(self.sink_write_roots))
        for name, value in (
            ("max_sessions", self.max_sessions),
            ("max_pipeline_yaml_bytes", self.max_pipeline_yaml_bytes),
            ("max_buffer_upload_bytes", self.max_buffer_upload_bytes),
        ):
            if value is not None and value < 1:
                raise ValueError(f"{name} must be a positive integer or None.")

    @classmethod
    def trusted(cls, **overrides: Any) -> "RuntimePolicy":
        """Return the backward-compatible local/trusted policy."""

        return cls(**overrides)

    @classmethod
    def restricted(cls, **overrides: Any) -> "RuntimePolicy":
        """Return the recommended policy baseline for network-facing services."""

        defaults = {
            "allow_pipeline_yaml_path": False,
            "allow_process_step_filesystem_discovery": False,
            "allow_custom_io_class_path": False,
            "require_io_path_roots": True,
        }
        defaults.update(overrides)
        return cls(**defaults)

    def create_process_step_registry(self) -> ProcessStepRegistry:
        """Build the process-step registry used to parse server pipelines."""

        return ProcessStepRegistry(
            allow_filesystem_discovery=self.allow_process_step_filesystem_discovery,
        )

    def validate_session_capacity(self, current_sessions: int) -> None:
        if self.max_sessions is None:
            return
        if current_sessions >= self.max_sessions:
            raise RuntimeError(
                f"Runtime policy max_sessions={self.max_sessions} has been reached; "
                "delete an existing session or raise the configured limit."
            )

    def validate_pipeline_yaml_text(self, yaml_text: str) -> None:
        self._validate_bytes(
            len(yaml_text.encode("utf-8")),
            self.max_pipeline_yaml_bytes,
            "pipeline.yaml_text",
        )

    def validate_pipeline_yaml_path(self, yaml_path: Path | str) -> Path:
        if not self.allow_pipeline_yaml_path:
            raise ValueError(
                "Runtime policy disables pipeline.yaml_path. Submit pipeline.yaml_text, "
                "or run the service with a trusted policy for local-only use."
            )
        path = _normalise_path(yaml_path)
        roots = self.pipeline_yaml_read_roots or self.source_read_roots
        self._validate_path_roots(path, roots, "pipeline.yaml_path", "read")
        if self.max_pipeline_yaml_bytes is not None:
            try:
                size = path.stat().st_size
            except OSError:
                size = None
            if size is not None:
                self._validate_bytes(size, self.max_pipeline_yaml_bytes, "pipeline.yaml_path")
        return path

    def validate_source_registration(self, registration: Mapping[str, Any]) -> None:
        source_type = str(registration.get("type", "")).strip().lower()
        self._validate_custom_io("source", registration)
        if source_type in _FILE_SOURCE_TYPES:
            self._validate_path_roots(
                _normalise_path(str(registration.get("location", ""))),
                self.source_read_roots,
                "source.location",
                "read",
            )

    def validate_sink_registration(self, registration: Mapping[str, Any]) -> None:
        sink_type = str(registration.get("type", "")).strip().lower()
        self._validate_custom_io("sink", registration)
        if sink_type in _FILE_SINK_TYPES:
            self._validate_path_roots(
                _normalise_path(str(registration.get("location", ""))),
                self.sink_write_roots,
                "sink.location",
                "write",
            )

    def validate_write_hdf_path(self, path: Path | str) -> None:
        self._validate_path_roots(_normalise_path(path), self.sink_write_roots, "write_hdf.path", "write")

    def validate_buffer_upload(self, payload: bytes) -> None:
        self._validate_bytes(len(payload), self.max_buffer_upload_bytes, "buffer source array")

    def source_builder_kwargs(self) -> dict[str, Any]:
        return {
            "allow_custom_class_path": self.allow_custom_io_class_path,
            "custom_classes": self.custom_source_classes,
        }

    def sink_builder_kwargs(self) -> dict[str, Any]:
        return {
            "allow_custom_class_path": self.allow_custom_io_class_path,
            "custom_classes": self.custom_sink_classes,
        }

    def _validate_custom_io(self, kind: str, registration: Mapping[str, Any]) -> None:
        reg_type = str(registration.get("type", "")).strip().lower()
        if reg_type != "custom":
            return
        kwargs = registration.get("kwargs", {}) or {}
        if not isinstance(kwargs, Mapping):
            return
        class_alias = kwargs.get("class_alias")
        if class_alias:
            registry = self.custom_source_classes if kind == "source" else self.custom_sink_classes
            if str(class_alias) not in registry:
                raise ValueError(f"Runtime policy rejected custom {kind}: unknown kwargs.class_alias {class_alias!r}.")
            return
        if kwargs.get("class_path") and not self.allow_custom_io_class_path:
            raise ValueError(
                f"Runtime policy disables arbitrary custom {kind} imports through kwargs.class_path. "
                f"Use a registered kwargs.class_alias or a built-in {kind} type."
            )

    def _validate_path_roots(self, path: Path, roots: tuple[Path, ...], field_name: str, access: str) -> None:
        if not roots:
            if self.require_io_path_roots:
                raise ValueError(
                    f"Runtime policy rejected {field_name}={str(path)!r}: no allowed {access} roots are configured."
                )
            return
        if any(_is_relative_to(path, root) for root in roots):
            return
        raise ValueError(
            f"Runtime policy rejected {field_name}={str(path)!r}: path is outside allowed {access} roots "
            f"({_format_roots(roots)})."
        )

    @staticmethod
    def _validate_bytes(actual: int, limit: int | None, field_name: str) -> None:
        if limit is None or actual <= limit:
            return
        raise ValueError(f"Runtime policy rejected {field_name}: {actual} bytes exceeds limit {limit}.")
