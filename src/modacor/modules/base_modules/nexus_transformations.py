# SPDX-License-Identifier: BSD-3-Clause
# /usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

__coding__ = "utf-8"
__authors__ = ["Brian R. Pauw"]
__copyright__ = "Copyright 2026, The MoDaCor team"
__date__ = "02/09/2026"
__status__ = "Development"

__all__ = [
    "NexusDetectorFrameInputs",
    "NexusTransformResult",
    "load_nexus_detector_frame_inputs",
    "resolve_nexus_transform_chain",
    "resolve_nexus_transform_path",
]
__version__ = "20260902.1"

import posixpath
from dataclasses import dataclass
from typing import Any

import numpy as np

from modacor import ureg
from modacor.dataclasses.basedata import BaseData
from modacor.io.io_sources import IoSources


@dataclass(frozen=True, slots=True)
class NexusTransformResult:
    """Resolved affine transform from a NeXus transformation chain."""

    matrix: np.ndarray
    paths: tuple[str, ...]

    @property
    def rotation(self) -> np.ndarray:
        return np.asarray(self.matrix[:3, :3], dtype=float)

    @property
    def translation(self) -> np.ndarray:
        return np.asarray(self.matrix[:3, 3], dtype=float)


@dataclass(frozen=True, slots=True)
class NexusDetectorFrameInputs:
    """Detector-frame quantities in the form used by detector coordinate calculators."""

    det_coord_x: BaseData
    det_coord_y: BaseData
    det_coord_z: BaseData
    pixel_pitch_fast: BaseData
    pixel_pitch_slow: BaseData
    basis_fast: np.ndarray
    basis_slow: np.ndarray
    basis_normal: np.ndarray
    transform_paths: tuple[str, ...]


def _decode(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray) and value.shape == ():
        return _decode(value.item())
    if isinstance(value, np.ndarray) and value.dtype.kind in {"S", "O", "U"}:
        return np.array([_decode(item) for item in value.ravel()]).reshape(value.shape)
    return value


def _as_scalar(value: Any, *, name: str) -> float:
    array = np.asarray(_decode(value), dtype=float)
    if array.size != 1:
        raise ValueError(f"{name} must be scalar; got shape {array.shape}.")
    return float(array.reshape(-1)[0])


def _quantity(value: Any, units: Any, target_units: str) -> float:
    return float(
        ureg.Quantity(_as_scalar(value, name="transform value"), str(_decode(units))).to(target_units).magnitude
    )


def _vector3(value: Any, *, name: str) -> np.ndarray:
    vector = np.asarray(_decode(value), dtype=float).reshape(3)
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        raise ValueError(f"{name} must be non-zero.")
    return vector / norm


def _optional_vector3(value: Any, *, name: str) -> np.ndarray | None:
    vector = np.asarray(_decode(value), dtype=float).reshape(3)
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        return None
    return vector / norm


def _offset3(attrs: dict[str, Any]) -> np.ndarray:
    if "offset" not in attrs:
        return np.zeros(3, dtype=float)
    offset = np.asarray(_decode(attrs["offset"]), dtype=float).reshape(3)
    units = attrs.get("offset_units", attrs.get("units", "m"))
    return np.asarray(ureg.Quantity(offset, str(_decode(units))).to("m").magnitude, dtype=float)


def _identity() -> np.ndarray:
    return np.eye(4, dtype=float)


def _translation(vector: np.ndarray) -> np.ndarray:
    matrix = _identity()
    matrix[:3, 3] = vector
    return matrix


def _rotation(axis: np.ndarray, angle_radians: float) -> np.ndarray:
    x, y, z = axis
    c = float(np.cos(angle_radians))
    s = float(np.sin(angle_radians))
    c1 = 1.0 - c
    matrix = _identity()
    matrix[:3, :3] = np.array(
        [
            [c + x * x * c1, x * y * c1 - z * s, x * z * c1 + y * s],
            [y * x * c1 + z * s, c + y * y * c1, y * z * c1 - x * s],
            [z * x * c1 - y * s, z * y * c1 + x * s, c + z * z * c1],
        ],
        dtype=float,
    )
    return matrix


def _normalise_path(path: str) -> str:
    normalised = posixpath.normpath("/" + str(path).strip().lstrip("/"))
    return "/" if normalised == "/." else normalised


def _join_relative_path(base_path: str, relative_path: str) -> str:
    if relative_path.startswith("/"):
        return _normalise_path(relative_path)
    return _normalise_path(posixpath.join(posixpath.dirname(base_path), relative_path))


def _attrs(io_sources: IoSources, source_reference: str, path: str) -> dict[str, Any]:
    return io_sources.get_data_attributes(f"{source_reference}::{path}")


def _data(io_sources: IoSources, source_reference: str, path: str) -> Any:
    return io_sources.get_static_metadata(f"{source_reference}::{path}")


def resolve_nexus_transform_path(
    io_sources: IoSources,
    source_reference: str,
    current_path: str,
    depends_on: Any,
) -> str | None:
    """Resolve a NeXus ``depends_on`` value relative to the current dataset path."""

    depends_on = _decode(depends_on)
    if isinstance(depends_on, np.ndarray):
        if depends_on.size != 1:
            raise ValueError(f"depends_on at {current_path!r} must be scalar; got {depends_on!r}.")
        depends_on = depends_on.reshape(-1)[0]
    depends_on = str(depends_on).strip()
    if depends_on in {"", "."}:
        return None
    return _join_relative_path(current_path, depends_on)


def _transform_matrix(value: Any, attrs: dict[str, Any], *, path: str) -> np.ndarray:
    transform_type = str(_decode(attrs.get("transformation_type", ""))).strip().lower()

    if transform_type == "translation":
        distance = _quantity(value, attrs.get("units", "m"), "m")
        vector = _optional_vector3(attrs.get("vector", [0.0, 0.0, 1.0]), name=f"{path}@vector")
        if vector is None:
            if distance == 0.0:
                return _identity()
            raise ValueError(f"{path}@vector must be non-zero.")
        return _translation(distance * vector)

    if transform_type == "rotation":
        vector = _vector3(attrs.get("vector", [0.0, 0.0, 1.0]), name=f"{path}@vector")
        offset = _offset3(attrs)
        angle = _quantity(value, attrs.get("units", "radian"), "radian")
        return _translation(offset) @ _rotation(vector, angle) @ _translation(-offset)

    raise ValueError(
        f"Unsupported NeXus transformation_type {transform_type!r} at {path!r}; "
        "expected 'translation' or 'rotation'."
    )


def resolve_nexus_transform_chain(
    io_sources: IoSources,
    source_reference: str,
    transform_path: str,
) -> NexusTransformResult:
    """
    Resolve a NeXus transformation chain into a 4x4 affine transform matrix.

    The returned matrix is expressed in SI length units. The ``paths`` tuple is
    ordered from the root transform towards ``transform_path``.
    """

    terminal_path = _normalise_path(transform_path)
    reverse_paths: list[str] = []
    current_path: str | None = terminal_path
    seen: set[str] = set()

    while current_path is not None:
        if current_path in seen:
            raise ValueError(f"Cycle detected in NeXus depends_on chain at {current_path!r}.")
        seen.add(current_path)
        reverse_paths.append(current_path)

        attrs = _attrs(io_sources, source_reference, current_path)
        current_path = resolve_nexus_transform_path(
            io_sources,
            source_reference,
            current_path,
            attrs.get("depends_on", "."),
        )

    paths = tuple(reversed(reverse_paths))
    matrix = _identity()
    for path in paths:
        matrix = matrix @ _transform_matrix(
            _data(io_sources, source_reference, path),
            _attrs(io_sources, source_reference, path),
            path=path,
        )

    return NexusTransformResult(matrix=matrix, paths=paths)


def _relative_depends_on(io_sources: IoSources, source_reference: str, path: str) -> str | None:
    attrs = _attrs(io_sources, source_reference, path)
    depends_on = attrs.get("depends_on")
    if depends_on is None:
        return None
    return resolve_nexus_transform_path(io_sources, source_reference, path, depends_on)


def _basedata_scalar(value: float, units: str) -> BaseData:
    return BaseData(signal=np.asarray(value, dtype=float), units=ureg.Unit(units), rank_of_data=0)


def load_nexus_detector_frame_inputs(
    io_sources: IoSources,
    *,
    source_reference: str,
    detector_path: str,
    detector_module_name: str = "detector_module",
    module_origin: str = "corner",
) -> NexusDetectorFrameInputs:
    """
    Read a planar NeXus ``NXdetector``/``NXdetector_module`` frame.

    ``module_origin`` controls how the resolved module offset is adapted for
    coordinate calculators that add a half-pixel centre shift themselves:

    - ``"corner"``: use the module offset directly.
    - ``"first_pixel_center"``: subtract half a fast and half a slow pixel.
    """

    detector_path = _normalise_path(detector_path)
    module_path = _normalise_path(posixpath.join(detector_path, detector_module_name))
    module_offset_path = _normalise_path(posixpath.join(module_path, "module_offset"))
    fast_path = _normalise_path(posixpath.join(module_path, "fast_pixel_direction"))
    slow_path = _normalise_path(posixpath.join(module_path, "slow_pixel_direction"))

    fast_attrs = _attrs(io_sources, source_reference, fast_path)
    slow_attrs = _attrs(io_sources, source_reference, slow_path)

    fast_transform_path = _relative_depends_on(io_sources, source_reference, fast_path) or module_offset_path
    slow_transform_path = _relative_depends_on(io_sources, source_reference, slow_path) or module_offset_path
    origin_transform = resolve_nexus_transform_chain(io_sources, source_reference, module_offset_path)
    fast_transform = resolve_nexus_transform_chain(io_sources, source_reference, fast_transform_path)
    slow_transform = resolve_nexus_transform_chain(io_sources, source_reference, slow_transform_path)

    basis_fast = _vector3(
        fast_transform.rotation @ _vector3(fast_attrs["vector"], name=f"{fast_path}@vector"), name="basis_fast"
    )
    basis_slow = _vector3(
        slow_transform.rotation @ _vector3(slow_attrs["vector"], name=f"{slow_path}@vector"), name="basis_slow"
    )
    basis_normal = _vector3(np.cross(basis_fast, basis_slow), name="basis_normal")

    pixel_fast_m = _quantity(_data(io_sources, source_reference, fast_path), fast_attrs.get("units", "m"), "m")
    pixel_slow_m = _quantity(_data(io_sources, source_reference, slow_path), slow_attrs.get("units", "m"), "m")
    origin_m = np.asarray(origin_transform.translation, dtype=float)

    if module_origin == "first_pixel_center":
        origin_m = origin_m - 0.5 * pixel_fast_m * basis_fast - 0.5 * pixel_slow_m * basis_slow
    elif module_origin != "corner":
        raise ValueError("module_origin must be 'corner' or 'first_pixel_center'.")

    transform_paths = tuple(dict.fromkeys((*origin_transform.paths, *fast_transform.paths, *slow_transform.paths)))

    return NexusDetectorFrameInputs(
        det_coord_x=_basedata_scalar(float(origin_m[0]), "m"),
        det_coord_y=_basedata_scalar(float(origin_m[1]), "m"),
        det_coord_z=_basedata_scalar(float(origin_m[2]), "m"),
        pixel_pitch_fast=_basedata_scalar(pixel_fast_m, "m"),
        pixel_pitch_slow=_basedata_scalar(pixel_slow_m, "m"),
        basis_fast=basis_fast,
        basis_slow=basis_slow,
        basis_normal=basis_normal,
        transform_paths=transform_paths,
    )
