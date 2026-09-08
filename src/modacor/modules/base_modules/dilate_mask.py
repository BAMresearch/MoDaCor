# SPDX-License-Identifier: BSD-3-Clause
# /usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

__coding__ = "utf-8"
__authors__ = ["Brian R. Pauw"]
__copyright__ = "Copyright 2026, The MoDaCor team"
__date__ = "02/09/2026"
__status__ = "Development"

__all__ = ["DilateMask"]
__version__ = "20260902.1"

from pathlib import Path
from typing import Any

import numpy as np
from scipy import ndimage

try:
    from skimage.morphology import binary_dilation as skimage_binary_dilation
except ImportError:  # pragma: no cover - exercised only when optional extra is installed
    skimage_binary_dilation = None

from modacor.dataclasses.basedata import BaseData
from modacor.dataclasses.databundle import DataBundle
from modacor.dataclasses.process_step import ProcessStep, ProcessStepDependencies, processing_key_patterns
from modacor.dataclasses.process_step_describer import ProcessStepDescriber


class DilateMask(ProcessStep):
    """Dilate a 2D integer mask while preserving NeXus bitfield reason values."""

    documentation = ProcessStepDescriber(
        calling_name="Dilate mask",
        calling_id="DilateMask",
        calling_module_path=Path(__file__),
        calling_version=__version__,
        required_data_keys=[],
        modifies={},
        arguments={
            "with_processing_keys": {
                "type": list,
                "required": True,
                "default": ["sample"],
                "doc": "Single processing key identifying the DataBundle to update.",
            },
            "source_mask_key": {
                "type": str,
                "default": "mask",
                "doc": "BaseData key for the mask to dilate.",
            },
            "target_mask_key": {
                "type": str,
                "default": "mask",
                "doc": "BaseData key for the dilated mask output.",
            },
            "radius": {
                "type": int,
                "default": 1,
                "doc": "Dilation radius in pixels. Aliases: number_of_pixels, number_of_pixels_to_dilate.",
            },
            "footprint_shape": {
                "type": str,
                "default": "square",
                "doc": "2D dilation footprint: 'square', 'disk', or 'cross'.",
            },
            "axes": {
                "type": (list, tuple, type(None)),
                "default": None,
                "doc": "Two axes to dilate over. Defaults to the last two axes.",
            },
            "backend": {
                "type": str,
                "default": "auto",
                "doc": "Use 'auto', 'skimage', or 'scipy'. 'auto' prefers scikit-image when installed.",
            },
        },
        step_keywords=["mask", "dilate", "morphology", "bitfield"],
        step_doc="Dilate a 2D integer mask over selected axes while preserving uint32 reason bits.",
        step_note="""
            Configuration:
              with_processing_keys: [static]     # required, single databundle key
              source_mask_key: mask              # optional, default: mask
              target_mask_key: mask              # optional, default: mask
              radius: 1                          # optional
              footprint_shape: square            # square, disk, or cross

            If the mask has leading dimensions, dilation is applied plane-by-plane
            over the configured 2D axes. Each NeXus bit is dilated independently.
        """,
    )

    @staticmethod
    def _normalize_axes(axes: Any, ndim: int) -> tuple[int, int]:
        if ndim < 2:
            raise ValueError("DilateMask requires a mask with at least two dimensions.")
        if axes is None:
            return ndim - 2, ndim - 1
        try:
            axes_tuple = tuple(int(axis) for axis in axes)
        except TypeError as exc:
            raise ValueError("DilateMask axes must be an iterable containing exactly two integers.") from exc
        if len(axes_tuple) != 2:
            raise ValueError("DilateMask axes must contain exactly two axes.")
        normalized = tuple(axis if axis >= 0 else ndim + axis for axis in axes_tuple)
        if any(axis < 0 or axis >= ndim for axis in normalized):
            raise ValueError(f"DilateMask axes {axes_tuple!r} are invalid for mask ndim {ndim}.")
        if normalized[0] == normalized[1]:
            raise ValueError(f"DilateMask axes must not contain duplicates: {axes_tuple!r}.")
        return normalized  # type: ignore[return-value]

    @staticmethod
    def _radius_from_config(cfg: dict[str, Any]) -> int:
        value = cfg.get("radius", cfg.get("number_of_pixels", cfg.get("number_of_pixels_to_dilate", 1)))
        radius = int(value)
        if radius < 0:
            raise ValueError("DilateMask radius must be >= 0.")
        return radius

    @staticmethod
    def _footprint(radius: int, shape: str) -> np.ndarray:
        if radius == 0:
            return np.ones((1, 1), dtype=bool)

        size = 2 * radius + 1
        shape_normalized = str(shape).strip().lower()
        if shape_normalized == "square":
            return np.ones((size, size), dtype=bool)

        y, x = np.ogrid[-radius : radius + 1, -radius : radius + 1]
        if shape_normalized == "disk":
            return (x * x + y * y) <= radius * radius
        if shape_normalized == "cross":
            return (np.abs(x) + np.abs(y)) <= radius
        raise ValueError("DilateMask footprint_shape must be 'square', 'disk', or 'cross'.")

    @staticmethod
    def _dilate_binary(mask: np.ndarray, footprint: np.ndarray, backend: str) -> np.ndarray:
        backend_normalized = str(backend).strip().lower()
        if backend_normalized not in {"auto", "skimage", "scipy"}:
            raise ValueError("DilateMask backend must be 'auto', 'skimage', or 'scipy'.")
        if backend_normalized in {"auto", "skimage"} and skimage_binary_dilation is not None:
            return np.asarray(skimage_binary_dilation(mask, footprint=footprint), dtype=bool)
        if backend_normalized == "skimage":
            raise ImportError("DilateMask backend='skimage' requires installing MoDaCor[masks] or scikit-image.")
        return np.asarray(ndimage.binary_dilation(mask, structure=footprint), dtype=bool)

    @classmethod
    def _dilate_plane(cls, plane: np.ndarray, footprint: np.ndarray, backend: str) -> np.ndarray:
        result = np.zeros(plane.shape, dtype=np.uint32)
        plane_u32 = plane.astype(np.uint32, copy=False)
        for bit_index in range(32):
            bit = np.uint32(1 << bit_index)
            bit_mask = (plane_u32 & bit) != 0
            if not np.any(bit_mask):
                continue
            result[cls._dilate_binary(bit_mask, footprint, backend)] |= bit
        return result

    @classmethod
    def _dilate(cls, mask: np.ndarray, axes: tuple[int, int], footprint: np.ndarray, backend: str) -> np.ndarray:
        moved = np.moveaxis(mask, axes, (-2, -1))
        result = np.empty(moved.shape, dtype=np.uint32)
        planes_in = moved.reshape((-1, *moved.shape[-2:]))
        planes_out = result.reshape((-1, *moved.shape[-2:]))
        for index, plane in enumerate(planes_in):
            planes_out[index] = cls._dilate_plane(plane, footprint, backend)
        return np.moveaxis(result, (-2, -1), axes)

    def dependency_contract(self) -> ProcessStepDependencies:
        cfg = self.configuration or {}
        keys = cfg.get("with_processing_keys")
        source_key = cfg.get("source_mask_key", "mask")
        target_key = cfg.get("target_mask_key", "mask")
        return ProcessStepDependencies(
            source_refs=(),
            processing_reads=processing_key_patterns(keys, basedata_key=source_key),
            processing_writes=processing_key_patterns(keys, basedata_key=target_key),
        )

    def calculate(self) -> dict[str, DataBundle]:
        keys = self._normalised_processing_keys()
        assert len(keys) == 1, "DilateMask requires a single databundle processing key."

        processing_key = keys[0]
        source_key = self.configuration.get("source_mask_key", "mask")
        target_key = self.configuration.get("target_mask_key", "mask")
        radius = self._radius_from_config(self.configuration)
        footprint = self._footprint(radius, self.configuration.get("footprint_shape", "square"))
        backend = self.configuration.get("backend", "auto")

        bundle = self.processing_data[processing_key]
        source_mask: BaseData = bundle[source_key]
        source = np.asarray(source_mask.signal)
        if not np.issubdtype(source.dtype, np.integer):
            raise TypeError(f"{processing_key}::{source_key} must be an integer mask, got {source.dtype}.")

        axes = self._normalize_axes(self.configuration.get("axes"), source.ndim)
        dilated = self._dilate(source, axes, footprint, backend)
        bundle[target_key] = BaseData(
            signal=dilated,
            units=source_mask.units,
            uncertainties={},
            weights=np.array(source_mask.weights, copy=True),
            axes=list(source_mask.axes),
            rank_of_data=source_mask.rank_of_data,
        )
        return {processing_key: bundle}
