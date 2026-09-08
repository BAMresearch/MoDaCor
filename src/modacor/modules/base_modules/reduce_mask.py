# SPDX-License-Identifier: BSD-3-Clause
# /usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

__coding__ = "utf-8"
__authors__ = ["Brian R. Pauw"]
__copyright__ = "Copyright 2026, The MoDaCor team"
__date__ = "02/09/2026"
__status__ = "Development"

__all__ = ["ReduceMask"]
__version__ = "20260902.1"

from pathlib import Path
from typing import Any

import numpy as np

from modacor import ureg
from modacor.dataclasses.basedata import BaseData
from modacor.dataclasses.databundle import DataBundle
from modacor.dataclasses.process_step import ProcessStep, ProcessStepDependencies, processing_key_patterns
from modacor.dataclasses.process_step_describer import ProcessStepDescriber


class ReduceMask(ProcessStep):
    """Reduce an integer mask over one or more axes while preserving NeXus bitfields."""

    documentation = ProcessStepDescriber(
        calling_name="Reduce mask over axes",
        calling_id="ReduceMask",
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
                "doc": "BaseData key for the mask to reduce.",
            },
            "target_mask_key": {
                "type": str,
                "default": "mask",
                "doc": "BaseData key for the reduced mask output.",
            },
            "axes": {
                "type": (int, list, tuple, type(None)),
                "default": None,
                "doc": "Axis or axes to reduce. Use None to reduce all axes.",
            },
            "reduction": {
                "type": str,
                "default": "any",
                "doc": "Use 'any' to OR mask bits across axes, or 'all' to AND bits across axes.",
            },
        },
        step_keywords=["mask", "reduce", "bitfield"],
        step_doc="Reduce an integer mask over configured axes while preserving uint32 reason bits.",
        step_note=(
            "For 'any', a bit remains set if it is set in any reduced element. "
            "For 'all', a bit remains set only if it is set in all reduced elements."
        ),
    )

    @staticmethod
    def _normalize_axes(axes: Any, ndim: int) -> tuple[int, ...]:
        if axes is None:
            return tuple(range(ndim))
        if isinstance(axes, int):
            axes_tuple = (axes,)
        else:
            try:
                axes_tuple = tuple(int(axis) for axis in axes)
            except TypeError as exc:
                raise ValueError("ReduceMask axes must be an int, an iterable of ints, or None.") from exc

        normalized = tuple(axis if axis >= 0 else ndim + axis for axis in axes_tuple)
        if any(axis < 0 or axis >= ndim for axis in normalized):
            raise ValueError(f"ReduceMask axes {axes_tuple!r} are invalid for mask ndim {ndim}.")
        if len(set(normalized)) != len(normalized):
            raise ValueError(f"ReduceMask axes must not contain duplicates: {axes_tuple!r}.")
        return normalized

    @staticmethod
    def _rank_after_reduction(mask: BaseData, reduced_axes: tuple[int, ...], new_ndim: int) -> int:
        rank = int(getattr(mask, "rank_of_data", 0))
        if rank <= 0:
            return 0
        detector_axes = set(range(mask.signal.ndim - rank, mask.signal.ndim))
        remaining_detector_axes = detector_axes - set(reduced_axes)
        return min(len(remaining_detector_axes), new_ndim)

    @staticmethod
    def _reduce(mask: np.ndarray, axes: tuple[int, ...], reduction: str) -> np.ndarray:
        reduced = mask.astype(np.uint32, copy=False)
        reducer = {"any": np.bitwise_or.reduce, "all": np.bitwise_and.reduce}.get(reduction)
        if reducer is None:
            raise ValueError("ReduceMask reduction must be 'any' or 'all'.")
        for axis in sorted(axes, reverse=True):
            reduced = reducer(reduced, axis=axis)
        return np.asarray(reduced, dtype=np.uint32)

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
        assert len(keys) == 1, "ReduceMask requires a single databundle processing key."

        processing_key = keys[0]
        source_key = self.configuration.get("source_mask_key", "mask")
        target_key = self.configuration.get("target_mask_key", "mask")
        reduction = str(self.configuration.get("reduction", "any")).strip().lower()

        bundle = self.processing_data[processing_key]
        source_mask: BaseData = bundle[source_key]
        source = np.asarray(source_mask.signal)
        if not np.issubdtype(source.dtype, np.integer):
            raise TypeError(f"{processing_key}::{source_key} must be an integer mask, got {source.dtype}.")

        axes = self._normalize_axes(self.configuration.get("axes"), source.ndim)
        reduced = self._reduce(source, axes, reduction)
        bundle[target_key] = BaseData(
            signal=reduced,
            units=ureg.dimensionless,
            uncertainties={},
            axes=[],
            rank_of_data=self._rank_after_reduction(source_mask, axes, reduced.ndim),
        )
        return {processing_key: bundle}
