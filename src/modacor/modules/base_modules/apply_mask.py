# SPDX-License-Identifier: BSD-3-Clause
# /usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

__coding__ = "utf-8"
__authors__ = [
    "Brian R. Pauw",
    "Anja F. Hörmann",
]  # add names to the list as appropriate
__copyright__ = "Copyright 2026, The MoDaCor team"
__date__ = "16/07/2026"
__status__ = "Development"  # "Development", "Production"
# end of header and standard imports

__all__ = ["ApplyMask"]
__version__ = "20260716.1"

from pathlib import Path

import numpy as np

from modacor.dataclasses.basedata import BaseData
from modacor.dataclasses.databundle import DataBundle
from modacor.dataclasses.process_step import ProcessStep
from modacor.dataclasses.process_step_describer import ProcessStepDescriber


class ApplyMask(ProcessStep):
    """
    Mask data using a mask BaseData entry.

    MoDaCor's Masks are 32-bit integer bitfields (NeXus convention). This step updates the
    selected BaseData arrays in-place by replacing invalid pixels according to
    the mask. By default, masked pixels are set to NaN so downstream numerical
    reductions do not silently include invalid pixels as real zeroes.
    """

    documentation = ProcessStepDescriber(
        calling_name="Apply mask to the signal in a DataBundle",
        calling_id="ApplyMask",
        calling_module_path=Path(__file__),
        calling_version=__version__,
        required_data_keys=["mask", "signal"],
        modifies={"signal": ["signal"]},
        arguments={
            "with_processing_keys": {
                "type": list,
                "required": True,
                "default": ["sample"],
                "doc": "Single processing key identifying the DataBundle to update.",
            },
            "mask_key": {
                "type": str,
                "default": "mask",
                "doc": "BaseData key for the mask to be used inside the DataBundle.",
                "dependency_role": "processing_read_basedata_key",
            },
            "basedata_to_mask": {
                "type": list,
                "required": True,
                "default": ["signal"],
                "doc": "List of BaseData keys to apply the mask to.",
                "dependency_role": "processing_read_write_basedata_key_list",
            },
            "masked_value": {
                "type": (str, int, float, type(None)),
                "default": "nan",
                "doc": "Replacement value for masked pixels. Use 'nan' for NaN, or a numeric value such as 0 or -1.",
            },
        },
        step_keywords=["mask", "signal", "apply", "databundle"],
        step_doc="Apply a uint32 bitfield mask to one or more BaseData signal arrays in the same DataBundle.",
        step_reference="NeXus mask bit-field convention (NXdata/NXdetector masks)",
        step_note="""
            Configuration:
              with_processing_keys: [sample]     # required, single databundle key
              mask_key: mask                     # optional, default: mask
              basedata_to_mask: [signal, ...]    # optional, default: [signal]
              masked_value: nan                   # optional; use 0 or -1 for explicit sentinels

            Performs:
              basedata[mask != 0] = masked_value  (in-place, for each source)
        """,
    )

    @staticmethod
    def _require_int(arr: np.ndarray, name: str) -> None:
        assert np.issubdtype(arr.dtype, np.integer), f"{name} must be an integer mask, got {arr.dtype}."

    @staticmethod
    def _masked_value(value):
        if value is None:
            return np.nan
        if isinstance(value, str):
            stripped = value.strip().lower()
            if stripped in {"nan", "na"}:
                return np.nan
            return float(stripped)
        return value

    @staticmethod
    def _signal_for_masked_value(signal: np.ndarray, value) -> np.ndarray:
        value_array = np.asarray(value)
        if value_array.shape != ():
            raise ValueError("ApplyMask masked_value must be scalar.")

        needs_float_nan = np.issubdtype(value_array.dtype, np.floating) and np.isnan(value_array)
        if needs_float_nan and not np.issubdtype(signal.dtype, np.floating):
            return signal.astype(float)

        if np.issubdtype(signal.dtype, np.integer):
            if np.issubdtype(value_array.dtype, np.integer):
                limits = np.iinfo(signal.dtype)
                if limits.min <= int(value) <= limits.max:
                    return signal
            if np.issubdtype(value_array.dtype, np.floating):
                value_float = float(value)
                limits = np.iinfo(signal.dtype)
                if value_float.is_integer() and limits.min <= value_float <= limits.max:
                    return signal

        if not np.can_cast(value_array.dtype, signal.dtype, casting="safe"):
            return signal.astype(np.result_type(signal.dtype, value_array.dtype))

        return signal

    def calculate(self) -> dict[str, DataBundle]:
        cfg = self.configuration

        keys = self._normalised_processing_keys()
        assert len(keys) == 1, "ApplyMask requires a single databundle processing key."
        processing_key = keys[0]
        mask_key = cfg.get("mask_key", "mask")
        source_keys = cfg.get("basedata_to_mask", ["signal"])
        masked_value = self._masked_value(cfg.get("masked_value", "nan"))

        assert isinstance(source_keys, list) and source_keys, "basedata_to_mask must be a non-empty list."

        bundle = self.processing_data[processing_key]
        mask_bd: BaseData = bundle[mask_key]
        mask = mask_bd.signal

        self._require_int(mask, f"{processing_key}::{mask_key}")  # noqa: E231

        # Canonicalize target to uint32 once (needed for NeXus-style 32-bit bitfields)
        if mask.dtype != np.uint32:
            mask = mask.astype(np.uint32, copy=True)  # one-time allocation
            mask_bd.signal = mask

        for sk in source_keys:
            src_bd: BaseData = bundle[sk]
            src = self._signal_for_masked_value(src_bd.signal, masked_value)
            if src is not src_bd.signal:
                src_bd.signal = src

            np.copyto(src, masked_value, where=mask != 0)

        return {processing_key: bundle}
