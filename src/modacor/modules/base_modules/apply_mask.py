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
    Mask the data using the 'mask' BaseData entry. Data is set to zero where the mask is 1.

    MoDaCor's Masks are 32-bit integer bitfields (NeXus convention). This step updates the
    signal in-place by masking invalid pixels according to the mask.
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
            },
            "basedata_to_mask": {
                "type": list,
                "required": True,
                "default": ["signal"],
                "doc": "List of BaseData keys to apply the mask to.",
            },
        },
        step_keywords=["mask", "signal", "apply", "databundle"],
        step_doc="Combine multiple mask arrays stored as different BaseData keys in the same DataBundle.",
        step_reference="NeXus mask bit-field convention (NXdata/NXdetector masks)",
        step_note="""
            Configuration:
              with_processing_keys: [sample]     # required, single databundle key
              mask_key: mask                     # optional, default: mask
              basedata_to_mask: [signal, ...]    # optional, default: [signal]

            Performs:
              basedata[mask] = 0  (in-place, for each source)
        """,
    )

    @staticmethod
    def _require_int(arr: np.ndarray, name: str) -> None:
        assert np.issubdtype(arr.dtype, np.integer), (
            f"{name} must be an integer mask, got {arr.dtype}."
        )

    def calculate(self) -> dict[str, DataBundle]:
        cfg = self.configuration

        keys = self._normalised_processing_keys()
        assert len(keys) == 1, (
            "BitwiseOrMasks requires a single databundle processing key."
        )
        processing_key = keys[0]
        mask_key = cfg.get("mask_key", "mask")
        source_keys = cfg.get("basedata_to_mask", ["signal"])

        assert isinstance(source_keys, list) and source_keys, (
            "basedata_to_mask must be a non-empty list."
        )

        bundle = self.processing_data[processing_key]
        mask_bd: BaseData = bundle[mask_key]
        mask = mask_bd.signal

        self._require_int(mask, f"{processing_key}::{mask_key}")  # noqa: E231

        # Canonicalize target to uint32 once (needed for NeXus-style 32-bit bitfields)
        if mask.dtype != np.uint32:
            mask = mask.astype(np.uint32, copy=True)  # one-time allocation
            target_bd.signal = tgt

        for sk in source_keys:
            src_bd: BaseData = bundle[sk]
            src = src_bd.signal

            src[mask == 1] = 0

        return {processing_key: bundle}
