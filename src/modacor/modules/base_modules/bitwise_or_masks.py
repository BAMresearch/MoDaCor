# SPDX-License-Identifier: BSD-3-Clause
# /usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

__coding__ = "utf-8"
__authors__ = ["Brian R. Pauw"]  # add names to the list as appropriate
__copyright__ = "Copyright 2026, The MoDaCor team"
__date__ = "09/01/2026"
__status__ = "Development"  # "Development", "Production"
# end of header and standard imports

__all__ = ["BitwiseOrMasks"]
__version__ = "20260109.3"

from pathlib import Path

import numpy as np

from modacor.dataclasses.basedata import BaseData
from modacor.dataclasses.databundle import DataBundle
from modacor.dataclasses.process_step import ProcessStep
from modacor.dataclasses.process_step_describer import ProcessStepDescriber


class BitwiseOrMasks(ProcessStep):
    """
    Bitwise-OR one or more mask BaseData entries into a target mask BaseData entry
    within the same DataBundle.

    MoDaCor's Masks are 32-bit integer bitfields (NeXus convention). This step updates the
    target mask in-place and preserves reason bits.
    """

    documentation = ProcessStepDescriber(
        calling_name="Combine masks within one DataBundle (bitwise OR)",
        calling_id="BitwiseOrMasksInBundle",
        calling_module_path=Path(__file__),
        calling_version=__version__,
        required_data_keys=["mask"],
        modifies={"mask": ["signal"]},
        arguments={
            "with_processing_keys": {
                "type": list,
                "required": True,
                "default": ["sample"],
                "doc": "Single processing key identifying the DataBundle to update.",
            },
            "target_mask_key": {
                "type": str,
                "default": "mask",
                "doc": "BaseData key for the target mask inside the DataBundle.",
            },
            "source_mask_keys": {
                "type": list,
                "required": True,
                "default": [],
                "doc": "List of BaseData keys to OR into the target mask.",
            },
        },
        step_keywords=["mask", "bitmask", "bitwise", "or", "databundle"],
        step_doc="Combine multiple mask arrays stored as different BaseData keys in the same DataBundle.",
        step_reference="NeXus mask bit-field convention (NXdata/NXdetector masks)",
        step_note="""
            Configuration:
              with_processing_keys: [sample]     # required, single databundle key
              target_mask_key: mask              # optional, default: mask
              source_mask_keys: [bs_mask, ...]   # required, one or more

            Performs:
              target_mask |= source_mask  (in-place, for each source)
        """,
    )

    @staticmethod
    def _require_int(arr: np.ndarray, name: str) -> None:
        assert np.issubdtype(arr.dtype, np.integer), f"{name} must be an integer mask, got {arr.dtype}."

    def calculate(self) -> dict[str, DataBundle]:
        cfg = self.configuration

        keys = self._normalised_processing_keys()
        assert len(keys) == 1, "BitwiseOrMasks requires a single databundle processing key."
        processing_key = keys[0]
        target_key = cfg.get("target_mask_key", "mask")
        source_keys = cfg["source_mask_keys"]

        assert isinstance(source_keys, list) and source_keys, "source_mask_keys must be a non-empty list."

        bundle = self.processing_data[processing_key]
        target_bd: BaseData = bundle[target_key]
        tgt = target_bd.signal

        self._require_int(tgt, f"{processing_key}::{target_key}")  # noqa: E231

        # Canonicalize target to uint32 once (needed for NeXus-style 32-bit bitfields)
        if tgt.dtype != np.uint32:
            tgt = tgt.astype(np.uint32, copy=True)  # one-time allocation
            target_bd.signal = tgt

        for sk in source_keys:
            src_bd: BaseData = bundle[sk]
            src = src_bd.signal
            self._require_int(src, f"{processing_key}::{sk}")  # noqa: E231

            # Convert only if needed (uint8/int16/etc -> uint32)
            src_u32 = src if src.dtype == np.uint32 else src.astype(np.uint32, copy=False)

            # In-place OR; NumPy handles broadcasting or raises
            np.bitwise_or(tgt, src_u32, out=tgt)

        return {processing_key: bundle}
