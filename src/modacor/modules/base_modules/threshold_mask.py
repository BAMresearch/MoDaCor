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
__date__ = "17/07/2026"
__status__ = "Development"  # "Development", "Production"
# end of header and standard imports

__all__ = ["ThresholdMask"]
__version__ = "20260717.1"

from pathlib import Path
from typing import Any

import numpy as np

from modacor.dataclasses.basedata import BaseData
from modacor.dataclasses.databundle import DataBundle
from modacor.dataclasses.process_step import ProcessStep
from modacor.dataclasses.process_step_describer import ProcessStepDescriber
from modacor.modules.helpers import attach_prepared_data


class ThresholdMask(ProcessStep):
    """
    Create a mask based on lower and/or upper bounds applied to a BaseData entry.

    By default, the bounds are applied to ``signal`` and pixels outside the
    configured range are masked. This can be used to mask detector panel gaps,
    hot pixels, or invalid correction maps such as flatfield matrices.

    MoDaCor's Masks are 32-bit integer bitfields (NeXus convention). This step creates a
    new mask at the specified target_mask_key (default: threshold_mask).
    """

    documentation = ProcessStepDescriber(
        calling_name="Create a threshold mask from a BaseData entry",
        calling_id="ThresholdMask",
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
            "target_mask_key": {
                "type": str,
                "default": "threshold_mask",
                "doc": "BaseData key for the mask to create inside the DataBundle.",
                "dependency_role": "processing_write_basedata_key",
            },
            "source_basedata_key": {
                "type": str,
                "default": "signal",
                "doc": "BaseData key whose signal array is evaluated to create the mask.",
                "dependency_role": "processing_read_basedata_key",
            },
            "lower_bound": {
                "type": (float, int, type(None)),
                "default": None,
                "doc": "Optional inclusive lower bound.",
            },
            "upper_bound": {
                "type": (float, int, type(None)),
                "default": None,
                "doc": "Optional inclusive upper bound.",
            },
            "mask_mode": {
                "type": str,
                "default": "outside",
                "doc": "Use 'outside' to mask values below/above the bounds, or 'inside' to mask values within them.",
            },
            "threshold": {
                "type": (float, int, type(None)),
                "default": None,
                "doc": "Deprecated compatibility alias for upper_bound when upper_bound is not set.",
            },
        },
        step_keywords=["mask", "threshold", "databundle"],
        step_doc="Create a uint32 mask by evaluating bounds on a selected BaseData signal array.",
        step_reference="",
        step_note="""
            Configuration:
              with_processing_keys: [sample]        # required, single databundle key
              source_basedata_key: flatfield        # optional, default: signal
              target_mask_key: flatfield_mask       # optional, default: threshold_mask
              lower_bound: 0.8                      # optional
              upper_bound: 1.2                      # optional
              mask_mode: outside                    # outside or inside

            Performs without changing the source array dimensionality:
              outside: mask = (source < lower_bound) | (source > upper_bound)
              inside:  mask = (source >= lower_bound) & (source <= upper_bound)

            The legacy threshold option is treated as upper_bound when
            upper_bound is not configured.
        """,
    )

    @staticmethod
    def _resolve_bounds(cfg: dict[str, Any]) -> tuple[float | None, float | None]:
        lower_bound = cfg.get("lower_bound")
        upper_bound = cfg.get("upper_bound")
        threshold = cfg.get("threshold")

        if upper_bound is None and threshold is not None:
            upper_bound = threshold

        lower = None if lower_bound is None else float(lower_bound)
        upper = None if upper_bound is None else float(upper_bound)

        if lower is None and upper is None:
            raise ValueError("ThresholdMask requires at least one of lower_bound, upper_bound, or threshold.")
        if lower is not None and upper is not None and lower > upper:
            raise ValueError("ThresholdMask requires lower_bound <= upper_bound.")

        return lower, upper

    @staticmethod
    def _threshold_match(data: np.ndarray, lower: float | None, upper: float | None, mask_mode: str) -> np.ndarray:
        mode = str(mask_mode).strip().lower()
        if mode not in {"inside", "outside"}:
            raise ValueError("ThresholdMask mask_mode must be either 'inside' or 'outside'.")

        if mode == "inside":
            matches = np.ones(data.shape, dtype=bool)
            if lower is not None:
                matches &= data >= lower
            if upper is not None:
                matches &= data <= upper
            return matches

        matches = np.zeros(data.shape, dtype=bool)
        if lower is not None:
            matches |= data < lower
        if upper is not None:
            matches |= data > upper
        return matches

    def calculate(self) -> dict[str, DataBundle]:
        cfg = self.configuration

        keys = self._normalised_processing_keys()
        assert len(keys) == 1, "ThresholdMask requires a single databundle processing key."
        processing_key = keys[0]
        target_key = cfg.get("target_mask_key", "threshold_mask")
        source_key = cfg.get("source_basedata_key", "signal")
        lower_bound, upper_bound = self._resolve_bounds(cfg)
        mask_mode = cfg.get("mask_mode", "outside")

        bundle = self.processing_data[processing_key]
        source_bd: BaseData = bundle[source_key]
        source = source_bd.signal

        mask = self._threshold_match(source, lower_bound, upper_bound, mask_mode)
        mask_u32 = mask.astype(np.uint32, copy=False)

        mask_bd = BaseData(signal=mask_u32, units="", rank_of_data=source_bd.rank_of_data)

        self._prepared_data = {target_key: mask_bd}

        output = attach_prepared_data(
            self.processing_data,
            keys,
            self._prepared_data,
            logger=self.logger,
            module_name="ThresholdMask",
        )

        return output
