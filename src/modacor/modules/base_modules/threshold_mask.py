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

import numpy as np

from modacor.dataclasses.basedata import BaseData
from modacor.dataclasses.databundle import DataBundle
from modacor.dataclasses.process_step import ProcessStep
from modacor.dataclasses.process_step_describer import ProcessStepDescriber
from modacor.modules.helpers import attach_prepared_data


class ThresholdMask(ProcessStep):
    """
    Create a mask based on a threshold, where values of 'signal' above the threshold are
    masked. This can be used to mask detector panel gaps marked by a large integer and
    hot pixels.

    MoDaCor's Masks are 32-bit integer bitfields (NeXus convention). This step creates a
    new mask at the specified target_mask_key (default: threshold_mask).
    """

    documentation = ProcessStepDescriber(
        calling_name="Mask signal above a threshold",
        calling_id="ThresholdMask",
        calling_module_path=Path(__file__),
        calling_version=__version__,
        required_data_keys=["signal"],
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
            },
            "threshold": {
                "type": float,
                "default": 1.0e9,
                "doc": "Threshold above which the data shall be masked.",
            },
        },
        step_keywords=["mask", "threshold", "databundle"],
        step_doc="Create a mask that masks values in 'signal' above a given threshold.",
        step_reference="",
        step_note="""
            Configuration:
              with_processing_keys: [sample]     # required, single databundle key
              target_mask_key: threshold_mask    # optional, default: threshold_mask
              threshold: 1e9                     # optional, default: 1e9

            Performs:
              threshold_mask = np.where(signal > threshold, 1, 0)
        """,
    )

    def calculate(self) -> dict[str, DataBundle]:
        cfg = self.configuration

        keys = self._normalised_processing_keys()
        assert len(keys) == 1, (
            "ThresholdMask requires a single databundle processing key."
        )
        processing_key = keys[0]
        target_key = cfg.get("target_mask_key", "threshold_mask")
        threshold = float(cfg.get("threshold", 1.0e9))

        assert isinstance(threshold, float), (
            "threshold must be a floating-point number."
        )

        bundle = self.processing_data[processing_key]
        signal_bd: BaseData = bundle["signal"]
        signal = signal_bd.signal

        # signal may have more than 2 dimensions but mask should be of rank 2
        while signal.ndim > 2:
            signal = np.mean(signal, axis=0)

        mask = np.where(signal > threshold, 1, 0)
        # Convert only if needed (uint8/int16/etc -> uint32)
        mask_u32 = (
            mask if mask.dtype == np.uint32 else mask.astype(np.uint32, copy=False)
        )

        mask_bd = BaseData(signal=mask_u32, units="", rank_of_data=2)

        self._prepared_data = {target_key: mask_bd}

        output = attach_prepared_data(
            self.processing_data,
            keys,
            self._prepared_data,
            logger=self.logger,
            module_name="ThresholdMask",
        )

        return output

        # return {processing_key: bundle}
