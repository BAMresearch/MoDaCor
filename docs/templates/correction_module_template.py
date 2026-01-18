# SPDX-License-Identifier: BSD-3-Clause
# /usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

__coding__ = "utf-8"
__authors__ = ["Your Name"]  # add names to the list as appropriate
__copyright__ = "Copyright 2025, The MoDaCor team"
__date__ = "01/01/2026"
__status__ = "Development"  # "Development", "Production"
# end of header and standard imports

__all__ = ["YourCorrectionStep"]
__version__ = "20260101.1"

from pathlib import Path
from typing import Any

import numpy as np

from modacor.dataclasses.basedata import BaseData
from modacor.dataclasses.databundle import DataBundle
from modacor.dataclasses.messagehandler import MessageHandler
from modacor.dataclasses.process_step import ProcessStep
from modacor.dataclasses.process_step_describer import ProcessStepDescriber

# Module-level handler; facilities can swap MessageHandler implementation as needed
logger = MessageHandler(name=__name__)


class YourCorrectionStep(ProcessStep):
    """
    Template for a correction module.

    Fill in:
    - required_data_keys: BaseData keys required in each DataBundle
    - default_configuration: runtime configuration defaults
    - argument_specs: docs for configuration keys
    - calculate(): core logic, returning updated DataBundles
    """

    documentation = ProcessStepDescriber(
        calling_name="My Correction Step",
        calling_id="YourCorrectionStep",
        calling_module_path=Path(__file__),
        calling_version=__version__,
        required_data_keys=["signal"],
        modifies={"signal": ["signal", "uncertainties", "units"]},
        required_arguments=[],
        default_configuration={
            # Use with_processing_keys to select which ProcessingData entries to update.
            # None -> auto-apply when only one DataBundle exists.
            "with_processing_keys": None,
            # Example parameter for a scalar correction factor.
            "correction_factor": 1.0,
            # Optional: name of BaseData key to read/write.
            "signal_key": "signal",
        },
        argument_specs={
            "with_processing_keys": {
                "type": (str, list, type(None)),
                "required": False,
                "doc": "ProcessingData key(s) to correct (string, list, or None).",
            },
            "correction_factor": {
                "type": (float, int),
                "required": False,
                "doc": "Scalar factor applied to the signal.",
            },
            "signal_key": {
                "type": str,
                "required": False,
                "doc": "BaseData key to read/write within each DataBundle.",
            },
        },
        step_keywords=["correction", "scale"],
        step_doc="Apply a simple scalar correction factor to signal data.",
        step_reference="",
        step_note="Replace this note with details about the correction algorithm.",
    )

    # ------------------------------------------------------------------
    # Optional helper: one-time setup before execution
    # ------------------------------------------------------------------
    def prepare_execution(self) -> None:
        """
        Pre-compute any expensive intermediates and cache them on the instance.
        This runs once per ProcessStep instance (until reset()).
        """
        # Example: precompute a lookup table or kernel
        # self._prepared_data["lookup"] = compute_lookup(...)
        # logger.debug("Prepared lookup table for correction step.")
        pass

    # Core calculation
    # ------------------------------------------------------------------
    def calculate(self) -> dict[str, DataBundle]:
        """
        Apply the correction and return updated DataBundles.
        """
        output: dict[str, DataBundle] = {}

        keys = self._normalised_processing_keys()
        signal_key = self.configuration.get("signal_key", "signal")
        factor = float(self.configuration.get("correction_factor", 1.0))
        lookup = self._prepared_data.get("lookup", None)

        for key in keys:
            databundle = self.processing_data.get(key)
            if databundle is None:
                raise KeyError(f"ProcessingData key {key!r} not found.")

            # Copy if you need to avoid mutating inputs; otherwise edit in-place.
            signal_bd: BaseData = databundle[signal_key]

            # Example correction: multiply signal by a scalar, optionally using a cached lookup
            signal = np.asarray(signal_bd.signal)
            if lookup is not None:
                signal = signal * lookup
            signal_bd.signal = signal * factor

            # Optional: log what happened (avoid overly chatty logs in production)
            logger.debug("Applied correction_factor=%s to %s::%s", factor, key, signal_key)

            output[key] = databundle

        return output
