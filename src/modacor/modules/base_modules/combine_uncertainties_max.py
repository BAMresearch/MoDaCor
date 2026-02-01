# SPDX-License-Identifier: BSD-3-Clause
# /usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

__coding__ = "utf-8"
__authors__ = ["Brian R. Pauw"]  # add names to the list as appropriate
__copyright__ = "Copyright 2026, The MoDaCor team"
__date__ = "20/01/2026"
__status__ = "Development"  # "Development", "Production"
# end of header and standard imports

__all__ = ["CombineUncertaintiesMax"]
__version__ = "20260120.1"

from pathlib import Path

from modacor.dataclasses.databundle import DataBundle
from modacor.dataclasses.process_step import ProcessStep
from modacor.dataclasses.process_step_describer import ProcessStepDescriber
from modacor.dataclasses.uncertainty_tools import (
    combine_uncertainty_keys,
    maximum_aggregator,
    normalize_uncertainty_combinations,
)


class CombineUncertaintiesMax(ProcessStep):
    """Combine uncertainties by taking the element-wise maximum across selected keys."""

    documentation = ProcessStepDescriber(
        calling_name="Combine uncertainties by maximum",
        calling_id="CombineUncertaintiesMax",
        calling_module_path=Path(__file__),
        calling_version=__version__,
        required_data_keys=["signal"],
        modifies={"signal": ["uncertainties"]},
        arguments={
            "target_basedata_key": {
                "type": str,
                "default": "signal",
                "doc": "Name of the BaseData entry within each DataBundle to modify (default: 'signal').",
            },
            "combinations": {
                "type": dict,
                "required": True,
                "default": {},
                "doc": "Mapping of output uncertainty key to an iterable of source keys to combine.",
            },
            "drop_source_keys": {
                "type": bool,
                "default": False,
                "doc": "Remove source uncertainty keys after combination (default: False).",
            },
            "ignore_missing": {
                "type": bool,
                "default": False,
                "doc": (
                    "If True, missing source keys are ignored. "
                    "If all listed keys are missing, the combination is skipped."
                ),
            },
        },
        step_keywords=["uncertainties", "combine", "maximum", "propagation"],
        step_doc="Select the maximum absolute uncertainty among configured source keys.",
        step_reference="DOI 10.1088/0953-8984/25/38/383201",
        step_note=(
            "Useful when systematic contributions must be bounded by the most conservative estimate, "
            "mirroring needs in certain MOUSE workflows."
        ),
    )

    def calculate(self) -> dict[str, DataBundle]:
        combinations_raw = self.configuration.get("combinations", {})
        combinations = normalize_uncertainty_combinations(combinations_raw)
        if not combinations:
            raise ValueError(
                "CombineUncertaintiesMax requires a non-empty 'combinations' mapping in its configuration."
            )

        target_basedata_key = str(self.configuration.get("target_basedata_key", "signal"))
        drop_sources = bool(self.configuration.get("drop_source_keys", False))
        ignore_missing = bool(self.configuration.get("ignore_missing", False))

        output: dict[str, DataBundle] = {}

        for processing_key in self._normalised_processing_keys():
            databundle: DataBundle = self.processing_data.get(processing_key)
            if target_basedata_key not in databundle:
                raise KeyError(f"DataBundle '{processing_key}' does not contain BaseData '{target_basedata_key}'.")

            combine_uncertainty_keys(
                basedata=databundle[target_basedata_key],
                combinations=combinations,
                aggregator=maximum_aggregator,
                drop_sources=drop_sources,
                ignore_missing=ignore_missing,
                logger=self.logger,
                target_name=f"BaseData '{target_basedata_key}' in DataBundle '{processing_key}'",
            )

            output[processing_key] = databundle

        return output
