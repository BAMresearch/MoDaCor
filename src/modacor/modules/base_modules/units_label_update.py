# SPDX-License-Identifier: BSD-3-Clause
# /usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

__coding__ = "utf-8"
__authors__ = ["Brian R. Pauw", "Armin Moser"]  # add names to the list as appropriate
__copyright__ = "Copyright 2025, The MoDaCor team"
__date__ = "16/12/2025"
__status__ = "Development"  # "Development", "Production"

__all__ = ["UnitsLabelUpdate"]
__version__ = "20251216.1"

from pathlib import Path

from modacor import ureg
from modacor.dataclasses.databundle import DataBundle
from modacor.dataclasses.process_step import ProcessStep
from modacor.dataclasses.process_step_describer import ProcessStepDescriber


class UnitsLabelUpdate(ProcessStep):
    """
    Update the units of one or more BaseData elements in a DataBundle.
    Note: this only changes the *unit label* (no numerical conversion).
    """

    documentation = ProcessStepDescriber(
        calling_name="Update unit labels",
        calling_id="UnitsLabelUpdate",
        calling_module_path=Path(__file__),
        calling_version=__version__,
        required_data_keys=[""],  # provided via update_pairs
        modifies={"": ["units"]},
        default_configuration={
            # update_pairs:
            #   <basedata_key>: { units: "<pint unit str>" }
            # or shorthand:
            #   <basedata_key>: "<pint unit str>"
            "update_pairs": {},
        },
        argument_specs={
            "update_pairs": {
                "type": dict,
                "required": True,
                "doc": "Mapping of BaseData key to unit string or {'units': str}.",
            },
        },
        step_keywords=["units", "update", "standardize"],
        step_doc="Update unit labels of one or more BaseData elements (no conversion).",
        step_reference="DOI 10.1088/0953-8984/25/38/383201",
    )

    def calculate(self) -> dict[str, DataBundle]:
        pairs = self.configuration["update_pairs"]
        parsed = {
            bd_key: ureg.Unit(spec["units"] if isinstance(spec, dict) else spec) for bd_key, spec in pairs.items()
        }

        output: dict[str, DataBundle] = {}
        for key in self._normalised_processing_keys():
            databundle = self.processing_data.get(key)
            for bd_key, unit in parsed.items():
                databundle[bd_key].units = unit
            output[key] = databundle
            info_msg = f"UnitsLabelUpdate: updated units for DataBundle '{key}': " + ", ".join(
                f"{bd_key} -> {databundle[bd_key].units}" for bd_key in parsed.keys()
            )
            self.logger.info(info_msg)
        return output
