# SPDX-License-Identifier: BSD-3-Clause
# /usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

__coding__ = "utf-8"
__authors__ = ["Brian R. Pauw", "Armin Moser"]  # add names to the list as appropriate
__copyright__ = "Copyright 2025, The MoDaCor team"
__date__ = "29/10/2025"
__status__ = "Development"  # "Development", "Production"
# end of header and standard imports

__all__ = ["SolidAngleCorrection"]
__version__ = "20251029.1"

from pathlib import Path

# from modacor import ureg
# from modacor.dataclasss.basedata import BaseData
from modacor.dataclasses.databundle import DataBundle
from modacor.dataclasses.process_step import ProcessStep
from modacor.dataclasses.process_step_describer import ProcessStepDescriber


class SolidAngleCorrection(ProcessStep):
    """
    Normalize a signal by a solid angle "Omega" calculated using XSGeometry
    """

    documentation = ProcessStepDescriber(
        calling_name="Solid Angle Correction",
        calling_id="SolidAngleCorrection",
        calling_module_path=Path(__file__),
        calling_version=__version__,
        required_data_keys=["signal", "Omega"],
        modifies={"signal": ["signal", "uncertainties", "units"]},
        arguments={
            "with_processing_keys": {
                "type": list,
                "required": True,
                "default": None,
                "doc": "ProcessingData keys whose signal should be divided by Omega.",
            },
        },
        step_keywords=["divide", "normalize", "solid angle"],
        step_doc="Divide the pixels in a signal by their solid angle coverage",
        step_reference="DOI 10.1088/0953-8984/25/38/383201",
        step_note="""This divides the signal by the value previously calculated
            using the XSGeometry module""",
    )

    def calculate(self) -> dict[str, DataBundle]:
        output: dict[str, DataBundle] = {}

        # actual work happens here:
        for key in self._normalised_processing_keys():
            databundle = self.processing_data.get(key)
            # divide the data
            # Rely on BaseData.__truediv__ for units + uncertainty propagation
            databundle["signal"] /= databundle["Omega"]
            output[key] = databundle
        return output
