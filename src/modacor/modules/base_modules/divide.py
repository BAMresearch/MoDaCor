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

__all__ = ["Divide"]
__version__ = "20251029.1"

from pathlib import Path

# from modacor import ureg
# from modacor.dataclasss.basedata import BaseData
from modacor.dataclasses.databundle import DataBundle
from modacor.dataclasses.helpers import basedata_from_sources
from modacor.dataclasses.process_step import ProcessStep
from modacor.dataclasses.process_step_describer import ProcessStepDescriber


class Divide(ProcessStep):
    """
    Divide DataBundle by a BaseData from an IoSource
    """

    documentation = ProcessStepDescriber(
        calling_name="Divide by IoSource data",
        calling_id="DivideBySourceData",
        calling_module_path=Path(__file__),
        calling_version=__version__,
        required_data_keys=["signal"],
        modifies={"signal": ["signal", "uncertainties", "units"]},
        required_arguments={},
        calling_arguments={
            "divisor_source": None,  # IoSources key for signal
            "divisor_units_source": None,  # IoSources key for units
            "divisor_uncertainties_sources": {},  # dict of uncertainty name: source, or 'propagate_to_all': source
        },
        step_keywords=["divide", "scalar", "array"],
        step_doc="Divide a DataBundle element by a divisor loaded from a data source",
        step_reference="DOI 10.1088/0953-8984/25/38/383201",
        step_note="""This loads a scalar (value, units and uncertainty)
            from an IOSource and applies it to the data signal""",
    )

    def calculate(self) -> dict[str, DataBundle]:
        # build up the divisor BaseData object from the IoSources

        divisor = basedata_from_sources(
            io_sources=self.io_sources,
            signal_source=self.configuration.get("divisor_source"),
            units_source=self.configuration.get("divisor_units_source", None),
            uncertainty_sources=self.configuration.get("divisor_uncertainties_sources", {}),
        )

        output: dict[str, DataBundle] = {}

        # actual work happens here:
        for key in self.configuration["with_processing_keys"]:
            databundle = self.processing_data.get(key)
            # divide the data
            # Rely on BaseData.__truediv__ for units + uncertainty propagation
            databundle["signal"] /= divisor
            output[key] = databundle
        return output
