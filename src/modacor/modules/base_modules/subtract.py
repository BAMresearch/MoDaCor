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

__all__ = ["Subtract"]
__version__ = "20251029.1"

from pathlib import Path

from modacor.dataclasses.databundle import DataBundle
from modacor.dataclasses.helpers import basedata_from_sources
from modacor.dataclasses.process_step import ProcessStep
from modacor.dataclasses.process_step_describer import ProcessStepDescriber

# from modacor.dataclasses.processing_data import ProcessingData
from modacor.math.basic_operations import subtract_basedata_elements


class Subtract(ProcessStep):
    """
    Subtract a DataBundle by a BaseData from an IoSource
    """

    documentation = ProcessStepDescriber(
        calling_name="Subtract by IoSource data",
        calling_id="SubtractBySourceData",
        calling_module_path=Path(__file__),
        calling_version=__version__,
        required_data_keys=["signal"],
        works_on={"signal": ["signal", "uncertainties", "units"]},
        calling_arguments={
            "subtrahend_source": None,  # IoSources key for signal
            "subtrahend_units_source": None,  # IoSources key for units
            "subtrahend_uncertainties_sources": {},  # dict of uncertainty name: source, or 'propagate_to_all': source
        },
        step_keywords=["subtract", "scalar", "array"],
        step_doc="Subtract a DataBundle element by a subtrahend loaded from a data source",
        step_reference="DOI 10.1088/0953-8984/25/38/383201",
        step_note="""This loads a scalar (value, units and uncertainty)
            from an IOSource and applies it to the data signal""",
    )

    def calculate(self) -> dict[str, DataBundle]:
        # build up the subtrahend BaseData object from the IoSources
        subtrahend = basedata_from_sources(
            io_sources=self.io_sources,
            signal_source=self.configuration.get("subtrahend_source"),
            units_source=self.configuration.get("subtrahend_units_source", None),
            uncertainty_sources=self.configuration.get("subtrahend_uncertainties_sources", {}),
        )
        # Get the data
        data = self.processing_data

        output = {}
        # actual work happens here:
        for key in self.configuration["with_processing_keys"]:
            databundle = data.get(key)
            # subtract the data
            # databundle['signal'] is a BaseData object
            databundle["signal"] = subtract_basedata_elements(databundle["signal"], subtrahend)
            output[key] = databundle
        return output
