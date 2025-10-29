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

__all__ = ["Multiply"]
__version__ = "20251029.1"

from pathlib import Path

from modacor import ureg
from modacor.dataclasses.basedata import BaseData
from modacor.dataclasses.databundle import DataBundle
from modacor.dataclasses.process_step import ProcessStep
from modacor.dataclasses.process_step_describer import ProcessStepDescriber

# from modacor.dataclasses.processing_data import ProcessingData
from modacor.math.basic_operations import multiply_basedata_elements


class Multiply(ProcessStep):
    """
    Multiply BaseData with another BaseData
    """

    documentation = ProcessStepDescriber(
        calling_name="Multiply by IoSource data",
        calling_id="MultiplyBySourceData",
        calling_module_path=Path(__file__),
        calling_version=__version__,
        required_data_keys=["signal"],
        works_on={"signal": ["signal", "uncertainties", "units"]},
        calling_arguments={
            "multiplier_source": None,  # IoSources key for signal
            "multiplier_units_source": None,  # IoSources key for units
            "multiplier_uncertainties_sources": {},  # dict of uncertainty name: source, or 'propagate_to_all': source
        },
        step_keywords=["multiply", "scalar", "array"],
        step_doc="Multiply a DataBundle element by a multiplier loaded from a data source",
        step_reference="DOI 10.1088/0953-8984/25/38/383201",
        step_note="""This loads a scalar (value, units and uncertainty)
            from an IOSource and applies it to the data signal""",
    )

    def calculate(self) -> dict[str, DataBundle]:
        # build up the multiplier BaseData object from the IoSources
        u_sources = self.configuration.get("multiplier_uncertainties_sources", {}).items()
        unit_source = self.configuration.get("multiplier_units_source", None)
        multiplier = BaseData(
            signal=self.io_sources.load_data(self.configuration.get("multiplier_source")),
            units=ureg.Unit(self.io_sources.load_data(unit_source)) if unit_source is not None else ureg.dimensionless,
            uncertainties={k: self.io_sources.load_data(v) for k, v in u_sources},
        )
        # Get the data
        data = self.processing_data

        output = {}
        for key in self.configuration["with_processing_keys"]:
            databundle = data.get(key)
            signal = databundle["signal"]
            # multiply the data

            output[key] = multiply_basedata_elements(signal, multiplier)
        return output
