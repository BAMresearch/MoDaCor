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

from modacor.dataclasses.databundle import DataBundle
from modacor.dataclasses.helpers import basedata_from_sources
from modacor.dataclasses.process_step import ProcessStep
from modacor.dataclasses.process_step_describer import ProcessStepDescriber


class Multiply(ProcessStep):
    """
    Multiply a DataBundle by a BaseData from an IoSource
    """

    documentation = ProcessStepDescriber(
        calling_name="Multiply by IoSource data",
        calling_id="MultiplyBySourceData",
        calling_module_path=Path(__file__),
        calling_version=__version__,
        required_data_keys=["signal"],
        modifies={"signal": ["signal", "uncertainties", "units"]},
        default_configuration={
            "multiplier_source": None,  # IoSources key for signal
            "multiplier_units_source": None,  # IoSources key for units
            "multiplier_uncertainties_sources": {},  # dict of uncertainty name: source, or 'propagate_to_all': source
        },
        argument_specs={
            "multiplier_source": {
                "type": str,
                "required": False,
                "doc": "IoSources key for the multiplier signal.",
            },
            "multiplier_units_source": {
                "type": str,
                "required": False,
                "doc": "IoSources key for multiplier units metadata.",
            },
            "multiplier_uncertainties_sources": {
                "type": dict,
                "required": False,
                "doc": "Mapping of uncertainty name to IoSources key.",
            },
        },
        step_keywords=["multiply", "scalar", "array"],
        step_doc="Multiply a DataBundle element by a multiplier loaded from a data source",
        step_reference="DOI 10.1088/0953-8984/25/38/383201",
        step_note="""This loads a scalar (value, units and uncertainty)
            from an IOSource and applies it to the data signal""",
    )

    def calculate(self) -> dict[str, DataBundle]:
        # build up the multiplier BaseData object from the IoSources
        multiplier = basedata_from_sources(
            io_sources=self.io_sources,
            signal_source=self.configuration.get("multiplier_source"),
            units_source=self.configuration.get("multiplier_units_source", None),
            uncertainty_sources=self.configuration.get("multiplier_uncertainties_sources", {}),
        )

        output: dict[str, DataBundle] = {}
        # actual work happens here:
        for key in self.configuration["with_processing_keys"]:
            databundle = self.processing_data.get(key)
            # multiply the data
            databundle["signal"] *= multiplier
            output[key] = databundle
        return output
