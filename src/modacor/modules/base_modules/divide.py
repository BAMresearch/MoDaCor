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

from modacor import ureg
from modacor.dataclasses.basedata import BaseData
from modacor.dataclasses.databundle import DataBundle
from modacor.dataclasses.process_step import ProcessStep
from modacor.dataclasses.process_step_describer import ProcessStepDescriber

# from modacor.dataclasses.processing_data import ProcessingData
from modacor.math.basic_operations import divide_basedata_elements


class Divide(ProcessStep):
    """
    Divide BaseData by another BaseData
    """

    documentation = ProcessStepDescriber(
        calling_name="Divide by IoSource data",
        calling_id="DivideBySourceData",
        calling_module_path=Path(__file__),
        calling_version=__version__,
        required_data_keys=["signal"],
        works_on={"signal": ["signal", "uncertainties", "units"]},
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
        # t = sources.get_data('sample::entry/instrument/detector/frame_exposure_time')
        # t_u_key = sources.get_data_attributes('sample::entry/instrument/detector/frame_exposure_time')['uncertainties']
        # t_u = {'propagate_to_all': sources.get_data(f'sample::entry/instrument/detector/{t_u_key}')} if t_u_key is not None else {}
        # t_units = sources.get_data_attributes('sample::entry/instrument/detector/frame_exposure_time')['units']
        # t_bd = BaseData(signal=t, uncertainties=t_u, units=ureg.Unit(t_units))

        s_source = self.configuration.get("divisor_source")
        u_sources = self.configuration.get("divisor_uncertainties_sources", {})
        unit_source = self.configuration.get("divisor_units_source", None)
        divisor = BaseData(
            signal=self.io_sources.get_data(s_source),
            units=ureg.Unit(self.io_sources.get_data(unit_source)) if unit_source is not None else ureg.dimensionless,
            uncertainties={k: self.io_sources.get_data(v) for k, v in u_sources},
        )
        # Get the data
        data = self.processing_data

        output = {}
        for key in self.configuration["with_processing_keys"]:
            databundle = data.get(key)
            signal = databundle["signal"]
            # divide the data

            output[key] = divide_basedata_elements(signal, divisor)
        return output
