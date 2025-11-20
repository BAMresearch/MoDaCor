# SPDX-License-Identifier: BSD-3-Clause
# /usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from modacor import ureg
from modacor.dataclasses.basedata import BaseData
from modacor.io.io_sources import IoSources

__coding__ = "utf-8"
__authors__ = ["Brian R. Pauw", "Armin Moser"]  # add names to the list as appropriate
__copyright__ = "Copyright 2025, The MoDaCor team"
__date__ = "29/10/2025"
__status__ = "Development"  # "Development", "Production"
# end of header and standard imports

__all__ = ["basedata_from_sources"]


def basedata_from_sources(
    io_sources: IoSources,
    signal_source: str,
    units_source: str | None = None,
    uncertainty_sources: dict[str, str] = {},
) -> BaseData:
    """Helper function to build a BaseData object from IoSources

    Parameters
    ----------
    io_sources : IoSources
        The IoSources object to load data from.
    signal_source : str
        The source key for the signal data.
    unit_source : str | None, optional
        The source key for the units data, by default None.
        for iosources that support attributes, the units can also be stored as an attribute.
        In that case, it can be specified by 'key to the dataset@[units_attribute_name]'
    uncertainty_sources : dict[str, str], optional
        A dictionary mapping uncertainty names to their source keys, by default an empty dictionary.
    """
    signal = io_sources.get_data(signal_source)
    units = ureg.Unit(io_sources.get_static_metadata(units_source)) if units_source is not None else ureg.dimensionless
    uncertainties = {name: io_sources.get_data(source) for name, source in uncertainty_sources.items()}
    return BaseData(signal=signal, units=units, uncertainties=uncertainties)
