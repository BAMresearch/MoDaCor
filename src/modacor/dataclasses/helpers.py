# SPDX-License-Identifier: BSD-3-Clause
# /usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

__coding__ = "utf-8"
__authors__ = ["Brian R. Pauw", "Armin Moser"]  # add names to the list as appropriate
__copyright__ = "Copyright 2025, The MoDaCor team"
__date__ = "16/11/2025"
__status__ = "Development"  # "Development", "Production"
# end of header and standard imports

import numpy as np

from modacor import ureg
from modacor.dataclasses.basedata import BaseData
from modacor.io.io_sources import IoSources

__all__ = ["basedata_from_sources"]


def basedata_from_sources(
    io_sources: IoSources,
    signal_source: str | None,
    units_source: str | None = None,
    uncertainty_sources: dict[str, str] = {},
    weights_source: str | None = None,
) -> BaseData:
    """Helper function to build a BaseData object from IoSources

    Parameters
    ----------
    io_sources : IoSources
        The IoSources object to load data from.
    signal_source : str | None
        Required non-blank source key for the signal data. None raises ValueError.
    unit_source : str | None, optional
        The source key for the units data, by default None.
        for iosources that support attributes, the units can also be stored as an attribute.
        In that case, it can be specified by 'key to the dataset@[units_attribute_name]'
    uncertainty_sources : dict[str, str], optional
        A dictionary mapping uncertainty names to their source keys, by default an empty dictionary.
    weights_source : str | None, optional
        Optional source key for BaseData weights, by default None.
    """
    if signal_source is None or not signal_source.strip():
        raise ValueError("signal_source is required")
    signal = io_sources.get_data(signal_source)
    units = ureg.Unit(io_sources.get_static_metadata(units_source)) if units_source is not None else ureg.dimensionless
    uncertainties = {name: io_sources.get_data(source) for name, source in uncertainty_sources.items()}
    kwargs = {"signal": signal, "units": units, "uncertainties": uncertainties}
    if weights_source is not None:
        kwargs["weights"] = io_sources.get_data(weights_source)
    return BaseData.from_scalar(**kwargs) if np.size(signal) == 1 else BaseData(**kwargs)
