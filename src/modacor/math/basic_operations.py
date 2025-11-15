# SPDX-License-Identifier: BSD-3-Clause
# /usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

__coding__ = "utf-8"
__authors__ = ["Brian R. Pauw"]  # add names to the list as appropriate
__copyright__ = "Copyright 2025, The MoDaCor team"
__date__ = "29/10/2025"
__status__ = "Development"  # "Development", "Production"
# end of header and standard imports

__all__ = [
    "divide_basedata_elements",
    "multiply_basedata_elements",
    "add_basedata_elements",
    "subtract_basedata_elements",
]

import numpy as np
from auto_uncertainties import Uncertainty

# from modacor import ureg
from modacor.dataclasses.basedata import BaseData

# basic operations between two basedata elements with uncertainty and units propagation, courtesy of auto_uncertainties package


def divide_basedata_elements(numerator: BaseData, denominator: BaseData) -> BaseData:
    """Divide two BaseData elements with uncertainty and units propagation"""
    base_result = numerator.signal * numerator.units / (denominator.signal * denominator.units)
    result = BaseData(
        signal=base_result.magnitude,
        units=base_result.units,
        uncertainties={
            key: np.full(base_result.shape, np.nan) for key in numerator.uncertainties.keys()
        },  # reserve memory
    )

    for key, unc in result.uncertainties.items():
        if key in denominator.uncertainties:
            result.uncertainties[key] = (
                Uncertainty(numerator.signal, numerator.uncertainties[key])
                / Uncertainty(denominator.signal, denominator.uncertainties[key])
            ).error
        if "propagate_to_all" in denominator.uncertainties:
            result.uncertainties[key] = (
                Uncertainty(numerator.signal, numerator.uncertainties[key])
                / Uncertainty(denominator.signal, denominator.uncertainties["propagate_to_all"])
            ).error

    return result


def multiply_basedata_elements(factor1: BaseData, factor2: BaseData) -> BaseData:
    """Multiply two BaseData elements with uncertainty and units propagation"""
    base_result = factor1.signal * factor1.units * (factor2.signal * factor2.units)
    result = BaseData(
        signal=base_result.magnitude,
        units=base_result.units,
        uncertainties={
            key: np.full(base_result.shape, np.nan) for key in factor1.uncertainties.keys()
        },  # reserve memory
    )

    for key, unc in result.uncertainties.items():
        if key in factor2.uncertainties:
            result.uncertainties[key] = (
                Uncertainty(factor1.signal, factor1.uncertainties[key])
                * Uncertainty(factor2.signal, factor2.uncertainties[key])
            ).error
        if "propagate_to_all" in factor2.uncertainties:
            result.uncertainties[key] = (
                Uncertainty(factor1.signal, factor1.uncertainties[key])
                * Uncertainty(factor2.signal, factor2.uncertainties["propagate_to_all"])
            ).error

    return result


def add_basedata_elements(addend1: BaseData, addend2: BaseData) -> BaseData:
    """Add two BaseData elements with uncertainty and units propagation"""
    base_result = addend1.signal * addend1.units + (addend2.signal * addend2.units)
    result = BaseData(
        signal=base_result.magnitude,
        units=base_result.units,
        uncertainties={
            key: np.full(base_result.shape, np.nan) for key in addend1.uncertainties.keys()
        },  # reserve memory
    )

    for key, unc in result.uncertainties.items():
        if key in addend2.uncertainties:
            result.uncertainties[key] = (
                Uncertainty(addend1.signal, addend1.uncertainties[key])
                + Uncertainty(addend2.signal, addend2.uncertainties[key])
            ).error
        if "propagate_to_all" in addend2.uncertainties:
            result.uncertainties[key] = (
                Uncertainty(addend1.signal, addend1.uncertainties[key])
                + Uncertainty(addend2.signal, addend2.uncertainties["propagate_to_all"])
            ).error

    return result


def subtract_basedata_elements(minuend: BaseData, subtrahend: BaseData) -> BaseData:
    """Subtract two BaseData elements with uncertainty and units propagation"""
    base_result = minuend.signal * minuend.units - (subtrahend.signal * subtrahend.units)
    result = BaseData(
        signal=base_result.magnitude,
        units=base_result.units,
        uncertainties={
            key: np.full(base_result.shape, np.nan) for key in minuend.uncertainties.keys()
        },  # reserve memory
    )

    for key, unc in result.uncertainties.items():
        if key in subtrahend.uncertainties:
            result.uncertainties[key] = (
                Uncertainty(minuend.signal, minuend.uncertainties[key])
                - Uncertainty(subtrahend.signal, subtrahend.uncertainties[key])
            ).error
        if "propagate_to_all" in subtrahend.uncertainties:
            result.uncertainties[key] = (
                Uncertainty(minuend.signal, minuend.uncertainties[key])
                - Uncertainty(subtrahend.signal, subtrahend.uncertainties["propagate_to_all"])
            ).error

    return result
