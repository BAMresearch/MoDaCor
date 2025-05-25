# src/modacor/dataclasses/validators.py
# -*- coding: utf-8 -*-
from __future__ import annotations

from numbers import Integral
from typing import Any, Type

import numpy as np

from modacor import ureg

from .databundle import DataBundle
from .messagehandler import MessageHandler

# from .scatteringdata import ScatteringData
_dummy_handler = MessageHandler()

__all__ = [
    "check_data_element_and_units",
    "is_list_of_ints",
]


def is_list_of_ints(instance: Type, attribute: str, value: Any):
    """
    Check if the value is a list of integers.
    """
    if not isinstance(value, list):
        return False
    return all(isinstance(i, Integral) for i in value)


def check_data(
    data: DataBundle,
    data_element_name: str = None,
    required_unit: ureg.Unit = None,
    message_handler: MessageHandler = _dummy_handler,
) -> bool:
    """
    Check that the required data element is present in the DataBundle object.
    """
    if not isinstance(data, DataBundle):
        return False
    if data_element_name is not None:
        if (intensity_object := data.data.get(data_element_name, None)) is None:
            message_handler.error(f"{data_element_name} is required.")
            return False
        if not (intensity_object.internal_units == required_unit):
            message_handler.error(f"{data_element_name} should have units of {required_unit}.")
            return False
    return True


def arrays_are_equal_shape(
    array1: np.ndarray,
    array2: np.ndarray,
) -> bool:
    """
    Check if two arrays have the same shape.
    """
    if array1.shape != array2.shape:
        return False
    return True


def check_data_element_and_units(
    data: DataBundle,
    data_element_name: str,
    required_unit: ureg.Unit,
    message_handler: MessageHandler,
) -> bool:
    """
    Check that the required data element is present with the correct units in the DataBundle object.
    """
    return check_data(data, data_element_name, required_unit, message_handler)
