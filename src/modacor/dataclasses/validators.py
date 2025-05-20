# src/modacor/dataclasses/validators.py
# -*- coding: utf-8 -*-

from __future__ import annotations
from modacor.dataclasses import DataBundle
from modacor.dataclasses.messagehandler import MessageHandler
import pint

def is_list_of_ints(value):
    """
    Check if the value is a list of integers.
    """
    if not isinstance(value, list):
        return False
    return all(isinstance(i, int) for i in value)

def check_data_element_and_units(
        data: DataBundle,
        data_element_name: str,
        required_unit: pint.Unit,
        message_handler: MessageHandler
        ) -> bool:
    """
    Check that the required data element is present with the correct units in the DataBundle object.
    """
    # Check if the required data is available.. these checks should probably be abstracted and made generally available.
    if (intensity_object := data.data.get(data_element_name, None)) is None:
        message_handler.error(
            f"{data_element_name} is required."
        )
        return False
    if not (intensity_object.internal_units == required_unit):
        message_handler.error(
            f"{data_element_name} should have units of {required_unit}."
        )
        return False

    return True
