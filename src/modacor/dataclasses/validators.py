# src/modacor/dataclasses/validators.py
# -*- coding: utf-8 -*-

from __future__ import annotations
from modacor.dataclasses import ScatteringData
from modacor.dataclasses.messagehandler import MessageHandler
import pint


def check_data_element_and_units(
        data: ScatteringData,
        data_element_name: str,
        required_unit: pint.Unit,
        message_handler: MessageHandler
        ) -> bool:
    """
    Check that the required data element is present with the correct units in the ScatteringData object.
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
