# SPDX-License-Identifier: BSD-3-Clause
# /usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

__coding__ = "utf-8"
__authors__ = ["Brian R. Pauw"]
__copyright__ = "Copyright 2025, The MoDaCor team"
__date__ = "22/11/2025"
__status__ = "Development"

import pytest
from pint import UnitRegistry
from pint.errors import UndefinedUnitError

from modacor import ureg
from modacor.units import remove_pixel_units


@pytest.fixture
def registry_without_pixels() -> UnitRegistry:
    ureg = UnitRegistry()
    remove_pixel_units(ureg)
    return ureg


@pytest.mark.parametrize(
    "unit_string",
    [
        "pixel",
        "pixels",
        "px",
        "mm/pixel",
        "m/pixel",
        "count/px",
        "counts/pixel/second",
    ],
)
def test_pixel_unit_strings_are_not_defined(registry_without_pixels: UnitRegistry, unit_string: str) -> None:
    with pytest.raises(UndefinedUnitError):
        registry_without_pixels.Unit(unit_string)


def test_application_registry_rejects_pixel_unit_strings() -> None:
    for unit_string in ("pixel", "pixels", "px", "mm/pixel", "count/px"):
        with pytest.raises(UndefinedUnitError):
            ureg.Unit(unit_string)


@pytest.mark.parametrize(
    "unit_string",
    [
        "mm",
        "m",
        "count",
        "counts/second",
        "1/(m sr)",
    ],
)
def test_normal_units_still_parse(registry_without_pixels: UnitRegistry, unit_string: str) -> None:
    registry_without_pixels.Unit(unit_string)


def test_normal_rate_units_still_convert(registry_without_pixels: UnitRegistry) -> None:
    rate = 2.0 * registry_without_pixels.count / registry_without_pixels.second

    assert rate.to("count/minute").magnitude == pytest.approx(120.0)


def test_detector_pitch_is_plain_length(registry_without_pixels: UnitRegistry) -> None:
    pitch = 0.172 * registry_without_pixels.mm
    detector_index_delta = 100.0
    length = pitch * detector_index_delta

    assert length.to("mm").magnitude == pytest.approx(17.2)


def test_detector_element_area_is_plain_area(registry_without_pixels: UnitRegistry) -> None:
    pitch_fast = 0.172 * registry_without_pixels.mm
    pitch_slow = 0.200 * registry_without_pixels.mm

    area = pitch_fast * pitch_slow

    assert area.to("mm^2").magnitude == pytest.approx(0.0344)
