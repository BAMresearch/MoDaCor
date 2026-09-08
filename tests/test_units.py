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

from modacor import ureg
from modacor.units import configure_detector_pixel_units


@pytest.fixture
def registry_with_detector_pixels() -> UnitRegistry:
    ureg = UnitRegistry()
    configure_detector_pixel_units(ureg)
    return ureg


@pytest.mark.parametrize(
    "unit_string",
    [
        "pixel",
        "pixels",
        "px",
        "css_pixel",
        "dot",
        "pel",
        "picture_element",
    ],
)
def test_pixel_unit_strings_are_dimensionless(
    registry_with_detector_pixels: UnitRegistry,
    unit_string: str,
) -> None:
    unit = registry_with_detector_pixels.Unit(unit_string)

    assert unit.is_compatible_with(registry_with_detector_pixels.dimensionless)
    assert (1 * unit).to(registry_with_detector_pixels.dimensionless).magnitude == pytest.approx(1.0)
    assert (1 * unit).to_base_units().units == registry_with_detector_pixels.dimensionless


def test_application_registry_accepts_dimensionless_pixel_unit_strings() -> None:
    for unit_string in ("pixel", "pixels", "px", "css_pixel", "dot", "pel", "picture_element"):
        assert ureg.Unit(unit_string).is_compatible_with(ureg.dimensionless)


def test_pixel_denominators_cancel_from_physical_units(registry_with_detector_pixels: UnitRegistry) -> None:
    pitch = 0.172 * registry_with_detector_pixels.Unit("mm/pixel")
    counts_per_pixel = 5.0 * registry_with_detector_pixels.Unit("count/px")
    count_rate_per_pixel = 10.0 * registry_with_detector_pixels.Unit("counts/pixel/second")

    assert pitch.to("mm").magnitude == pytest.approx(0.172)
    assert counts_per_pixel.to("count").magnitude == pytest.approx(5.0)
    assert count_rate_per_pixel.to("count/second").magnitude == pytest.approx(10.0)


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
def test_normal_units_still_parse(registry_with_detector_pixels: UnitRegistry, unit_string: str) -> None:
    registry_with_detector_pixels.Unit(unit_string)


def test_normal_rate_units_still_convert(registry_with_detector_pixels: UnitRegistry) -> None:
    rate = 2.0 * registry_with_detector_pixels.count / registry_with_detector_pixels.second

    assert rate.to("count/minute").magnitude == pytest.approx(120.0)


def test_detector_pitch_is_plain_length(registry_with_detector_pixels: UnitRegistry) -> None:
    pitch = 0.172 * registry_with_detector_pixels.mm
    detector_index_delta = 100.0
    length = pitch * detector_index_delta

    assert length.to("mm").magnitude == pytest.approx(17.2)


def test_detector_element_area_is_plain_area(registry_with_detector_pixels: UnitRegistry) -> None:
    pitch_fast = 0.172 * registry_with_detector_pixels.mm
    pitch_slow = 0.200 * registry_with_detector_pixels.mm

    area = pitch_fast * pitch_slow

    assert area.to("mm^2").magnitude == pytest.approx(0.0344)
