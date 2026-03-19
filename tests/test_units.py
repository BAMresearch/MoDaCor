# SPDX-License-Identifier: BSD-3-Clause
# /usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

__coding__ = "utf-8"
__authors__ = ["Brian R. Pauw"]  # add names to the list as appropriate
__copyright__ = "Copyright 2025, The MoDaCor team"
__date__ = "22/11/2025"
__status__ = "Development"  # "Development", "Production"

import pytest
from pint import UnitRegistry
from pint.errors import DimensionalityError

from modacor.units import configure_detector_pixel_units


def test_pixel_dimension_is_detector_element_after_configure() -> None:
    ureg = UnitRegistry()

    # Pint usually defines pixel as a printing/display unit by default.
    # We only assert the post-condition to keep this robust across Pint variants.
    configure_detector_pixel_units(ureg)

    dim_pixel = (1 * ureg.pixel).dimensionality
    assert "[detector_pixel]" in dim_pixel
    assert "[printing_unit]" not in dim_pixel

    # Aliases must all resolve to the same dimensionality
    assert (1 * ureg.px).dimensionality == dim_pixel
    assert (1 * ureg.pixels).dimensionality == dim_pixel


def test_parsing_common_metadata_spellings() -> None:
    ureg = UnitRegistry()
    configure_detector_pixel_units(ureg)

    q1 = ureg.Quantity("1 pixel")
    q2 = ureg.Quantity("1 pixels")
    q3 = ureg.Quantity("1 px")

    assert q1.dimensionality == q2.dimensionality == q3.dimensionality
    assert q1.to("pixel").magnitude == pytest.approx(1.0)
    assert q2.to("pixel").magnitude == pytest.approx(1.0)
    assert q3.to("pixel").magnitude == pytest.approx(1.0)


def test_mm_per_pixel_is_not_convertible_to_mm() -> None:
    ureg = UnitRegistry()
    configure_detector_pixel_units(ureg)

    q = 0.172 * ureg.mm / ureg.pixel
    with pytest.raises(DimensionalityError):
        _ = q.to("mm")


def test_pixel_cancels_when_multiplying_by_pixel_count() -> None:
    ureg = UnitRegistry()
    configure_detector_pixel_units(ureg)

    pixel_size = 0.172 * ureg.mm / ureg.pixel
    length = pixel_size * (100 * ureg.pixel)

    assert length.to("mm").magnitude == pytest.approx(17.2)


def test_area_and_volume_per_pixel_behave_sensibly() -> None:
    ureg = UnitRegistry()
    configure_detector_pixel_units(ureg)

    area_per_pixel = 0.0296 * ureg.mm**2 / ureg.pixel
    area = area_per_pixel * (10 * ureg.pixel)
    assert area.to("mm^2").magnitude == pytest.approx(0.296)

    vol_per_pixel = 0.005 * ureg.mm**3 / ureg.pixel
    vol = vol_per_pixel * (12 * ureg.px)  # alias
    assert vol.to("mm^3").magnitude == pytest.approx(0.06)
