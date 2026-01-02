# SPDX-License-Identifier: BSD-3-Clause
# /usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

__coding__ = "utf-8"
__authors__ = ["Brian R. Pauw"]  # add names to the list as appropriate
__copyright__ = "Copyright 2025, The MoDaCor team"
__date__ = "02/01/2026"
__status__ = "Development"  # "Development", "Production"

import numpy as np
import pytest

from modacor import ureg
from modacor.dataclasses.basedata import BaseData


def test_to_base_units_scales_signal_and_uncertainties():
    bd = BaseData(
        signal=np.array([100.0, 200.0]),  # cm
        units=ureg.cm,
        uncertainties={"propagate_to_all": np.array([1.0, 2.0])},  # cm
    )

    bd.to_base_units()

    assert bd.units == ureg.m
    np.testing.assert_allclose(bd.signal, [1.0, 2.0])  # 100 cm -> 1 m, 200 cm -> 2 m
    np.testing.assert_allclose(bd.uncertainties["propagate_to_all"], [0.01, 0.02])  # 1 cm -> 0.01 m


def test_to_base_units_noop_if_already_base_units():
    bd = BaseData(
        signal=np.array([1.0, 2.0]),
        units=ureg.m,
        uncertainties={"u": np.array([0.1, 0.2])},
    )

    bd.to_base_units()

    assert bd.units == ureg.m
    np.testing.assert_allclose(bd.signal, [1.0, 2.0])
    np.testing.assert_allclose(bd.uncertainties["u"], [0.1, 0.2])


def test_to_base_units_offset_units_raise():
    bd = BaseData(
        signal=np.array([20.0, 25.0]),
        units=ureg.degF,
        uncertainties={"u": np.array([0.5, 0.5])},
    )

    with pytest.raises(NotImplementedError):
        # the multiplicative_conversion is a bit of a cop-out. Tests whether a unit conversion is purely multiplicative are not straightforward and fast.
        bd.to_base_units(multiplicative_conversion=False)
