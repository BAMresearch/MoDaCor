# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest

from modacor import ureg
from modacor.dataclasses.basedata import BaseData
from modacor.modules.helpers import scalar_in_units


@pytest.mark.parametrize(
    "value,units,target", [(2, "mm", "m"), (90.0, "degree", "radian"), (3.0, "1/mm", "1/m"), (0.5, None, "count")]
)
def test_configuration_conversion_matches_old_pint_formula(value, units, target):
    expected = ureg.Quantity(value, units or target).to(target).magnitude
    assert scalar_in_units(value, target, name="configuration", units=units) == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize("shape", [(), (1,), (2, 3)])
def test_basedata_scalar_or_exactly_constant_conversion_preserves_input(shape):
    data = BaseData(np.full(shape, 2.0), ureg.mm, uncertainties={"measured": np.full(shape, 0.1)})
    assert scalar_in_units(data, "m", name="length") == pytest.approx(0.002, rel=1e-12)
    np.testing.assert_array_equal(data.signal, np.full(shape, 2.0))
    np.testing.assert_array_equal(data.uncertainties["measured"], np.full(shape, 0.1))
    assert data.units == ureg.mm


@pytest.mark.parametrize(
    "value,units,target",
    [
        (BaseData(np.array([1.0, 2.0]), ureg.m), None, "m"),
        (BaseData(np.asarray(1.0), ureg.m), "mm", "m"),
        (1.0, "second", "m"),
    ],
)
def test_conversion_errors_include_configuration_name(value, units, target):
    with pytest.raises(ValueError, match="calibration"):
        scalar_in_units(value, target, name="calibration", units=units)
