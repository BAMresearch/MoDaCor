# SPDX-License-Identifier: BSD-3-Clause

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


def test_basedata_requires_uncertainty_aware_accessor():
    with pytest.raises(TypeError, match="length.*configuration number"):
        scalar_in_units(BaseData(2.0, ureg.mm), "m", name="length")


@pytest.mark.parametrize(
    "value,units,target",
    [
        (1.0, "second", "m"),
    ],
)
def test_conversion_errors_include_configuration_name(value, units, target):
    with pytest.raises(ValueError, match="calibration"):
        scalar_in_units(value, target, name="calibration", units=units)
