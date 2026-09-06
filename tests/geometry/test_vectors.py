# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

import numpy as np
import pytest

from modacor.geometry import unit_vector3


def test_unit_vector3_normalizes_finite_vector():
    np.testing.assert_allclose(unit_vector3([3.0, 0.0, 4.0]), [0.6, 0.0, 0.8])


@pytest.mark.parametrize("value", [[0.0, 0.0, 0.0], [1.0, 2.0], [1.0, np.nan, 2.0]])
def test_unit_vector3_rejects_invalid_vector(value):
    with pytest.raises(ValueError):
        unit_vector3(value)
