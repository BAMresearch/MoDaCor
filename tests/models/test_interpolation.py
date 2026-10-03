from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from modacor.models.interpolation import remap_coefficients_1d


def test_nearest_uses_lower_coordinate_for_ties_and_never_extrapolates():
    left, right, left_weight, right_weight, valid = remap_coefficients_1d(
        np.array([0.0, 2.0, 4.0]),
        np.array([-1.0, 0.0, 1.0, 3.0, 4.0, 5.0]),
        mode="nearest",
    )

    assert_array_equal(left, [0, 0, 0, 1, 2, 2])
    assert_array_equal(right, left)
    assert_allclose(left_weight, 1.0)
    assert_allclose(right_weight, 0.0)
    assert_array_equal(valid, [False, True, True, True, True, False])


def test_linear_coefficients_and_maximum_bracket_width():
    left, right, left_weight, right_weight, valid = remap_coefficients_1d(
        np.array([0.0, 1.0, 3.0]),
        np.array([0.5, 2.0, 3.0]),
        mode="linear",
        max_gap=1.5,
    )

    assert_array_equal(left, [0, 1, 1])
    assert_array_equal(right, [1, 2, 2])
    assert_allclose(left_weight, [0.5, 0.5, 0.0])
    assert_allclose(right_weight, [0.5, 0.5, 1.0])
    assert_array_equal(valid, [True, False, True])


def test_remap_rejects_non_unique_source_axis():
    with pytest.raises(ValueError, match="strictly increasing"):
        remap_coefficients_1d(np.array([0.0, 1.0, 1.0]), np.array([0.5]), mode="nearest")


def test_nearest_rejects_tolerance_style_max_gap():
    with pytest.raises(ValueError, match="only supported for linear"):
        remap_coefficients_1d(
            np.array([0.0, 1.0]),
            np.array([0.5]),
            mode="nearest",
            max_gap=0.1,
        )
