# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

import numpy as np
import pytest

from modacor.geometry import identity_matrix4, rotation_matrix4, translation_matrix4


def test_identity_and_translation_matrices_transform_points():
    point = np.array([1.0, 2.0, 3.0, 1.0])
    np.testing.assert_array_equal(identity_matrix4() @ point, point)
    np.testing.assert_allclose(
        translation_matrix4([4.0, -2.0, 0.5]) @ point,
        [5.0, 0.0, 3.5, 1.0],
    )


def test_rotation_matrix_normalizes_axis_and_uses_right_hand_rule():
    result = rotation_matrix4([0.0, 0.0, 2.0], np.pi / 2.0) @ np.array([1.0, 0.0, 0.0, 1.0])
    np.testing.assert_allclose(result, [0.0, 1.0, 0.0, 1.0], atol=1e-15)


@pytest.mark.parametrize(
    ("function", "argument"),
    [
        (translation_matrix4, [1.0, 2.0]),
        (translation_matrix4, [1.0, np.nan, 3.0]),
        (lambda value: rotation_matrix4(value, 0.0), [0.0, 0.0, 0.0]),
        (lambda value: rotation_matrix4([1.0, 0.0, 0.0], value), np.nan),
    ],
)
def test_transform_matrices_reject_invalid_inputs(function, argument):
    with pytest.raises(ValueError):
        function(argument)
