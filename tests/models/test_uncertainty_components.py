# SPDX-License-Identifier: BSD-3-Clause
"""Independent-component attachment follows the same table as BaseData arithmetic."""

import numpy as np
import pytest

from modacor.dataclasses.basedata import BaseData, _uncertainty_key_plan
from modacor.models.uncertainty import (
    combine_uncertainty_component,
    combine_uncertainty_components,
    uncertainty_key_plan,
)

MAPPINGS = [
    {},
    {"propagate_to_all": np.array([3.0, 4.0])},
    {"shared": np.array([3.0, 4.0]), "left": np.array([5.0, 6.0])},
]


@pytest.mark.parametrize("left", MAPPINGS)
@pytest.mark.parametrize("right", MAPPINGS + [{"right": np.array([1.0, 2.0]), "shared": np.array([2.0, 3.0])}])
def test_mapping_combination_matches_independent_addition(left, right):
    result = combine_uncertainty_components(left, right)
    expected = (
        BaseData(np.zeros(2), "", uncertainties=left) + BaseData(np.zeros(2), "", uncertainties=right)
    ).uncertainties
    assert _uncertainty_key_plan is uncertainty_key_plan
    assert result.keys() == expected.keys()
    for key in expected:
        np.testing.assert_allclose(result[key], expected[key], rtol=1e-12, atol=0)
        for original in [*left.values(), *right.values()]:
            assert not np.shares_memory(result[key], original)


@pytest.mark.parametrize("reverse", [False, True])
def test_single_component_global_named_combination(reverse):
    target = {"named": np.array(3.0)} if reverse else {"propagate_to_all": np.array(3.0)}
    combine_uncertainty_component(target, "propagate_to_all" if reverse else "named", np.array(4.0))
    assert target == {"named": np.array(5.0)}


@pytest.mark.parametrize("reverse", [False, True])
def test_invalid_mapping_does_not_mutate_inputs(reverse):
    valid = {"estimate": np.array(2.0)}
    invalid = {"propagate_to_all": np.array(3.0), "named": np.array(4.0)}
    left, right = (invalid, valid) if reverse else (valid, invalid)
    before = {key: value.copy() for key, value in left.items()}
    with pytest.raises(ValueError, match="propagate_to_all"):
        combine_uncertainty_components(left, right)
    for key in before:
        np.testing.assert_array_equal(left[key], before[key])
