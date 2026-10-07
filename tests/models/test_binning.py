from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from modacor.models.binning import assign_bin_indices, generate_bin_edges, validate_bin_edges


def test_assign_bin_indices_uses_left_inclusive_bins_and_includes_final_edge() -> None:
    values = np.array([-1.0, 0.0, 0.5, 1.0, 2.0, 2.1, np.nan])

    result = assign_bin_indices(values, np.array([0.0, 1.0, 2.0]))

    assert_array_equal(result, [-1, 0, 0, 1, 1, -1, -1])


def test_generate_bin_edges_infers_ranges() -> None:
    values = np.array([-2.0, 1.0, 4.0, np.nan])

    assert_allclose(
        generate_bin_edges(values, lower=None, upper=None, n_bins=3, spacing="linear"),
        [-2.0, 0.0, 2.0, 4.0],
    )
    assert_allclose(
        generate_bin_edges(values, lower=None, upper=4.0, n_bins=2, spacing="log"),
        [1.0, 2.0, 4.0],
    )


@pytest.mark.parametrize(
    ("edges", "message"),
    [
        ([0.0], "At least two"),
        ([0.0, 0.0], "strictly increasing"),
        ([0.0, np.nan], "finite"),
        ([[0.0, 1.0]], "one-dimensional"),
    ],
)
def test_validate_bin_edges_rejects_invalid_edges(edges, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        validate_bin_edges(np.asarray(edges))


def test_generate_log_edges_requires_positive_range() -> None:
    with pytest.raises(ValueError, match="positive"):
        generate_bin_edges(np.array([-2.0, -1.0]), lower=None, upper=None, n_bins=2, spacing="log")
