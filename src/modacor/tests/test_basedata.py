import numpy as np
import pint
import pytest

from modacor import ureg

# import tiled.client  # not sure what the class of tiled.client is...
from ..dataclasses.basedata import BaseData  # adjust the import path as needed


@pytest.fixture
def sample_data():
    return np.arange(5)


def test_data_and_display_data_properties(sample_data):
    # Create an instance of BaseData with test values
    bd = BaseData(
        signal_units=ureg.m,  # meters
        signal=sample_data,
        normalization=np.ones_like(sample_data) * 2,
        # uncertainties=[],            # empty list for uncertainties
        # scalar=2.0,
        # scalar_uncertainty=0.1,
        rank_of_data=1,  # valid since sample_data.ndim is 1
    )

    # Test the 'data' property
    expected_data = sample_data / 2.0
    # Here we check that the computed values match the expected ones.
    assert (bd.mean() == expected_data).all(), "data property did not apply the scalar correctly."


def test_rank_validation_exceeds_ndim():
    # Create a 2D DataArray
    arr = np.arange(4).reshape((2, 2))

    # Attempting to set rank_of_data=3 (while arr.ndim is 2) should raise a ValueError.
    with pytest.raises(ValueError) as exc_info:
        BaseData(
            signal_units=ureg.m,
            signal=arr,
            # uncertainties=[],
            # scalar=1.0,
            # scalar_uncertainty=0.0,
            rank_of_data=3,  # invalid, since 3 > arr.ndim (2)
        )
    assert "cannot exceed the dimensionality of signal" in str(exc_info.value)
