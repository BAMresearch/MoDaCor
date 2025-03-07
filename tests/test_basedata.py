import pytest
import xarray as xr
import pint
import tiled.client  # not sure what the class of tiled.client is...
from modacor.dataclasses.basedata import BaseData  # adjust the import path as needed
from modacor.dataclasses.processstep import ProcessStep  # adjust the import path as needed

# Create a dummy client to use as the data_source
class DummyClient:
    pass

@pytest.fixture
def ureg():
    return pint.UnitRegistry()

@pytest.fixture
def sample_data():
    # Create a simple 1D xarray DataArray
    return xr.DataArray([1, 2, 3])

def test_data_and_display_data_properties(ureg, sample_data):
    # Create an instance of BaseData with test values
    bd = BaseData(
        ingest_units=ureg.m,         # meters
        internal_units=ureg.m,       # meters
        display_units=ureg.cm,       # centimeters
        data_source=DummyClient(),
        internal_data=sample_data,
        uncertainties=[],            # empty list for uncertainties
        scalar=2.0,
        scalar_uncertainty=0.1,
        provenance=[],               # empty provenance
        rank_of_data=1               # valid since sample_data.ndim is 1
    )
    
    # Test the 'data' property
    expected_data = sample_data * 2.0
    # Here we check that the computed values match the expected ones.
    assert (bd.data == expected_data).all(), "data property did not apply the scalar correctly."
    
    # Test the 'display_data' property
    # The conversion from meters to centimeters yields a factor of 100.
    conversion_factor = (1 * ureg.Unit("m")).to("cm").magnitude  # should be 100.0
    expected_display = sample_data * 2.0 * conversion_factor
    assert (bd.display_data == expected_display).all(), "display_data property did not convert units correctly."

def test_rank_validation_exceeds_ndim(ureg):
    # Create a 2D DataArray
    arr = xr.DataArray([[1, 2], [3, 4]])
    
    # Attempting to set rank_of_data=3 (while arr.ndim is 2) should raise a ValueError.
    with pytest.raises(ValueError) as exc_info:
        BaseData(
            ingest_units=ureg.m,
            internal_units=ureg.m,
            display_units=ureg.cm,
            data_source=DummyClient(),
            internal_data=arr,
            uncertainties=[],
            scalar=1.0,
            scalar_uncertainty=0.0,
            provenance=[],
            rank_of_data=3  # invalid, since 3 > arr.ndim (2)
        )
    assert "cannot exceed the dimensionality of internal_data" in str(exc_info.value)