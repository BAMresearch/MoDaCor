import tiled
import tiled.client
import xarray as xr
import pint
from typing import List, Union
from attrs import define, field
from attrs import validators as v
import logging
from .processstep import ProcessStep

logger = logging.getLogger(__name__)

# Custom validator for the rank_of_data field
def validate_rank_of_data(instance, attribute, value):
    # Ensure rank_of_data is between 1 and 3.
    if not (1 <= value <= 3):
        raise ValueError(f"{attribute.name} must be between 1 and 3, got {value}.")
    # Check that rank_of_data does not exceed the number of dimensions in internal_data.
    # This assumes that internal_data is provided and is a valid xarray DataArray.
    if instance.internal_data is not None and value > instance.internal_data.ndim:
        raise ValueError(
            f"{attribute.name} ({value}) cannot exceed the dimensionality of internal_data "
            f"(ndim={instance.internal_data.ndim})."
        )

@define
class BaseData:
    # Unit information using Pint units - required input (ingest, internal, and display)
    ingest_units: pint.Unit = field(validator=v.instance_of(pint.Unit))
    internal_units: pint.Unit = field(validator=v.instance_of(pint.Unit))
    display_units: pint.Unit = field(validator=v.instance_of(pint.Unit))
    # Data source placeholder (e.g., a Tiled instance)
    data_source: tiled.client = field()

    # Core data array stored as an xarray DataArray
    internal_data: xr.DataArray = field(factory=xr.DataArray, validator=[v.instance_of(xr.DataArray)])
    
    # List of uncertainties represented as xarray DataArray objects; defaulting to an empty list
    uncertainties: List[xr.DataArray] = field(factory=list, validator=[v.instance_of(list)])
    
    # Scalar multiplier and its uncertainty
    scalar: float = field(default=1.0, validator=v.instance_of(float))
    scalar_uncertainty: float = field(default=0.0, validator=v.instance_of(float))
    
    
    # Provenance can be a list containing either ProcessStep instances or lists of ProcessStep instances
    provenance: List[Union[ProcessStep, List[ProcessStep]]] = field(factory=list)
    
    # Rank of the data with custom validation:
    # Must be between 1 and 3 and not exceed the dimensionality of internal_data.
    rank_of_data: int = field(
        default=1,
        validator=[v.instance_of(int), validate_rank_of_data]
    )

    @property
    def data(self) -> xr.DataArray:
        """
        Returns the internal_data array with the scalar applied.
        The result is cast to internal units.
        """
        as_quantity = False
        # Here, scalar multiplication is applied element-wise.
        if not as_quantity:
            return self.internal_data * self.scalar
        else:
            return Q_(self.internal_data * self.scalar, self.internal_units)

    @property
    def display_data(self) -> xr.DataArray:
        """
        Returns the internal_data array with the scalar applied and converted
        to display units using Pint's unit conversion.
        """
        as_quantity = False
        # calculate the conversion factor:
        if not as_quantity:
            return self.internal_data * self.scalar * (1 * self.internal_units).to(self.display_units).magnitude
        else:
            return Q_(self.internal_data * self.scalar, self.internal_units).to(self.display_units)
