import tiled
import tiled.client
from tiled.client.array import DaskArrayClient
import dask.array as da
import pint
from typing import List, Optional, Union, Dict
from attrs import define, field
from attrs import validators as v
import logging
from .processstep import ProcessStepDescriber

logger = logging.getLogger(__name__)


# Custom validator for the rank_of_data field
def validate_rank_of_data(instance, attribute, value):
    # Ensure rank_of_data is between 1 and 3.
    if not (0 <= value <= 3):
        raise ValueError(f"{attribute.name} must be between 0 and 3, got {value}.")
    # Check that rank_of_data does not exceed the number of dimensions in internal_data.
    # This assumes that internal_data is provided and is a valid xarray DataArray.
    if instance.internal_data is not None and value > instance.internal_data.ndim:
        raise ValueError(
            f"{attribute.name} ({value}) cannot exceed the dimensionality of internal_data "
            f"(ndim={instance.internal_data.ndim})."
        )


@define
class BaseData:
    """
    BaseData is a data class that stores the core data array and its associated metadata.
    It is designed to be used as a base class for more specialized data classes.
    """

    # Unit information using Pint units - required input (ingest, internal, and display)
    ingest_units: pint.Unit = field(validator=v.instance_of(pint.Unit))
    internal_units: pint.Unit = field(validator=v.instance_of(pint.Unit))
    display_units: pint.Unit = field(validator=v.instance_of(pint.Unit))

    # Core data array stored as an xarray DataArray
    raw_data: da.Array = field(factory=da.array, validator=[v.instance_of(da.Array)])

    # Dict of variances represented as xarray DataArray objects; defaulting to an empty dict
    variances: Dict[str : da.Array] = field(factory=dict, validator=[v.instance_of(dict)])

    # array with some normalization (exposure time, solid-angle ....)
    normalization: da.Array = field(factory=da.array, validator=[v.instance_of(da.Array)])

    # Provenance can be a list containing either ProcessStep or lists of ProcessStep
    provenance: List[Union[ProcessStepDescriber, List[ProcessStepDescriber]]] = field(factory=list)

    # Rank of the data with custom validation:
    # Must be between 1 and 3 and not exceed the dimensionality of internal_data.
    rank_of_data: int = field(default=1, validator=[v.instance_of(int), validate_rank_of_data])

    # Data source placeholder (e.g., a Tiled instance, such as tiled.client.from_uri("http://localhost:8000", "dask"))
    # data_source: Optional[tiled.client.container.Container] = field(
    #     default=None, 
    #     validator=[v.optional(v.instance_of(tiled.client.container.Container))]
    #     )
    data_source: Optional[DaskArrayClient] = field(
        default=None, 
        validator=[v.optional(v.instance_of(DaskArrayClient))]
        )

    @property
    def mean(self) -> da.Array:
        """
        Returns the raw_data array with the normalization applied.
        The result is cast to internal units.
        """
        return self.raw_data / self.normalization

    def std(self, kind) -> da.Array:
        """
        Returns the uncertainties, i.e. standard deviation
        The result is cast to internal units.
        """
        return da.sqrt(self.variances[kind] / self.normalization)

    def std(self, kind) -> da.Array:
        """
        Returns the uncertainties, i.e. standard deviation
        The result is cast to internal units.
        """
        return da.sqrt(self.variances[kind]) / self.normalization

    @property
    def _unit_scale(self, display_units) -> float:
        return (1 * self.internal_units).to(display_units).magnitude

    @property
    def display_data(self) -> da.Array:
        """
        Returns the internal_data array with the scalar applied and converted
        to display units using Pint's unit conversion.
        """
        return self._unit_scale(self.display_units) * self.raw_data / self.normalization 
