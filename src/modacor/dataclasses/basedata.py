# import tiled
# import tiled.client
import logging
from typing import Dict, List, Self

import numpy as np
import pint
from attrs import define, field
from attrs import validators as v

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
    normalization_units: pint.Unit = field(validator=v.instance_of(pint.Unit))
    normalization_factor_unit: pint.Unit = field(validator=v.instance_of(pint.Unit))
    display_units: pint.Unit = field(validator=v.instance_of(pint.Unit))

    # Core data array stored as an xarray DataArray
    raw_data: np.ndarray = field(factory=np.ndarray, validator=[v.instance_of(np.ndarray)])

    # Dict of variances represented as xarray DataArray objects; defaulting to an empty dict
    variances: Dict[str, np.ndarray] = field(factory=dict, validator=[v.instance_of(dict)])

    # array with some normalization (exposure time, solid-angle ....)
    normalization: np.ndarray = field(factory=np.ndarray, validator=[v.instance_of(np.ndarray)])

    # Scalers to put on the denominator, sparated from the array for distinct uncertainty
    normalization_factor: float = field(
        factory=float, validator=[v.instance_of(float), validate_rank_of_data]
    )

    normalization_factor_variance: float = field(
        factory=float, validator=[v.instance_of(float), validate_rank_of_data]
    )

    # Provenance can be a list containing either ProcessStep or lists of ProcessStep
    provenance: List = field(factory=list)

    axes: List[Self | None] = field(factory=list, validator=[v.instance_of(list)])

    # Rank of the data with custom validation:
    # Must be between 0 and 3 and not exceed the dimensionality of internal_data.
    rank_of_data: int = field(factory=int, validator=[v.instance_of(int), validate_rank_of_data])

    @property
    def mean(self) -> np.ndarray:
        """
        Returns the raw_data array with the normalization applied.
        The result is cast to internal units.
        """
        return self.raw_data / self.normalization

    def std(self, kind) -> np.ndarray:
        """
        Returns the uncertainties, i.e. standard deviation
        The result is cast to internal units.
        """
        return np.sqrt(self.variances[kind] / self.normalization)

    def sem(self, kind) -> np.ndarray:
        """
        Returns the uncertainties, i.e. standard deviation
        The result is cast to internal units.
        """
        return np.sqrt(self.variances[kind]) / self.normalization

    @property
    def _unit_scale(self, display_units) -> float:
        return (1 * self.internal_units).to(display_units).magnitude

    @property
    def display_data(self) -> np.ndarray:
        """
        Returns the internal_data array with the scalar applied and converted
        to display units using Pint's unit conversion.
        """
        return self._unit_scale(self.display_units) * self.raw_data / self.normalization

    @property
    def mask(self) -> np.ndarray:
        """calculate the mask for the array"""
        return self.normalization == 0

    @mask.setter
    def mask(self, value):
        """Apply a mask to the data"""
        idx = np.where(value)
        self.raw_data[idx] = 0
        self.normalization[idx] = 0
        for var in self.variances.values():
            var[idx] = 0

    def add_poisson_noise(self):
        self.varinces["poisson"] = np.random.poisson(self.raw_data)
