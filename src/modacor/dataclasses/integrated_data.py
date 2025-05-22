import logging
from typing import Dict  # , List, Self

import numpy as np

# import pint
from attrs import define, field
from attrs import validators as v

from .basedata import BaseData

logger = logging.getLogger(__name__)


@define
class IntegratedData(BaseData):
    r"""
    IntegratedData is a data class that stores the data array from azimuthalIntegration
    and its associated metadata.

    TODO: deal with normalization factor/variance:
    ```
    \text{Var}\left(\frac{A}{B}\right) \approx \left(\frac{\mu_A}{\mu_B}\right)^2
    \left( \frac{\sigma_A^2}{\mu_A^2} + \frac{\sigma_B^2}{\mu_B^2} \right)
    ```
    """

    average: np.ndarray = field(factory=np.ndarray, validator=[v.instance_of(np.ndarray)])
    std: Dict[str, np.ndarray] = field(factory=dict, validator=[v.instance_of(dict)])
    sem: Dict[str, np.ndarray] = field(factory=dict, validator=[v.instance_of(dict)])
    # Core data array stored as an xarray DataArray
    sum_signal: np.ndarray = field(factory=np.ndarray, validator=[v.instance_of(np.ndarray)])

    # Dict of variances represented as xarray DataArray objects; defaulting to an empty dict
    sum_variances: Dict[str, np.ndarray] = field(factory=dict, validator=[v.instance_of(dict)])

    # array with some normalization (exposure time, solid-angle ....)
    sum_normalization: np.ndarray = field(factory=np.ndarray, validator=[v.instance_of(np.ndarray)])
    sum_normalization_squared: np.ndarray = field(factory=np.ndarray, validator=[v.instance_of(np.ndarray)])
