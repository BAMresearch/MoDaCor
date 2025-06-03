# import tiled
# import tiled.client
import logging
from typing import Any, Dict, List, Optional, Self

import numpy as np
import pint
from attrs import define, field
from attrs import validators as v

from modacor import ureg

logger = logging.getLogger(__name__)


def validate_rank_of_data(instance, attribute, value):
    # Must be between 0 and 3
    if not 0 <= value <= 3:
        raise ValueError(f"{attribute.name} must be between 0 and 3, got {value}.")

    # For array‐like signals, rank cannot exceed ndim
    if instance.signal is not None and value > instance.signal.ndim:
        raise ValueError(
            f"{attribute.name} ({value}) cannot exceed signal dim (ndim={instance.signal.ndim})."
        )


def signal_converter(value):
    """
    Convert the input value to a numpy array if it is not already one.
    """
    return np.array(value, dtype=float) if not isinstance(value, np.ndarray) else value


def validate_broadcast(signal: np.ndarray, arr: np.ndarray, name: str) -> None:
    """
    Raise if `arr` cannot broadcast to `signal.shape`.
    Scalars (size=1) and None are always accepted.
    """
    if arr.size == 1:
        return  # compatible with any shape

    try:
        out_shape = np.broadcast_shapes(signal.shape, arr.shape)
    except ValueError:
        raise ValueError(
            f"'{name}' with shape {arr.shape} cannot broadcast to signal shape {signal.shape}."
        )

    if out_shape != signal.shape:
        raise ValueError(
            f"'{name}' with shape {arr.shape} does not broadcast to signal shape {signal.shape}."
        )


@define
class BaseData:
    """
    BaseData is a data class that stores a data array and its associated metadata.
    """

    # required:
    # Core data array stored as an xarray DataArray
    signal: np.ndarray = field(converter=signal_converter, validator=v.instance_of(np.ndarray))
    # Dict of variances represented as xarray DataArray objects; defaulting to an empty dict
    variances: Dict[str, np.ndarray] = field(validator=v.instance_of(dict))
    # Unit of signal*scalar - required input 'dimensionless' for dimensionless data
    units: ureg.Unit = field(validator=v.instance_of(ureg.Unit))  # type: ignore
    # optional:
    # weights for the signal, can be used for weighted averaging
    weighting: np.ndarray = field(
        default=np.array(1.0),
        converter=signal_converter,
        validator=v.instance_of(np.ndarray),
    )
    # scalar for the signal, should be applied before certain operations to signal,
    # at which point signal_variance is normalized to scalar^2, and scalar to scalar (=1)
    scalar: float = field(default=1.0, converter=float, validator=v.instance_of(float))
    scalar_variance: float = field(default=0.0, converter=float, validator=v.instance_of(float))

    # metadata
    axes: List[Self | None] = field(factory=list, validator=v.instance_of(list))
    # Rank of the data with custom validation:
    # Must be between 0 and 3 and not exceed the dimensionality of signal.
    rank_of_data: int = field(
        default=0, converter=int, validator=[v.instance_of(int), validate_rank_of_data]
    )

    def __attrs_post_init__(self):
        """
        Post-initialization to ensure that the shapes of elements in variances dict,
        and the shapes of weighting are compatible with the signal array.
        """
        # Validate variances
        for kind, var in self.variances.items():
            validate_broadcast(self.signal, var, f"variances[{kind}]")

        # Validate weighting
        validate_broadcast(self.signal, self.weighting, "weighting")

    def apply_scalar(self) -> None:
        """
        Apply the internal scalar to the signal and update the scalar and scalar_variance.
        """
        self.signal *= self.scalar
        self.scalar_variance /= self.scalar**2
        self.scalar = 1.0  # normalize by self == 1
