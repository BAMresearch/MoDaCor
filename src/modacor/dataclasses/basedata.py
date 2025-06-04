# import tiled
# import tiled.client
import logging
from typing import Dict, List, Self

import numpy as np
import pint
from attrs import define, field
from attrs import validators as v

from modacor import ureg

logger = logging.getLogger(__name__)


def validate_rank_of_data(instance, attribute, value) -> None:
    # Must be between 0 and 3
    if not 0 <= value <= 3:
        raise ValueError(f"{attribute.name} must be between 0 and 3, got {value}.")

    # For array‐like signals, rank cannot exceed ndim
    if instance.signal is not None and value > instance.signal.ndim:
        raise ValueError(
            f"{attribute.name} ({value}) cannot exceed signal dim (ndim={instance.signal.ndim})."
        )


def signal_converter(value: int | float | np.ndarray) -> np.ndarray:
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
    # find out if it can be broadcast at all
    try:
        out_shape = np.broadcast_shapes(signal.shape, arr.shape)
    except ValueError:
        raise ValueError(
            f"'{name}' with shape {arr.shape} cannot broadcast to signal shape {signal.shape}."
        )
    # and find out whether the resulting shape does not change the shape of signal
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
    uncertainties: Dict[str, np.ndarray] = field(validator=v.instance_of(dict))
    # variances: Dict[str, np.ndarray] = field(validator=v.instance_of(dict))
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
    scalar_uncertainty: float = field(default=0.0, converter=float, validator=v.instance_of(float))
    # scalar_variance: float = field(default=0.0, converter=float, validator=v.instance_of(float))

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

    @property
    def scalar_variance(self) -> float:
        """
        Calculate the variance of the scalar.
        If scalar_uncertainty is provided, it is used to calculate the variance.
        Otherwise, it defaults to 0.0.
        """
        return self.scalar_uncertainty**2

    @scalar_variance.setter
    def scalar_variance(self, value: float) -> None:
        """
        Set the scalar variance.
        """
        if not isinstance(value, (int, float, np.ndarray)):
            raise TypeError(f"scalar_variance must be a number, got {type(value)}.")
        if isinstance(value, np.ndarray):
            if value.size != 1:
                raise ValueError("scalar_variance must be a scalar value, got an array.")
            value = value.item()
        self.scalar_uncertainty = value**0.5  # much faster than np.sqrt(value)

    @property
    def variances(self) -> Dict[str, np.ndarray]:
        """
        Get the variances dictionary, calculated from uncertainties.
        """
        return {kind: var**2 for kind, var in self.uncertainties.items()}

    @variances.setter
    def variances(self, value: Dict[str, np.ndarray]) -> None:
        """
        Set the uncertainties dictionary via the variances.
        """
        if not isinstance(value, dict):
            raise TypeError(f"variances must be a dict, got {type(value)}.")
        if not all(isinstance(v, (np.ndarray, int, float)) for v in value.values()):
            raise TypeError("All variances must be int, float or numpy arrays.")
        # validate using validate_broadcast
        for kind, var in value.items():
            if not isinstance(var, np.ndarray):
                var = np.array(var, dtype=float)
            validate_broadcast(self.signal, var, f"variances[{kind}]")
        self.uncertainties.update({kind: var**0.5 for kind, var in value.items()})

    def apply_scalar(self) -> None:
        """
        Apply the internal scalar to the signal and update the scalar and scalar_variance.
        """
        self.signal *= self.scalar
        self.scalar_variance /= self.scalar**2
        self.scalar = 1.0  # normalize by self == 1

    def to_units(self, new_units: pint.Unit) -> None:
        """
        Convert the signal and variances to new units.
        """
        if not isinstance(new_units, ureg.Unit):
            raise TypeError(f"new_units must be a pint.Unit, got {type(new_units)}.")

        if not self.units.is_compatible_with(new_units):
            raise ValueError(
                f"Cannot convert from {self.units} to {new_units}. Units are not compatible."
            )

        # Convert signal
        cfact = new_units.m_from(self.units)
        self.signal *= cfact
        self.units = new_units

        # Convert variances
        for kind in self.uncertainties.keys():
            self.uncertainties[kind] *= cfact
