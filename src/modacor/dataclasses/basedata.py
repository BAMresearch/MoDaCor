__all__ = ["BaseData"]

# import tiled
# import tiled.client
import logging
from collections.abc import MutableMapping
from typing import Dict, List, Self

import numpy as np

# from modacor import ureg
import pint
from attrs import define, field, setters
from attrs import validators as v

logger = logging.getLogger(__name__)


# make a varianceDict that quacks like a dict, but is secretly a view on the uncertainties dict
class _VarianceDict(MutableMapping):
    def __init__(self, parent: "BaseData"):
        self._parent = parent

    def __getitem__(self, key):
        # Return variance = (one‐sigma uncertainty)**2
        return (self._parent.uncertainties[key]) ** 2

    def __iter__(self):
        # Iterate over keys in the underlying uncertainties dict
        return iter(self._parent.uncertainties)

    def __len__(self):
        return len(self._parent.uncertainties)

    def __setitem__(self, key, var):
        # Accept scalar or array‐like, coerce to ndarray
        arr = np.asarray(var, dtype=float)
        # Validate broadcast to signal
        validate_broadcast(self._parent.signal, arr, f"variances[{key}]")
        # Store sqrt(var) as the one‐sigma uncertainty
        self._parent.uncertainties[key] = np.asarray(arr**0.5)

    def __delitem__(self, key):
        del self._parent.uncertainties[key]

    def __repr__(self):
        """
        Display exactly like a normal dict of {key: variance_array}, so that
        printing `a.variances` looks familiar.
        """
        # Build a plain Python dict of {key: self[key]} for repr
        d = {k: self[k] for k in self._parent.uncertainties}
        return repr(d)

    def __str__(self):
        # Optional: same as __repr__
        return self.__repr__()


def validate_rank_of_data(instance, attribute, value) -> None:
    # Must be between 0 and 3
    if not 0 <= value <= 3:
        raise ValueError(f"{attribute.name} must be between 0 and 3, got {value}.")

    # For array‐like signals, rank cannot exceed ndim
    if instance.signal is not None and value > instance.signal.ndim:
        raise ValueError(f"{attribute.name} ({value}) cannot exceed signal dim (ndim={instance.signal.ndim}).")


def signal_converter(value: int | float | np.ndarray) -> np.ndarray:
    """
    Convert the input value to a numpy array if it is not already one.
    """
    return np.array(value, dtype=float) if not isinstance(value, np.ndarray) else value


def dict_signal_converter(value: Dict[str, int | float | np.ndarray]) -> Dict[str, np.ndarray]:
    """
    Convert a dictionary of values to a dictionary of numpy arrays.
    Each value in the dictionary is converted to a numpy array if it is not already one.
    """
    return {k: signal_converter(v) for k, v in value.items()}


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
        raise ValueError(f"'{name}' with shape {arr.shape} cannot broadcast to signal shape {signal.shape}.")
    # and find out whether the resulting shape does not change the shape of signal
    if out_shape != signal.shape:
        raise ValueError(f"'{name}' with shape {arr.shape} does not broadcast to signal shape {signal.shape}.")


@define
class BaseData:
    """
    BaseData stores a core data array (`signal`) with associated uncertainties, units,
    and metadata. It validates that any weighting or uncertainty arrays broadcast to
    the shape of `signal`, and provides utilities for scalar operations and unit conversion.

    Attributes
    ----------
    signal : np.ndarray
        The primary data array. All weighting/uncertainty arrays must be broadcastable
        to this shape.
    uncertainties : Dict[str, np.ndarray]
        Uncertainty (as one‐sigma standard deviation) arrays keyed by type (e.g., “poisson”,
        “SEM”). Each array must broadcast to `signal.shape`. Variances are computed as
        `uncertainties[k]**2`.
        Uncertainty propagation in operations that combine BaseData elements will try to
        apply the incoming uncertainties by matched key,
        If only a single uncertainty is found (ideally named 'propagate_to_all'), it will be
        applied to all uncertainties.
    units : pint.Unit
        Physical units of `signal*scalar` and their uncertainties.
    weighting : np.ndarray, optional
        Weights for `signal` (default is a scalar 1.0) for use in averaging operations.
        Must broadcast to `signal.shape`.
    scalar : float, default=1.0
        Multiplicative factor for `signal`. Calling `apply_scalar()` multiplies `signal`
        by this and adjusts its variance/uncertainty accordingly.
    scalar_uncertainty : float, default=0.0
        One‐sigma uncertainty (std) on `scalar`, used in `scalar_variance`.
    axes : List[BaseData | None]
        Optional metadata for each axis of `signal`. Defaults to an empty list.
    rank_of_data : int, default=0
        Rank (0–3) of the data; 1 is line data, 2 is image data. Must not exceed
        `signal.ndim`.

    Properties
    ----------
    scalar_variance : float
        Returns `scalar_uncertainty**2`. Setting this expects a numeric (or size‐1 array)
        and stores `scalar_uncertainty = sqrt(value)`.
    variances : Dict[str, np.ndarray]
        Returns `{k: u**2 for k, u in uncertainties.items()}`. Assigning expects a dict
        of variance arrays; each is validated against `signal.shape` and
        converted into `uncertainties[k] = sqrt(var)`.

    Methods
    -------
    apply_scalar():
        Multiplies `signal` by `scalar`, normalizes `scalar_uncertainty` by `scalar`,
        and finally resets `scalar` to 1.0.
    to_units(new_units: pint.Unit):
        Converts internal `signal` and all `uncertainties` to `new_units` if compatible with
        the existing `units`. Raises `TypeError` or `ValueError` on invalid input.
    """

    # required:
    # Core data array stored as an xarray DataArray
    signal: np.ndarray = field(
        converter=signal_converter, validator=v.instance_of(np.ndarray), on_setattr=setters.validate
    )
    # Dict of variances represented as xarray DataArray objects; defaulting to an empty dict
    uncertainties: Dict[str, np.ndarray] = field(
        converter=dict_signal_converter, validator=v.instance_of(dict), on_setattr=setters.validate
    )
    # variances: Dict[str, np.ndarray] = field(validator=v.instance_of(dict))
    # Unit of signal*scalar - required input 'dimensionless' for dimensionless data
    units: pint.Unit = field(validator=v.instance_of(pint.Unit), on_setattr=setters.validate)  # type: ignore
    # optional:
    # weights for the signal, can be used for weighted averaging
    weighting: np.ndarray = field(
        default=np.array(1.0),
        converter=signal_converter,
        validator=v.instance_of(np.ndarray),
        on_setattr=setters.validate,
    )
    # scalar for the signal, should be applied before certain operations to signal,
    # at which point signal_variance is normalized to scalar^2, and scalar to scalar (=1)
    scalar: float = field(default=1.0, converter=float, validator=v.instance_of(float), on_setattr=setters.validate)
    scalar_uncertainty: float = field(
        default=0.0, converter=float, validator=v.instance_of(float), on_setattr=setters.validate
    )

    # metadata
    axes: List[Self | None] = field(factory=list, validator=v.instance_of(list), on_setattr=setters.validate)
    # Rank of the data with custom validation:
    # Must be between 0 and 3 and not exceed the dimensionality of signal.
    rank_of_data: int = field(
        default=0, converter=int, validator=[v.instance_of(int), validate_rank_of_data], on_setattr=setters.validate
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
    def variances(self) -> _VarianceDict:
        """
        A dict‐like view of variances.
        • Reading:    bd.variances['foo']  → returns uncertainties['foo']**2
        • Writing:    bd.variances['foo'] = var_array  → validates + sets uncertainties['foo']=sqrt(var_array)
        • Deleting:   del bd.variances['foo']  → removes 'foo' from uncertainties
        """
        return _VarianceDict(self)

    @variances.setter
    def variances(self, value: Dict[str, np.ndarray]) -> None:
        """
        Set the uncertainties dictionary via the variances.
        """
        if not isinstance(value, dict):
            raise TypeError(f"variances must be a dict, got {type(value)}.")
        if not all(isinstance(v, (np.ndarray, int, float)) for v in value.values()):
            raise TypeError("All variances must be int, float or numpy arrays.")
        # (Optionally clear existing uncertainties, or merge—here we'll just overwrite keys:)
        self.uncertainties.clear()  # clear existing uncertainties
        for kind, var in value.items():
            arr = np.asarray(var, dtype=float)
            validate_broadcast(self.signal, arr, f"variances[{kind}]")
            self.uncertainties[kind] = arr**0.5

        # validate using validate_broadcast
        # for kind, var in value.items():
        #     if not isinstance(var, np.ndarray):
        #         var = np.array(var, dtype=float)
        #     validate_broadcast(self.signal, var, f"variances[{kind}]")
        # self.uncertainties.update({kind: var**0.5 for kind, var in value.items()})

    def apply_scalar(self) -> None:
        """
        Apply the internal scalar to the signal and update the scalar and scalar_variance.
        """
        self.signal *= self.scalar
        self.scalar_uncertainty /= self.scalar
        self.scalar = 1.0  # normalize by self == 1

    # move to a separate processing module for separation of concerns:
    # def to_units(self, new_units: pint.Unit) -> None:
    #     """
    #     Convert the signal and variances to new units.
    #     """
    #     if not isinstance(new_units, ureg.Unit):
    #         raise TypeError(f"new_units must be a pint.Unit, got {type(new_units)}.")

    #     if not self.units.is_compatible_with(new_units):
    #         raise ValueError(f"""
    #           Cannot convert from {self.units} to {new_units}. Units are not compatible.
    #         """)

    #     # Convert signal
    #     cfact = new_units.m_from(self.units)
    #     self.scalar *= cfact
    #     self.units = new_units
    #     # Convert uncertainty
    #     self.scalar_uncertainty *= cfact
