# SPDX-License-Identifier: BSD-3-Clause
# /usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

__coding__ = "utf-8"
__authors__ = ["Brian R. Pauw, Jérôme Kieffer"]  # add names to the list as appropriate
__copyright__ = "Copyright 2025, The MoDaCor team"
__date__ = "16/11/2025"
__status__ = "Development"  # "Development", "Production"
# end of header and standard imports

__all__ = ["BaseData"]

# import tiled
# import tiled.client
import logging
import numbers
import operator
from collections.abc import MutableMapping
from typing import Any, Callable, Dict, List, Self

import numpy as np
import pint
from attrs import define, field, setters
from attrs import validators as v

from modacor import ureg

# trial uncertainties handling via auto_uncertainties. This seems much more performant for arrays than the built-in uncertainties package
# from auto_uncertainties import Uncertainty

logger = logging.getLogger(__name__)


# make a varianceDict that quacks like a dict, but is secretly a view on the uncertainties dict
class _VarianceDict(MutableMapping, dict):
    def __init__(self, parent: BaseData):
        self._parent = parent

    def __getitem__(self, key):
        # Return variance = (one‐sigma uncertainty)**2
        return (self._parent.uncertainties[key]) ** 2

    def __iter__(self):
        # Iterate over keys in the underlying uncertainties dict
        return iter(self._parent.uncertainties)

    def __len__(self):
        return len(self._parent.uncertainties)

    def keys(self):
        # Return keys of the underlying uncertainties dict
        return self._parent.uncertainties.keys()

    def items(self):
        # Return items as (key, variance) pairs
        tmp = {key: self[key] for key in self._parent.uncertainties}
        return tmp.items()

    def __contains__(self, x) -> bool:
        return x in self._parent.uncertainties

    def __setitem__(self, key, var):
        # Accept scaling or array‐like, coerce to ndarray
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


def _copy_uncertainties(unc_dict: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Deep-copy the uncertainties dict (to avoid aliasing between objects)."""
    return {k: np.array(v, copy=True) for k, v in unc_dict.items()}


def _inherit_metadata(source: BaseData, result: BaseData) -> BaseData:
    """
    Copy metadata-like attributes from `source` to `result` (axes, rank_of_data, weights)
    without touching numerical content (signal, units, uncertainties).
    """
    # Shallow-copy axes list to avoid aliasing of the list object itself
    result.axes = list(source.axes)

    # Keep the same rank_of_data; attrs validation ensures it is still valid
    result.rank_of_data = source.rank_of_data

    # Try to propagate weights; if shapes are incompatible, keep defaults
    try:
        arr = np.asarray(source.weights)
        validate_broadcast(result.signal, arr, "weights")
        result.weights = np.broadcast_to(arr, result.signal.shape).copy()
    except ValueError:
        logger.debug("Could not broadcast source weights to result shape; leaving default weights on result BaseData.")

    return result


def _binary_basedata_op(
    left: BaseData,
    right: BaseData,
    op: Callable[[Any, Any], Any],
) -> BaseData:
    """
    Apply a binary arithmetic operation to two BaseData objects, propagating
    uncertainties using standard first-order, uncorrelated error propagation.

    Rules:
    - result = op(A, B), where A = left.signal * left.units, B = right.signal * right.units
    - For each uncertainty key present on `left`:
        * If `right` has the same key, use that.
        * Else, if `right` has 'propagate_to_all', use that.
        * Else, result.uncertainties[key] = NaN.
    - Binary formulas:
        * Addition / subtraction:
            σ_result^2 = σ_A^2 + σ_B^2  (σ_A, σ_B in the same units as the result)
        * Multiplication / division:
            σ_result / |result| = sqrt( (σ_A / A)^2 + (σ_B / B)^2 )
    """
    # Let pint handle dimensional checks and broadcasting for the nominal values
    A_q = left.signal * left.units
    B_q = right.signal * right.units
    base_result = op(A_q, B_q)  # pint.Quantity
    base_signal = np.asarray(base_result.magnitude)
    result_units = base_result.units

    # Pre-allocate result uncertainties for all keys present on left
    result = BaseData(
        signal=base_signal,
        units=result_units,
        uncertainties={key: np.full(base_signal.shape, np.nan, dtype=float) for key in left.uncertainties.keys()},
    )

    propagate_all = right.uncertainties.get("propagate_to_all", None)

    # Broadcast signals to the result shape (pure magnitudes)
    A_val = np.broadcast_to(np.asarray(left.signal, dtype=float), base_signal.shape)
    B_val = np.broadcast_to(np.asarray(right.signal, dtype=float), base_signal.shape)

    # For multiplication / division, we need |result| as magnitudes for relative errors
    R_val = np.asarray(base_signal, dtype=float)

    for key in result.uncertainties.keys():
        left_err = left.uncertainties.get(key)
        if left_err is None:
            # no uncertainty on left for this key
            continue

        right_err = right.uncertainties.get(key, propagate_all)
        if right_err is None:
            # neither matching key nor propagate_to_all on right → leave NaN
            continue

        # Turn uncertainties into broadcastable float arrays
        left_err_arr = np.asarray(left_err, dtype=float)
        right_err_arr = np.asarray(right_err, dtype=float)

        left_err_b = np.broadcast_to(left_err_arr, base_signal.shape)
        right_err_b = np.broadcast_to(right_err_arr, base_signal.shape)

        if op in (operator.add, operator.sub):
            # Addition / subtraction: σ_result^2 = σ_A^2 + σ_B^2
            # Make sure σ_A and σ_B are in the same units as the result
            sigma_A_q = (left_err_b * left.units).to(result_units)
            sigma_B_q = (right_err_b * right.units).to(result_units)
            sigma = np.sqrt(sigma_A_q.magnitude**2 + sigma_B_q.magnitude**2)
        elif op in (operator.mul, operator.truediv):
            # Multiplication / division:
            #   σ_result / |result| = sqrt((σ_A / A)^2 + (σ_B / B)^2)
            # with A, B and their σ in consistent units → ratio is dimensionless.
            with np.errstate(divide="ignore", invalid="ignore"):
                rel_A = np.where(A_val == 0.0, 0.0, left_err_b / A_val)
                rel_B = np.where(B_val == 0.0, 0.0, right_err_b / B_val)

            rel_R = np.sqrt(rel_A**2 + rel_B**2)
            sigma = rel_R * np.abs(R_val)
        else:
            raise NotImplementedError(f"Operation {op} not supported in _binary_basedata_op")  # noqa: E713

        result.uncertainties[key] = sigma

    # Inherit metadata (axes, rank_of_data, weights) from the left operand
    return _inherit_metadata(left, result)


def _unary_basedata_op(
    element: BaseData,
    func: Callable[[np.ndarray], np.ndarray],
    dfunc: Callable[[np.ndarray], np.ndarray],
    out_units: pint.Unit,
    domain: Callable[[np.ndarray], np.ndarray] | None = None,
) -> BaseData:
    """
    Generic unary op: y = func(x), σ_y ≈ |dfunc(x)| σ_x, with an optional domain.

    Outside the domain, signal and uncertainties are set to NaN (using np.where-style masking).
    """
    x = np.asarray(element.signal)
    if domain is None:
        valid = np.ones_like(x, dtype=bool)
    else:
        valid = domain(x)

    y = np.full_like(x, np.nan, dtype=float)
    y[valid] = func(x[valid])

    deriv = np.zeros_like(x, dtype=float)
    deriv[valid] = np.abs(dfunc(x[valid]))

    result = BaseData(
        signal=y,
        units=out_units,
        uncertainties={},
    )

    for key, err in element.uncertainties.items():
        validate_broadcast(x, np.asarray(err), f"uncertainties['{key}']")
        err_b = np.broadcast_to(err, x.shape)

        sigma_y = np.full_like(x, np.nan, dtype=float)
        sigma_y[valid] = deriv[valid] * np.abs(err_b[valid])
        result.uncertainties[key] = sigma_y

    # Preserve metadata from the original element
    return _inherit_metadata(element, result)


class UncertaintyOpsMixin:
    """
    Mixin that adds arithmetic with uncertainty propagation to BaseData.

    Assumptions
    -----------
    - `signal` is a numpy array of nominal values.
    - `uncertainties` maps keys -> absolute 1σ uncertainty arrays (broadcastable to signal).
    - Binary operations assume uncorrelated uncertainties between operands.
    - Unary operations use first-order linear error propagation.
    """

    # ---- binary dunder ops ----

    def _binary_op(
        self: BaseData,
        other: Any,
        op: Callable[[Any, Any], Any],
        swapped: bool = False,
    ) -> BaseData:
        if not isinstance(self, BaseData):
            return NotImplemented

        # --- numbers.Real: treat as dimensionless for * and /, same units for + and -
        if isinstance(other, numbers.Real):
            scalar = float(other)
            if op in (operator.mul, operator.truediv):
                # dimensionless scalar
                scalar_units = ureg.dimensionless
            else:
                # additive scalar, interpret as self.units
                scalar_units = self.units

            signal = np.full_like(self.signal, scalar, dtype=float)
            other = BaseData(
                signal=signal,
                units=scalar_units,
                uncertainties={k: np.zeros_like(v, dtype=float) for k, v in self.uncertainties.items()},
            )

        # --- pint.Quantity: for +/-, use same units as self; for */÷, keep its own units
        elif isinstance(other, pint.Quantity):
            if op in (operator.add, operator.sub):
                q = other.to(self.units)
                scalar_units = self.units
            else:
                q = other
                scalar_units = q.units

            scalar = float(q.magnitude)
            signal = np.full_like(self.signal, scalar, dtype=float)
            other = BaseData(
                signal=signal,
                units=scalar_units,
                uncertainties={k: np.zeros_like(v, dtype=float) for k, v in self.uncertainties.items()},
            )

        elif not isinstance(other, BaseData):
            # unsupported type
            return NotImplemented

        # Now both operands are BaseData
        if swapped:
            left, right = other, self
        else:
            left, right = self, other

        return _binary_basedata_op(left, right, op)

    def __add__(self, other: Any) -> BaseData:
        return self._binary_op(other, operator.add)

    def __radd__(self, other: Any) -> BaseData:
        return self._binary_op(other, operator.add, swapped=True)

    def __sub__(self, other: Any) -> BaseData:
        return self._binary_op(other, operator.sub)

    def __rsub__(self, other: Any) -> BaseData:
        return self._binary_op(other, operator.sub, swapped=True)

    def __mul__(self, other: Any) -> BaseData:
        return self._binary_op(other, operator.mul)

    def __rmul__(self, other: Any) -> BaseData:
        return self._binary_op(other, operator.mul, swapped=True)

    def __truediv__(self, other: Any) -> BaseData:
        return self._binary_op(other, operator.truediv)

    def __rtruediv__(self, other: Any) -> BaseData:
        return self._binary_op(other, operator.truediv, swapped=True)

    # ---- unary dunder + convenience methods ----

    def __neg__(self) -> BaseData:
        return negate_basedata_element(self)

    def __pow__(self, exponent: float, modulo=None) -> BaseData:
        if modulo is not None:
            return NotImplemented
        return powered_basedata_element(self, exponent)

    def sqrt(self) -> BaseData:
        return sqrt_basedata_element(self)

    def square(self) -> BaseData:
        return square_basedata_element(self)

    def log(self) -> BaseData:
        return log_basedata_element(self)

    def exp(self) -> BaseData:
        return exp_basedata_element(self)

    def sin(self) -> BaseData:
        return sin_basedata_element(self)

    def cos(self) -> BaseData:
        return cos_basedata_element(self)

    def tan(self) -> BaseData:
        return tan_basedata_element(self)

    def arcsin(self) -> BaseData:
        return arcsin_basedata_element(self)

    def arccos(self) -> BaseData:
        return arccos_basedata_element(self)

    def arctan(self) -> BaseData:
        return arctan_basedata_element(self)

    def reciprocal(self) -> BaseData:
        return reciprocal_basedata_element(self)


@define
class BaseData(UncertaintyOpsMixin):
    """
    BaseData stores a core data array (`signal`) with associated uncertainties, units,
    and metadata. It validates that any weights or uncertainty arrays broadcast to
    the shape of `signal`, and provides utilities for scaling operations and unit conversion.

    Attributes
    ----------
    signal : np.ndarray
        The primary data array. All weights/uncertainty arrays must be broadcastable
        to this shape.
    units : pint.Unit
        Physical units of `signal*scaling` and their uncertainties.
    uncertainties : Dict[str, np.ndarray]
        Uncertainty (as one‐sigma standard deviation) arrays keyed by type (e.g., “poisson”,
        “SEM”). Each array must broadcast to `signal.shape`. Variances are computed as
        `uncertainties[k]**2`.
        Uncertainty propagation in operations that combine BaseData elements will try to
        apply the incoming uncertainties by matched key,
        If only a single uncertainty is found (that should be named 'propagate_to_all'), it will be
        applied to all uncertainties.
    weights : np.ndarray, optional
        Weights for `signal` (default is a scaling 1.0) for use in averaging operations.
        Must broadcast to `signal.shape`.
    axes : List[BaseData | None]
        Optional metadata for each axis of `signal`. Defaults to an empty list.
    rank_of_data : int, default=0
        Rank (0–3) of the data; 1 is line data, 2 is image data. Must not exceed
        `signal.ndim`.

    Properties
    ----------
    variances : Dict[str, np.ndarray]
        Returns `{k: u**2 for k, u in uncertainties.items()}`. Assigning expects a dict
        of variance arrays; each is validated against `signal.shape` and
        converted into `uncertainties[k] = sqrt(var)`.
    shape : tuple[int, ...]
        Shape of the `signal` array.
    size : int
        Size of the `signal` array.

    Methods
    -------
    to_units(new_units: pint.Unit):
        Converts internal `signal` and all `uncertainties` to `new_units` if compatible with
        the existing `units`. Raises `TypeError` or `ValueError` on invalid input.
    """

    # required:
    # Core data array stored as an xarray DataArray
    signal: np.ndarray = field(
        converter=signal_converter, validator=v.instance_of(np.ndarray), on_setattr=setters.validate
    )
    # Unit of signal*scaling+offset - required input 'dimensionless' for dimensionless data
    units: pint.Unit = field(validator=v.instance_of(pint.Unit), converter=ureg.Unit, on_setattr=setters.validate)  # type: ignore
    # optional:
    # Dict of variances represented as xarray DataArray objects; defaulting to an empty dict
    uncertainties: Dict[str, np.ndarray] = field(
        factory=dict, converter=dict_signal_converter, validator=v.instance_of(dict), on_setattr=setters.validate
    )
    # weights for the signal, can be used for weighted averaging
    weights: np.ndarray = field(
        default=np.array(1.0),
        converter=signal_converter,
        validator=v.instance_of(np.ndarray),
        on_setattr=setters.validate,
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
        and the shapes of weights are compatible with the signal array.
        """
        # Validate variances
        for kind, var in self.variances.items():
            validate_broadcast(self.signal, var, f"variances[{kind}]")

        # Validate weights
        validate_broadcast(self.signal, self.weights, "weights")

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

    @property
    def shape(self) -> tuple[int, ...]:
        """
        Get the shape of the BaseData signal.

        Returns
        -------
        tuple[int, ...] :
            The shape of the signal.
        """
        return self.signal.shape

    @property
    def size(self) -> int:
        """
        Get the size of the BaseData signal.

        Returns
        -------
        int :
            The size of the signal.
        """
        return self.signal.size

    def to_units(self, new_units: pint.Unit, multiplicative_conversion=True) -> None:
        """
        Convert the signal and uncertainties to new units.
        """
        try:
            new_units = ureg.Unit(new_units)  # ensure new_units is a pint.Unit
        except pint.errors.UndefinedUnitError as e:
            raise ValueError(f"Invalid unit provided: {new_units}.") from e

        if not isinstance(new_units, ureg.Unit):
            raise TypeError(f"new_units must be a pint.Unit, got {type(new_units)}.")

        if not self.units.is_compatible_with(new_units):
            raise ValueError(
                f"""
              Cannot convert from {self.units} to {new_units}. Units are not compatible.
            """
            )

        # If the units are the same, no conversion is needed
        if self.units == new_units:
            logger.debug("No unit conversion needed, units are the same.")
            return

        logger.debug(f"Converting from {self.units} to {new_units}.")

        if multiplicative_conversion:
            # simple unit conversion, can be done to scalar
            # Convert signal
            cfact = new_units.m_from(self.units)
            self.signal *= cfact
            self.units = new_units
            # Convert uncertainty
            for key in self.uncertainties:  # fastest as far as my limited testing goes against iterating over items():
                self.uncertainties[key] *= cfact

        else:
            new_signal = ureg.Quantity(self.signal, self.units).to(new_units).magnitude
            # Convert uncertainties
            for key in self.uncertainties:
                # I am not sure but I think this would be the right way for non-multiplicative conversions
                self.uncertainties[key] *= new_signal / self.signal

    def __repr__(self):
        return f"BaseData(signal={self.signal}, uncertainties={self.uncertainties}, units={self.units})"

    def __str__(self):
        return f'{self.signal} {self.units} ± {[f"{u} ({k})" for k, u in self.uncertainties.items()]}'


# ---------------------------------------------------------------------------
# Unary operations built on the generic helper
# ---------------------------------------------------------------------------


def negate_basedata_element(element: BaseData) -> BaseData:
    """Negate a BaseData element with uncertainty and units propagation."""
    result = BaseData(
        signal=-element.signal,
        units=element.units,
        uncertainties=_copy_uncertainties(element.uncertainties),
    )
    return _inherit_metadata(element, result)


def sqrt_basedata_element(element: BaseData) -> BaseData:
    """Square root: y = sqrt(x), σ_y ≈ σ_x / (2 sqrt(x)). x must be >= 0."""
    return _unary_basedata_op(
        element=element,
        func=np.sqrt,
        dfunc=lambda x: 0.5 / np.sqrt(x),
        out_units=element.units**0.5,
        domain=lambda x: x >= 0,
    )


def square_basedata_element(element: BaseData) -> BaseData:
    """Square: y = x^2, σ_y ≈ |2x| σ_x."""
    return _unary_basedata_op(
        element=element,
        func=lambda x: x**2,
        dfunc=lambda x: 2.0 * x,
        out_units=element.units**2,
    )


def powered_basedata_element(element: BaseData, exponent: float) -> BaseData:
    """Power: y = x**n, σ_y ≈ |n x^(n-1)| σ_x."""
    # If exponent is non-integer, restrict to x >= 0 to avoid complex results.
    exp_float = float(exponent)
    if float(exp_float).is_integer():
        domain = None  # all real x are allowed
    else:
        domain = lambda x: x >= 0  # noqa: E731

    return _unary_basedata_op(
        element=element,
        func=lambda x: x**exp_float,
        dfunc=lambda x: exp_float * (x ** (exp_float - 1.0)),
        out_units=element.units**exponent,
        domain=domain,
    )


def log_basedata_element(element: BaseData) -> BaseData:
    """Natural log: y = ln(x), σ_y ≈ |1/x| σ_x. x must be > 0."""
    return _unary_basedata_op(
        element=element,
        func=np.log,
        dfunc=lambda x: 1.0 / x,
        out_units=ureg.dimensionless,
        domain=lambda x: x > 0,
    )


def exp_basedata_element(element: BaseData) -> BaseData:
    """Exponential: y = exp(x), σ_y ≈ exp(x) σ_x. Argument should be dimensionless."""
    return _unary_basedata_op(
        element=element,
        func=np.exp,
        dfunc=np.exp,
        out_units=ureg.dimensionless,
    )


def sin_basedata_element(element: BaseData) -> BaseData:
    """Sine: y = sin(x), σ_y ≈ |cos(x)| σ_x. x in radians."""
    return _unary_basedata_op(
        element=element,
        func=np.sin,
        dfunc=np.cos,
        out_units=ureg.dimensionless,
    )


def cos_basedata_element(element: BaseData) -> BaseData:
    """Cosine: y = cos(x), σ_y ≈ |sin(x)| σ_x. x in radians."""
    return _unary_basedata_op(
        element=element,
        func=np.cos,
        dfunc=np.sin,  # derivative is -sin(x), abs removes sign
        out_units=ureg.dimensionless,
    )


def tan_basedata_element(element: BaseData) -> BaseData:
    """Tangent: y = tan(x), σ_y ≈ |1/cos^2(x)| σ_x. x in radians."""
    return _unary_basedata_op(
        element=element,
        func=np.tan,
        dfunc=lambda x: 1.0 / (np.cos(x) ** 2),
        out_units=ureg.dimensionless,
    )


def arcsin_basedata_element(element: BaseData) -> BaseData:
    """Arcsin: y = arcsin(x), σ_y ≈ |1/sqrt(1-x^2)| σ_x. x dimensionless, |x| <= 1."""
    return _unary_basedata_op(
        element=element,
        func=np.arcsin,
        dfunc=lambda x: 1.0 / np.sqrt(1.0 - x**2),
        out_units=ureg.radian,
        domain=lambda x: np.abs(x) <= 1.0,
    )


def arccos_basedata_element(element: BaseData) -> BaseData:
    """Arccos: y = arccos(x), σ_y ≈ |1/sqrt(1-x^2)| σ_x. x dimensionless, |x| <= 1."""
    return _unary_basedata_op(
        element=element,
        func=np.arccos,
        dfunc=lambda x: 1.0 / np.sqrt(1.0 - x**2),  # abs removes the sign
        out_units=ureg.radian,
        domain=lambda x: np.abs(x) <= 1.0,
    )


def arctan_basedata_element(element: BaseData) -> BaseData:
    """Arctan: y = arctan(x), σ_y ≈ |1/(1+x^2)| σ_x. x dimensionless."""
    return _unary_basedata_op(
        element=element,
        func=np.arctan,
        dfunc=lambda x: 1.0 / (1.0 + x**2),
        out_units=ureg.radian,
    )


def reciprocal_basedata_element(element: BaseData) -> BaseData:
    """Reciprocal: y = 1/x, σ_y ≈ |1/x^2| σ_x."""
    return _unary_basedata_op(
        element=element,
        func=lambda x: 1.0 / x,
        dfunc=lambda x: 1.0 / (x**2),
        out_units=element.units**-1,  # <-- unit, not quantity
        domain=lambda x: x != 0,
    )
