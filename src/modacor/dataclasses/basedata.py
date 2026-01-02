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
class _VarianceDict(MutableMapping):
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

    Semantics
    ---------
    - Uncertainty dict entries are treated as independent sources keyed by name.
    - "propagate_to_all" is treated as a *global fallback* only when it is the sole key
      on an operand. Otherwise it is treated like a normal key name and participates via
      rule 4 (union, no cross-key combining).

    Rules for result uncertainty keys:
    1) If both operands have only "propagate_to_all":
         result.uncertainties = {"propagate_to_all": σ}
    2) If exactly one operand has only "propagate_to_all" and the other has non-global keys:
         result.uncertainties contains only the other operand's non-global keys.
         The global term contributes into each of those keys.
    3) Matching non-global keys propagate and combine by key.
    4) Non-matching non-global keys: transfer/propagate each key independently into the result
         (union of keys), without cross-key combining.

    Propagation formulas (uncorrelated, first-order)
    -----------------------------------------------
    Let A = left.signal * left.units, B = right.signal * right.units, R = op(A, B).
    Using magnitudes A_val, B_val and absolute σA, σB:

    - Add/Sub:   σR² = σA² + σB²  (after converting σA, σB to result units)
    - Mul:       σR² = (B σA)² + (A σB)²
    - Div:       σR² = (σA/B)² + (A σB/B²)²
    """
    # Nominal result (pint handles unit logic & broadcasting)
    A_q = left.signal * left.units
    B_q = right.signal * right.units
    base_result = op(A_q, B_q)
    base_signal = np.asarray(base_result.magnitude, dtype=float)
    result_units = base_result.units
    out_shape = base_signal.shape

    # Broadcast nominal magnitudes to the result shape
    A_val = np.broadcast_to(np.asarray(left.signal, dtype=float), out_shape)
    B_val = np.broadcast_to(np.asarray(right.signal, dtype=float), out_shape)

    left_unc = left.uncertainties
    right_unc = right.uncertainties

    # "global-only" if and only if propagate_to_all is the sole key
    left_global = left_unc.get("propagate_to_all") if set(left_unc.keys()) == {"propagate_to_all"} else None
    right_global = right_unc.get("propagate_to_all") if set(right_unc.keys()) == {"propagate_to_all"} else None

    left_non_global_keys = set(left_unc.keys()) - {"propagate_to_all"}
    right_non_global_keys = set(right_unc.keys()) - {"propagate_to_all"}

    # Decide output keys per rules
    if left_global is not None and right_global is not None:
        out_keys = {"propagate_to_all"}
        drop_global_key = False  # we *do* want it
    elif left_global is not None and right_non_global_keys:
        out_keys = set(right_non_global_keys)
        drop_global_key = True
    elif right_global is not None and left_non_global_keys:
        out_keys = set(left_non_global_keys)
        drop_global_key = True
    else:
        # General case: union of non-global keys
        out_keys = left_non_global_keys | right_non_global_keys
        drop_global_key = True  # we never emit propagate_to_all unless both-global-only

    def _as_broadcast_float(err: Any) -> np.ndarray:
        arr = np.asarray(err, dtype=float)
        return np.broadcast_to(arr, out_shape)

    def _get_err(unc_map: Dict[str, np.ndarray], key: str, global_err: Any | None) -> Any:
        """Return uncertainty for `key`, falling back to global_err if provided, else 0."""
        if key in unc_map:
            return unc_map[key]
        if global_err is not None:
            return global_err
        return 0.0

    # Precompute unit conversion factors (magnitudes)
    if op in (operator.add, operator.sub):
        cf_A = result_units.m_from(left.units)
        cf_B = result_units.m_from(right.units)

        result_unc: Dict[str, np.ndarray] = {}
        for key in out_keys:
            sigma_A = _as_broadcast_float(_get_err(left_unc, key, left_global)) * cf_A
            sigma_B = _as_broadcast_float(_get_err(right_unc, key, right_global)) * cf_B
            result_unc[key] = np.sqrt(sigma_A**2 + sigma_B**2)

    elif op is operator.mul:
        # Convert from left.units * right.units to result_units
        cf_AB = result_units.m_from(left.units * right.units)

        result_unc = {}
        for key in out_keys:
            sigma_A = _as_broadcast_float(_get_err(left_unc, key, left_global))
            sigma_B = _as_broadcast_float(_get_err(right_unc, key, right_global))
            termA = (B_val * sigma_A) * cf_AB
            termB = (A_val * sigma_B) * cf_AB
            result_unc[key] = np.sqrt(termA**2 + termB**2)

    elif op is operator.truediv:
        # Convert from left.units / right.units to result_units
        cf_A_div_B = result_units.m_from(left.units / right.units)

        result_unc = {}
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            for key in out_keys:
                sigma_A = _as_broadcast_float(_get_err(left_unc, key, left_global))
                sigma_B = _as_broadcast_float(_get_err(right_unc, key, right_global))

                termA = (sigma_A / B_val) * cf_A_div_B
                termB = (A_val * sigma_B / (B_val**2)) * cf_A_div_B
                sigma = np.sqrt(termA**2 + termB**2)

                # Division by zero -> undefined uncertainty
                sigma = np.where(B_val == 0.0, np.nan, sigma)
                result_unc[key] = sigma
    else:
        raise NotImplementedError(f"Operation {op} not supported in _binary_basedata_op")  # noqa: E713

    # Only emit propagate_to_all in the both-global-only case
    if drop_global_key:
        result_unc.pop("propagate_to_all", None)

    result = BaseData(signal=base_signal, units=result_units, uncertainties=result_unc)
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
        Uncertainty (as one‐sigma standard deviation) arrays keyed by type (e.g. "Poisson",
        "pixel_index"). Each array must broadcast to ``signal.shape``. Variances are computed
        as ``uncertainties[k]**2``.

        Binary operations propagate uncertainties using first-order, uncorrelated propagation.
        Keys are treated as *independent sources*.

        Special key: ``"propagate_to_all"``
        ---------------------------------
        ``"propagate_to_all"`` is treated as a *global* uncertainty **only when it is the sole
        key present** on an operand.

        The propagation rules for an operation ``result = op(a, b)`` are:

        1. If both operands only contain ``"propagate_to_all"``, the result contains only
           ``"propagate_to_all"`` (propagated normally).

        2. If one operand only contains ``"propagate_to_all"`` and the other contains one or
           more non-``"propagate_to_all"`` keys, the result contains only those non-global keys.
           The global uncertainty contributes as a fallback term to each of those keys.

        3. If both operands contain matching non-global keys, those keys are propagated and
           combined by key.

        4. If both operands contain non-global keys but with no matches, the result contains
           the union of keys, and each key is propagated from its originating operand only
           (no cross-key combining).
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
    # Core data array stored as a numpy ndarray
    signal: np.ndarray = field(
        converter=signal_converter, validator=v.instance_of(np.ndarray), on_setattr=setters.validate
    )
    # Unit of signal*scaling+offset - required input 'dimensionless' for dimensionless data
    units: pint.Unit = field(validator=v.instance_of(pint.Unit), converter=ureg.Unit, on_setattr=setters.validate)  # type: ignore
    # optional:
    # Dict of variances represented as numpy ndarray objects; defaulting to an empty dict
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

        # Warn if axes length does not match signal.ndim
        if self.axes and len(self.axes) != self.signal.ndim:
            logger.debug(
                "BaseData.axes length (%d) does not match signal.ndim (%d).",
                len(self.axes),
                self.signal.ndim,
            )

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

    def to_dimensionless(self) -> None:
        """
        Convert the signal and uncertainties to dimensionless units if possible.
        """
        if not self.is_dimensionless:
            self.to_units(ureg.dimensionless)

    @property
    def is_dimensionless(self) -> bool:
        """
        Check if the BaseData is dimensionless.

        Returns
        -------
        bool :
            True if the units are dimensionless, False otherwise.
        """
        return self.units == ureg.dimensionless

    # not funcional yet, but needs to be implemented for data ingestion. and recommended as standard procedure to always convert to base units.
    # def to_base_units(self) -> None:
    #     """
    #     Convert the signal and uncertainties to base units.
    #     """
    #     base_units = self.units.to_base_units().units
    #     self.to_units(base_units)

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

        if not multiplicative_conversion:
            # This path is subtle for offset units (e.g. degC <-> K) and we
            # don't want to silently get uncertainties wrong.
            raise NotImplementedError(
                "Non-multiplicative unit conversions are not yet implemented for BaseData.\n"
                "If you need this, we should design explicit rules (e.g. using delta units)."
            )

        logger.debug(f"Converting from {self.units} to {new_units}.")

        # simple unit conversion, can be done to scalar
        # Convert signal
        cfact = new_units.m_from(self.units)
        self.signal *= cfact
        self.units = new_units
        # Convert uncertainty
        for key in self.uncertainties:  # fastest as far as my limited testing goes against iterating over items():
            self.uncertainties[key] *= cfact

    def indexed(self, indexer: Any, *, rank_of_data: int | None = None) -> "BaseData":
        """
        Return a new BaseData corresponding to ``self`` indexed by ``indexer``.

        Parameters
        ----------
        indexer :
            Any valid NumPy indexer (int, slice, tuple of slices, boolean mask, ...),
            applied consistently to ``signal`` and all uncertainty / weight arrays.
        rank_of_data :
            Optional explicit rank_of_data for the returned BaseData. If omitted,
            it will default to ``min(self.rank_of_data, result.signal.ndim)``.

        Notes
        -----
        - Units are preserved.
        - Uncertainties and weights are sliced with the same indexer where possible.
        - Axes handling is conservative: existing axes are kept unchanged. If you
          want axes to track slicing semantics more strictly, a higher-level
          helper can be added later.
        """
        sig = np.asarray(self.signal)[indexer]

        # Slice uncertainties with the same indexer
        new_uncs: Dict[str, np.ndarray] = {}
        for k, u in self.uncertainties.items():
            u_arr = np.asarray(u, dtype=float)
            # broadcast to signal shape, then apply the same indexer
            u_full = np.broadcast_to(u_arr, self.signal.shape)
            new_uncs[k] = u_full[indexer].copy()

        # Try to slice weights; if shapes don't line up, fall back to scalar 1.0
        try:
            w_arr = np.asarray(self.weights, dtype=float)
            new_weights = w_arr[indexer].copy()
        except Exception:
            new_weights = np.array(1.0, dtype=float)

        # Decide rank_of_data for the result
        if rank_of_data is None:
            new_rank = min(self.rank_of_data, np.ndim(sig))
        else:
            new_rank = int(rank_of_data)

        result = BaseData(
            signal=np.asarray(sig, dtype=float),
            units=self.units,
            uncertainties=new_uncs,
            weights=new_weights,
            # For now we keep axes as-is; more sophisticated axis handling can be
            # added once the usage patterns are clear.
            axes=list(self.axes),
            rank_of_data=new_rank,
        )
        return result

    def copy(self, with_axes: bool = True) -> "BaseData":
        """
        Return a new BaseData with copied signal/uncertainties/weights.
        Axes are shallow-copied (list copy) by default, so axis objects
        themselves are still shared.
        """
        new = BaseData(
            signal=np.array(self.signal, copy=True),
            units=self.units,
            uncertainties=_copy_uncertainties(self.uncertainties),
            weights=np.array(self.weights, copy=True),
            axes=list(self.axes) if with_axes else [],
            rank_of_data=self.rank_of_data,
        )
        return new

    def __repr__(self):
        return (
            f"BaseData(shape={self.signal.shape}, dtype={self.signal.dtype}, units={self.units}, "
            f"n_uncertainties={len(self.uncertainties)}, rank_of_data={self.rank_of_data})"
        )

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
    # ensure element is dimensionless:
    element.to_dimensionless()
    return _unary_basedata_op(
        element=element,
        func=np.log,
        dfunc=lambda x: 1.0 / x,
        out_units=ureg.dimensionless,
        domain=lambda x: x > 0,
    )


def exp_basedata_element(element: BaseData) -> BaseData:
    """Exponential: y = exp(x), σ_y ≈ exp(x) σ_x. Argument should be dimensionless."""
    # ensure element is dimensionless:
    element.to_dimensionless()
    return _unary_basedata_op(
        element=element,
        func=np.exp,
        dfunc=np.exp,
        out_units=ureg.dimensionless,
    )


def sin_basedata_element(element: BaseData) -> BaseData:
    """Sine: y = sin(x), σ_y ≈ |cos(x)| σ_x. x in radians."""
    # ensure element is in radian:
    element.to_units(ureg.radian)
    return _unary_basedata_op(
        element=element,
        func=np.sin,
        dfunc=np.cos,
        out_units=ureg.dimensionless,
    )


def cos_basedata_element(element: BaseData) -> BaseData:
    """Cosine: y = cos(x), σ_y ≈ |sin(x)| σ_x. x in radians."""
    # ensure element is in radian:
    element.to_units(ureg.radian)
    return _unary_basedata_op(
        element=element,
        func=np.cos,
        dfunc=np.sin,  # derivative is -sin(x), abs removes sign
        out_units=ureg.dimensionless,
    )


def tan_basedata_element(element: BaseData) -> BaseData:
    """Tangent: y = tan(x), σ_y ≈ |1/cos^2(x)| σ_x. x in radians."""
    # ensure element is in radian:
    element.to_units(ureg.radian)
    return _unary_basedata_op(
        element=element,
        func=np.tan,
        dfunc=lambda x: 1.0 / (np.cos(x) ** 2),
        out_units=ureg.dimensionless,
    )


def arcsin_basedata_element(element: BaseData) -> BaseData:
    """Arcsin: y = arcsin(x), σ_y ≈ |1/sqrt(1-x^2)| σ_x. x dimensionless, |x| <= 1."""
    # ensure element is dimensionless:
    element.to_dimensionless()
    return _unary_basedata_op(
        element=element,
        func=np.arcsin,
        dfunc=lambda x: 1.0 / np.sqrt(1.0 - x**2),
        out_units=ureg.radian,
        domain=lambda x: np.abs(x) <= 1.0,
    )


def arccos_basedata_element(element: BaseData) -> BaseData:
    """Arccos: y = arccos(x), σ_y ≈ |1/sqrt(1-x^2)| σ_x. x dimensionless, |x| <= 1."""
    # ensure element is dimensionless:
    element.to_dimensionless()
    return _unary_basedata_op(
        element=element,
        func=np.arccos,
        dfunc=lambda x: 1.0 / np.sqrt(1.0 - x**2),  # abs removes the sign
        out_units=ureg.radian,
        domain=lambda x: np.abs(x) <= 1.0,
    )


def arctan_basedata_element(element: BaseData) -> BaseData:
    """Arctan: y = arctan(x), σ_y ≈ |1/(1+x^2)| σ_x. x dimensionless."""
    # ensure element is dimensionless:
    element.to_dimensionless()
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
