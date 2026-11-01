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
from modacor.models.uncertainty import validate_uncertainty_keys, with_uncertainty_estimates

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
        _validate_uncertainties(self._parent)
        # Accept scaling or array‐like, coerce to ndarray
        arr = np.asarray(var, dtype=float)
        # Validate broadcast to signal
        validate_broadcast(self._parent.signal, arr, f"variances[{key}]")
        if np.any(arr < 0):
            raise ValueError(f"variances[{key}] must be nonnegative.")
        # Store sqrt(var) as the one‐sigma uncertainty
        self._parent.uncertainties = with_uncertainty_estimates(self._parent.uncertainties, {key: np.asarray(arr**0.5)})

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
    Convert the input to an owned array, preserving an array input's dtype.
    """
    return value.copy() if isinstance(value, np.ndarray) else np.array(value, dtype=float)


def dict_signal_converter(value: Dict[str, int | float | np.ndarray]) -> Dict[str, np.ndarray]:
    """
    Convert a dictionary of values to a dictionary of numpy arrays.
    Every value is copied so separate containers never share uncertainty storage.
    """
    validate_uncertainty_keys(value)
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


def _inherit_metadata(source: BaseData, result: BaseData) -> BaseData:
    """
    Copy metadata-like attributes from `source` to `result` (axes, rank_of_data, weights)
    without touching numerical content (signal, units, uncertainties).
    """
    # Shallow-copy axes list to avoid aliasing of the list object itself
    result.axes = list(source.axes)

    # Keep the same rank_of_data; attrs validation ensures it is still valid
    result.rank_of_data = source.rank_of_data

    # Try to propagate weights; if shapes are incompatible, keep defaults.
    # Scalar/default weights stay scalar so later operations do not mistake
    # broadcast defaults for real per-point weight maps.
    try:
        arr = np.asarray(source.weights)
        if arr.size == 1:
            result.weights = np.array(arr, copy=True)
        else:
            validate_broadcast(result.signal, arr, "weights")
            result.weights = np.broadcast_to(arr, result.signal.shape).copy()
    except ValueError:
        logger.debug("Could not broadcast source weights to result shape; leaving default weights on result BaseData.")

    return result


def _has_array_weights(data: BaseData) -> bool:
    """
    Return True for weights that carry per-point metadata.

    Size-1 weights are treated like scalar/default weights because they do not
    identify a particular data grid.
    """
    return np.asarray(data.weights).size != 1


def _has_structural_metadata(data: BaseData) -> bool:
    """
    Return True when `data` carries metadata that should constrain arithmetic.

    Units and uncertainties are numerical contracts handled by Pint and the
    uncertainty propagation code. This helper is intentionally limited to cheap
    structural metadata checks.
    """
    return bool(data.axes) or data.rank_of_data != 0 or _has_array_weights(data)


def _validate_binary_metadata_compatibility(left: BaseData, right: BaseData, out_shape: tuple[int, ...]) -> None:
    """
    Validate only cheap metadata contracts for binary BaseData arithmetic.

    This deliberately avoids deep axis comparison and unit policy decisions.
    Pint handles unit compatibility/algebra, while this check prevents silent
    inheritance of rank/axes/weight metadata across incompatible array layouts.
    """
    left_has_metadata = _has_structural_metadata(left)
    right_has_metadata = _has_structural_metadata(right)

    if left.axes and left.signal.shape != out_shape:
        raise ValueError(
            "Cannot preserve left BaseData axes after arithmetic: "
            f"left signal shape {left.signal.shape} broadcasts to result shape {out_shape}."
        )
    if right.axes and right.signal.shape != out_shape:
        raise ValueError(
            "Cannot preserve right BaseData axes after arithmetic: "
            f"right signal shape {right.signal.shape} broadcasts to result shape {out_shape}."
        )

    if not (left_has_metadata and right_has_metadata):
        return

    if left.rank_of_data != right.rank_of_data:
        raise ValueError(
            "Cannot combine BaseData metadata with different rank_of_data values: "
            f"{left.rank_of_data} != {right.rank_of_data}."
        )

    if left.axes and right.axes and len(left.axes) != len(right.axes):
        raise ValueError(
            "Cannot combine BaseData metadata with different axes lengths: " f"{len(left.axes)} != {len(right.axes)}."
        )


def _inherit_binary_metadata(left: BaseData, right: BaseData, result: BaseData) -> BaseData:
    """
    Preserve binary-operation metadata after cheap compatibility checks.

    Metadata-neutral operands, such as scalar factors or same-shaped correction
    arrays without rank/axes/array weights, do not override metadata-bearing
    operands. If only the right operand carries array-valued weights, preserve
    those weights after copying the primary metadata source.
    """
    out_shape = result.signal.shape
    _validate_binary_metadata_compatibility(left, right, out_shape)

    left_has_metadata = _has_structural_metadata(left)
    right_has_metadata = _has_structural_metadata(right)

    source = right if right_has_metadata and not left_has_metadata else left
    result = _inherit_metadata(source, result)

    if _has_array_weights(right) and not _has_array_weights(left):
        arr = np.asarray(right.weights)
        validate_broadcast(result.signal, arr, "weights")
        result.weights = np.broadcast_to(arr, result.signal.shape).copy()

    return result


def _validate_uncertainties(element: BaseData) -> None:
    """Validate the mutable mapping before any operation can write."""
    validate_uncertainty_keys(element.uncertainties)
    for key, value in element.uncertainties.items():
        arr = np.asarray(value)
        validate_broadcast(element.signal, arr, f"uncertainties[{key}]")
        if np.any(arr < 0):
            raise ValueError(f"uncertainties[{key}] must be nonnegative.")


def _binary_units(left: BaseData, right: BaseData, op) -> tuple[pint.Unit, float, float]:
    """Resolve units using scalar Pint arithmetic, without frame-sized quantities."""
    units = op(1 * left.units, 1 * right.units).units
    if op in (operator.add, operator.sub):
        return units, float(units.m_from(left.units)), float(units.m_from(right.units))
    source_units = left.units * right.units if op is operator.mul else left.units / right.units
    factor = float(units.m_from(source_units))
    return units, factor, factor


def _propagated_uncertainties(left: BaseData, right: BaseData, op, shape, plan, factors):
    """Propagate only present terms; finish before allocating the result signal."""
    keys, global_a, global_b = plan
    factor_a, factor_b = factors
    a, b = left.signal, right.signal
    result = {}
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        for key in keys:
            sigma_a = left.uncertainties.get(key, global_a)
            sigma_b = right.uncertainties.get(key, global_b)
            sigma = np.empty(shape, dtype=float)

            def term(source, side, out):
                if op in (operator.add, operator.sub):
                    np.multiply(source, abs(factor_a if side == "a" else factor_b), out=out)
                elif op is operator.mul:
                    np.multiply(source, b if side == "a" else a, out=out)
                    out *= abs(factor_a)
                elif side == "a":
                    np.divide(source, b, out=out)
                    out *= abs(factor_a)
                else:
                    np.multiply(source, a, out=out)
                    np.divide(out, b, out=out)
                    np.divide(out, b, out=out)
                    out *= abs(factor_b)
                np.abs(out, out=out)

            if sigma_a is not None:
                term(sigma_a, "a", sigma)
                if sigma_b is not None:
                    scratch = np.empty(shape, dtype=float)
                    term(sigma_b, "b", scratch)
                    # Preserve the generic arithmetic NaN/Inf policy, including
                    # NaN + Inf -> NaN, using only the two existing buffers.
                    np.square(sigma, out=sigma)
                    np.square(scratch, out=scratch)
                    np.add(sigma, scratch, out=sigma)
                    np.sqrt(sigma, out=sigma)
                    del scratch
            else:
                term(sigma_b, "b", sigma)
            if op is operator.truediv:
                np.copyto(sigma, np.nan, where=np.broadcast_to(b == 0, shape))
            result[key] = sigma
    return result


def _binary_basedata_op(left: BaseData, right: BaseData, op: Callable[[Any, Any], Any]) -> BaseData:
    """First-order independent-operand propagation using the Stage 0 key table."""
    _validate_uncertainties(left)
    _validate_uncertainties(right)
    plan = _uncertainty_key_plan(left.uncertainties, right.uncertainties)
    shape = np.broadcast_shapes(left.signal.shape, right.signal.shape)
    _validate_binary_metadata_compatibility(left, right, shape)
    units, factor_a, factor_b = _binary_units(left, right, op)
    uncertainties = _propagated_uncertainties(left, right, op, shape, plan, (factor_a, factor_b))
    signal = np.empty(shape, dtype=float)
    ufunc = {operator.add: np.add, operator.sub: np.subtract, operator.mul: np.multiply, operator.truediv: np.divide}[
        op
    ]
    if op in (operator.add, operator.sub):
        # The result uses the left unit, so its factor is normally one. Scale
        # the right operand into the output buffer rather than a transient frame.
        np.multiply(right.signal, factor_b, out=signal)
        ufunc(left.signal if factor_a == 1.0 else left.signal * factor_a, signal, out=signal)
    else:
        ufunc(left.signal, right.signal, out=signal)
        if factor_a != 1.0:
            signal *= factor_a
    # These arrays were allocated above and belong exclusively to this result.
    # Public constructors still copy all inputs. Adopt internal results without
    # duplicating an entire detector frame for each uncertainty key.
    result = BaseData(signal=0.0, units=units)
    result.signal = np.asarray(signal, dtype=float)
    result.uncertainties = uncertainties
    return _inherit_binary_metadata(left, right, result)


def _coerce_inplace_divisor(other: Any) -> BaseData | None:
    if isinstance(other, numbers.Real):
        return BaseData(signal=float(other), units=ureg.dimensionless)
    if isinstance(other, pint.Quantity):
        return BaseData(signal=float(other.magnitude), units=other.units)
    if isinstance(other, BaseData):
        return other
    return None


def _coerce_inplace_add_sub_operand(target: BaseData, other: Any) -> BaseData | None:
    if isinstance(other, numbers.Real):
        return BaseData(signal=float(other), units=target.units)
    if isinstance(other, pint.Quantity):
        return BaseData(signal=float(other.magnitude), units=other.units)
    if isinstance(other, BaseData):
        return other
    return None


def _uncertainty_key_plan(
    left_unc: Dict[str, np.ndarray],
    right_unc: Dict[str, np.ndarray],
) -> tuple[set[str], Any | None, Any | None]:
    validate_uncertainty_keys(left_unc)
    validate_uncertainty_keys(right_unc)
    left_global = left_unc.get("propagate_to_all")
    right_global = right_unc.get("propagate_to_all")
    left_keys, right_keys = set(left_unc), set(right_unc)
    if left_global is not None and right_keys - {"propagate_to_all"}:
        out_keys = right_keys
    elif right_global is not None and left_keys - {"propagate_to_all"}:
        out_keys = left_keys
    else:
        out_keys = left_keys | right_keys
    if left_keys and right_keys and left_global is None and right_global is None and left_keys != right_keys:
        logger.debug("Uncertainty key sets differ; propagating their union with absent terms treated as zero.")
    return out_keys, left_global, right_global


def _install_inplace_result(target: BaseData, result: BaseData) -> BaseData:
    """Commit a fully checked result while preserving the container identity.

    Always use the out-of-place numerical path, including for operands whose
    signals or uncertainty arrays share memory (S0.4). No operand buffer is
    written to, so self aliases, cross-array aliases and externally assigned
    views all read their original values without frame-sized snapshots.
    """
    target.signal = result.signal
    target.units = result.units
    target.uncertainties = result.uncertainties
    target.axes = result.axes
    target.rank_of_data = result.rank_of_data
    target.weights = result.weights
    return target


def _inplace_truediv_metadata_neutral(target: BaseData, other: BaseData) -> BaseData:
    # Share the key planner, shape/unit checks and numerical path with division.
    return _install_inplace_result(target, _binary_basedata_op(target, other, operator.truediv))


def _inplace_add_sub(target: BaseData, other: BaseData, *, sign: float) -> BaseData:
    op = operator.add if sign > 0 else operator.sub
    return _install_inplace_result(target, _binary_basedata_op(target, other, op))


def _unary_inputs(element: BaseData, input_units: pint.Unit | None = None):
    """Return signal and all sigmas with one inline conversion; never mutate input."""
    _validate_uncertainties(element)
    factor = 1.0 if input_units is None else float(input_units.m_from(element.units))
    if factor == 1.0:
        return element.signal, element.uncertainties
    return element.signal * factor, {
        key: np.asarray(value) * abs(factor) for key, value in element.uncertainties.items()
    }


def _unary_basedata_op(
    element: BaseData,
    func: Callable[[np.ndarray], np.ndarray],
    dfunc: Callable[[np.ndarray], np.ndarray],
    out_units: pint.Unit,
    domain: Callable[[np.ndarray], np.ndarray] | None = None,
    input_units: pint.Unit | None = None,
) -> BaseData:
    """
    Generic unary op: y = func(x), σ_y ≈ |dfunc(x)| σ_x, with an optional domain.

    Outside the domain, signal and uncertainties are set to NaN (using np.where-style masking).
    """
    signal, uncertainties = _unary_inputs(element, input_units)
    x = np.asarray(signal)
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

    for key, err in uncertainties.items():
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

        if op in (operator.add, operator.sub):
            other = _coerce_inplace_add_sub_operand(self, other)
        else:
            other = _coerce_inplace_divisor(other)
        if other is None:
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

    def __iadd__(self, other: Any) -> BaseData:
        other_basedata = _coerce_inplace_add_sub_operand(self, other)
        if other_basedata is None:
            return NotImplemented
        return _inplace_add_sub(self, other_basedata, sign=1.0)

    def __isub__(self, other: Any) -> BaseData:
        other_basedata = _coerce_inplace_add_sub_operand(self, other)
        if other_basedata is None:
            return NotImplemented
        return _inplace_add_sub(self, other_basedata, sign=-1.0)

    def __mul__(self, other: Any) -> BaseData:
        return self._binary_op(other, operator.mul)

    def __imul__(self, other: Any) -> BaseData:
        other_basedata = _coerce_inplace_divisor(other)
        if other_basedata is None:
            return NotImplemented
        return _install_inplace_result(self, _binary_basedata_op(self, other_basedata, operator.mul))

    def __rmul__(self, other: Any) -> BaseData:
        return self._binary_op(other, operator.mul, swapped=True)

    def __truediv__(self, other: Any) -> BaseData:
        return self._binary_op(other, operator.truediv)

    def __rtruediv__(self, other: Any) -> BaseData:
        return self._binary_op(other, operator.truediv, swapped=True)

    def __itruediv__(self, other: Any) -> BaseData:
        other_basedata = _coerce_inplace_divisor(other)
        if other_basedata is None:
            return NotImplemented

        return _inplace_truediv_metadata_neutral(self, other_basedata)

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

    def squeeze(self) -> BaseData:
        return squeeze_basedata_element(self)


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
        Each key is propagated separately using the independent-operand approximation.

        Special key: ``"propagate_to_all"``
        ---------------------------------
        ``"propagate_to_all"`` is treated as a *global* uncertainty **only when it is the sole
        key present** on an operand. A mixed mapping raises ``ValueError``.

        The propagation rules for an operation ``result = op(a, b)`` are:

        1. If both operands only contain ``"propagate_to_all"``, the result contains only
           ``"propagate_to_all"`` (propagated normally).

        2. If one operand only contains ``"propagate_to_all"`` and the other contains one or
           more non-``"propagate_to_all"`` keys, the result contains only those non-global keys.
           The global uncertainty contributes as a fallback term to each of those keys.

        3. If both operands contain matching non-global keys, those keys are propagated and
           combined by key.

        4. If both operands contain differing sets of non-global keys, the result contains
           the union of keys, and each key is propagated from its originating operand only
           (no cross-key combining).

        5. A global-only operand meeting an empty mapping retains ``"propagate_to_all"``.

        An empty mapping means unquantified, not exact, even though absent terms are
        treated as zero by this arithmetic. Aliases are computationally safe, but physical
        correlation is not represented: uncertainty in ``a - a`` does not cancel.
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
    to_base_units(): Converts internal signal and uncertainties to the base units implied by current units.
    to_dimensionless(): Converts internal signal and uncertainties to dimensionless units if possible.
    is_dimensionless() -> bool:
        Returns True if `units` is compatible with dimensionless, False otherwise.
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
        # Validate uncertainties directly. Going through ``variances`` would
        # square every array and can allocate gigabytes for large detector data.
        _validate_uncertainties(self)

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
        _validate_uncertainties(self)
        validate_uncertainty_keys(value)
        uncertainties = {}
        for kind, var in value.items():
            arr = np.asarray(var, dtype=float)
            validate_broadcast(self.signal, arr, f"variances[{kind}]")
            if np.any(arr < 0):
                raise ValueError(f"variances[{kind}] must be nonnegative.")
            uncertainties[kind] = np.asarray(arr**0.5)
        self.uncertainties = uncertainties

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
        self.to_units(ureg.dimensionless)

    @property
    def is_dimensionless(self) -> bool:
        """
        Check if the BaseData is dimensionless.

        Returns
        -------
        bool :
            True if the units are compatible with dimensionless, False otherwise.
        """
        return self.units.is_compatible_with(ureg.dimensionless)

    def to_base_units(self, *, multiplicative_conversion: bool = True) -> None:
        """
        Convert the signal and uncertainties to the pint *base units* of `self.units`.

        Notes
        -----
        - This is an in-place operation.
        - For offset / non-multiplicative units (e.g. degC <-> K), this will raise unless
          explicit support is implemented (see `to_units`).
        """
        # Determine the canonical base unit for the current units
        # (use a Quantity to let pint resolve to_base_units properly)
        base_units = (1 * self.units).to_base_units().units

        try:
            self.to_units(base_units, multiplicative_conversion=multiplicative_conversion)
        except pint.errors.OffsetUnitCalculusError as e:
            # Offset-unit conversions are affine, not purely multiplicative.
            raise NotImplementedError(
                "BaseData.to_base_units() encountered an offset / non-multiplicative unit conversion.\n"
                "This is not supported yet because uncertainties require explicit rules (e.g. delta units)."
            ) from e

    def to_units(self, new_units: pint.Unit, multiplicative_conversion=True) -> None:
        """
        Convert the signal and uncertainties to new units.
        """
        _validate_uncertainties(self)
        try:
            new_units = ureg.Unit(new_units)  # ensure new_units is a pint.Unit
        except pint.errors.UndefinedUnitError as e:
            raise ValueError(f"Invalid unit provided: {new_units}.") from e

        if not isinstance(new_units, ureg.Unit):
            raise TypeError(f"new_units must be a pint.Unit, got {type(new_units)}.")

        if not self.units.is_compatible_with(new_units):
            raise ValueError(f"""
              Cannot convert from {self.units} to {new_units}. Units are not compatible.
            """)

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
        signal = np.asanyarray(self.signal * cfact)
        uncertainties = {key: np.asanyarray(value * abs(cfact)) for key, value in self.uncertainties.items()}
        self.signal = signal
        self.uncertainties = uncertainties
        self.units = new_units

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
        _validate_uncertainties(self)
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
        _validate_uncertainties(self)
        new = BaseData(
            signal=self.signal,
            units=self.units,
            uncertainties=self.uncertainties,
            weights=self.weights,
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


def squeeze_basedata_element(element: BaseData) -> BaseData:
    """Squeeze: remove single-dimensional entries from the shape of the signal."""
    signal, uncertainties = _unary_inputs(element)
    result = BaseData(
        signal=np.squeeze(signal),
        units=element.units,
        uncertainties={k: np.squeeze(v) for k, v in uncertainties.items()},
    )
    return _inherit_metadata(element, result)


def negate_basedata_element(element: BaseData) -> BaseData:
    """Negate a BaseData element with uncertainty and units propagation."""
    signal, uncertainties = _unary_inputs(element)
    result = BaseData(
        signal=-signal,
        units=element.units,
        uncertainties=uncertainties,
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
        input_units=ureg.dimensionless,
        func=np.log,
        dfunc=lambda x: 1.0 / x,
        out_units=ureg.dimensionless,
        domain=lambda x: x > 0,
    )


def exp_basedata_element(element: BaseData) -> BaseData:
    """Exponential: y = exp(x), σ_y ≈ exp(x) σ_x. Argument should be dimensionless."""
    return _unary_basedata_op(
        element=element,
        input_units=ureg.dimensionless,
        func=np.exp,
        dfunc=np.exp,
        out_units=ureg.dimensionless,
    )


def sin_basedata_element(element: BaseData) -> BaseData:
    """Sine: y = sin(x), σ_y ≈ |cos(x)| σ_x. x in radians."""
    return _unary_basedata_op(
        element=element,
        input_units=ureg.radian,
        func=np.sin,
        dfunc=np.cos,
        out_units=ureg.dimensionless,
    )


def cos_basedata_element(element: BaseData) -> BaseData:
    """Cosine: y = cos(x), σ_y ≈ |sin(x)| σ_x. x in radians."""
    return _unary_basedata_op(
        element=element,
        input_units=ureg.radian,
        func=np.cos,
        dfunc=np.sin,  # derivative is -sin(x), abs removes sign
        out_units=ureg.dimensionless,
    )


def tan_basedata_element(element: BaseData) -> BaseData:
    """Tangent: y = tan(x), σ_y ≈ |1/cos^2(x)| σ_x. x in radians."""
    return _unary_basedata_op(
        element=element,
        input_units=ureg.radian,
        func=np.tan,
        dfunc=lambda x: 1.0 / (np.cos(x) ** 2),
        out_units=ureg.dimensionless,
    )


def arcsin_basedata_element(element: BaseData) -> BaseData:
    """Arcsin: y = arcsin(x), σ_y ≈ |1/sqrt(1-x^2)| σ_x. x dimensionless, |x| <= 1."""
    return _unary_basedata_op(
        element=element,
        input_units=ureg.dimensionless,
        func=np.arcsin,
        dfunc=lambda x: 1.0 / np.sqrt(1.0 - x**2),
        out_units=ureg.radian,
        domain=lambda x: np.abs(x) <= 1.0,
    )


def arccos_basedata_element(element: BaseData) -> BaseData:
    """Arccos: y = arccos(x), σ_y ≈ |1/sqrt(1-x^2)| σ_x. x dimensionless, |x| <= 1."""
    return _unary_basedata_op(
        element=element,
        input_units=ureg.dimensionless,
        func=np.arccos,
        dfunc=lambda x: 1.0 / np.sqrt(1.0 - x**2),  # abs removes the sign
        out_units=ureg.radian,
        domain=lambda x: np.abs(x) <= 1.0,
    )


def arctan_basedata_element(element: BaseData) -> BaseData:
    """Arctan: y = arctan(x), σ_y ≈ |1/(1+x^2)| σ_x. x dimensionless."""
    return _unary_basedata_op(
        element=element,
        input_units=ureg.dimensionless,
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
