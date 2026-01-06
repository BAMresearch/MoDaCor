# SPDX-License-Identifier: BSD-3-Clause
# /usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

__coding__ = "utf-8"
__authors__ = ["Brian R. Pauw"]
__copyright__ = "Copyright 2026, The MoDaCor team"
__date__ = "06/01/2026"
__status__ = "Development"  # "Development", "Production"

__version__ = "20260106.1"
__all__ = ["unit_vec3", "require_scalar", "prepare_static_scalar"]

from typing import Tuple

import numpy as np
import pint

from modacor import ureg
from modacor.dataclasses.basedata import BaseData


def unit_vec3(v: Tuple[float, float, float] | np.ndarray, *, name: str = "vector") -> np.ndarray:
    """Normalize a 3-vector to unit length."""
    v = np.asarray(v, dtype=float).reshape(3)
    n = float(np.linalg.norm(v))
    if n == 0.0:
        raise ValueError(f"{name} must be non-zero")
    return v / n


def require_scalar(name: str, bd: BaseData) -> BaseData:
    """Ensure a BaseData is scalar; returns a squeezed copy with RoD=0."""
    out = bd.squeeze().copy()
    if np.size(out.signal) != 1:
        raise ValueError(f"{name} must be scalar (size==1). Got shape={np.shape(out.signal)}.")
    out.rank_of_data = 0
    return out


def prepare_static_scalar(
    bd: BaseData,
    *,
    require_units: pint.Unit = ureg.m,
    uncertainty_key: str = "static_config_jitter",
) -> BaseData:
    """
    Reduce a possibly-array BaseData to a scalar via weighted mean (bd.weights) and
    assign uncertainty as standard error of the mean (SEM).

    - If bd is already scalar: squeeze+RoD=0 and return.
    - If bd.weights is None: uniform weights are assumed.

    Notes
    -----
    SEM here uses:
      - weighted mean
      - weighted variance (about mean)
      - effective sample size n_eff = (sum(w)^2) / sum(w^2)
      - sem = sqrt(var) / sqrt(n_eff)
    """
    if not bd.units.is_compatible_with(require_units):
        raise ValueError(f"Value must be in {require_units}, got {bd.units}")

    # scalar passthrough
    if np.size(bd.signal) == 1:
        out = bd.squeeze().copy()
        out.rank_of_data = 0
        return out

    x = np.asarray(bd.signal, dtype=float).ravel()

    # --- robust weights handling ---
    if bd.weights is None:
        w = np.ones_like(x)
    else:
        w_raw = np.asarray(bd.weights, dtype=float)

        # allow scalar/length-1 weights
        if w_raw.size == 1:
            w = np.full_like(x, float(w_raw.reshape(-1)[0]))
        else:
            # allow broadcastable weights (e.g. (5,1,1,1) vs (5,))
            try:
                w = np.broadcast_to(w_raw, np.shape(bd.signal)).ravel()
            except ValueError as e:
                raise ValueError(
                    f"weights shape {w_raw.shape} does not match signal shape {np.shape(bd.signal)}"
                ) from e

        if w.size != x.size:
            raise ValueError(f"weights size {w.size} does not match signal size {x.size}")

    wsum = float(np.sum(w))
    if wsum <= 0:
        raise ValueError("weights must sum to > 0")

    mean = float(np.sum(w * x) / wsum)

    # effective N for SEM (works for equal weights too)
    n_eff = float((wsum**2) / np.sum(w**2))

    # weighted population variance about the weighted mean
    var = float(np.sum(w * (x - mean) ** 2) / wsum)
    sem = float(np.sqrt(var) / np.sqrt(n_eff))

    return BaseData(
        signal=np.array(mean, dtype=float),
        units=bd.units,
        uncertainties={uncertainty_key: np.array(sem, dtype=float)},
        rank_of_data=0,
    )
