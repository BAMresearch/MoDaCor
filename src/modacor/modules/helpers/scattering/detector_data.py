# SPDX-License-Identifier: BSD-3-Clause
# /usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

__coding__ = "utf-8"
__authors__ = ["Brian R. Pauw"]
__copyright__ = "Copyright 2026, The MoDaCor team"
__date__ = "06/01/2026"
__status__ = "Development"  # "Development", "Production"

__version__ = "20261103.1"
__all__ = ["require_scalar", "prepare_static_scalar", "detector_index_basedata"]

import numpy as np
import pint

from modacor import ureg
from modacor.dataclasses.basedata import BaseData, _validate_uncertainties
from modacor.models.statistics import static_weighted_mean_sem
from modacor.models.uncertainty import with_uncertainty_estimates


def require_scalar(name: str, bd: BaseData) -> BaseData:
    """Ensure a BaseData is scalar; returns a squeezed copy with RoD=0."""
    out = bd.squeeze().copy()
    if np.size(out.signal) != 1:
        raise ValueError(f"{name} must be scalar (size==1). Got shape={np.shape(out.signal)}.")
    out.rank_of_data = 0
    return out


def detector_index_basedata(
    shape: tuple[int, ...],
    axis: int,
    *,
    uncertainty_key: str,
) -> BaseData:
    """
    Build a dimensionless detector index grid with center-of-element convention.

    Detector element indices are array coordinates, not physical units. The
    nominal coordinate is shifted by 0.5 so index 0 represents the center of the
    first detector element, and the default uncertainty is 0.5 index units.
    """
    if len(shape) == 0:
        signal = np.array(0.0, dtype=float)
    else:
        if axis < 0 or axis >= len(shape):
            raise ValueError(f"axis={axis} is outside detector shape with rank {len(shape)}.")
        grids = np.meshgrid(
            *[np.arange(n, dtype=float) + 0.5 for n in shape],
            indexing="ij",
        )
        signal = grids[axis]

    return BaseData(
        signal=signal,
        units=ureg.dimensionless,
        uncertainties={uncertainty_key: np.full_like(signal, 0.5, dtype=float)},
    )


def prepare_static_scalar(
    bd: BaseData,
    *,
    require_units: pint.Unit = ureg.m,
    uncertainty_key: str = "static_config_jitter",
) -> BaseData:
    """
    Reduce a possibly-array BaseData to a scalar via weighted mean (bd.weights) and
    retain unreduced configured components alongside the scatter SEM.

    - If bd is already scalar: squeeze+RoD=0 and return.
    - If bd.weights is None: uniform weights are assumed.
    - For an array, each configured component is the weighted mean of its sigma,
      using the same weights and weight handling as the nominal mean. Scalar and
      broadcastable sigmas are supported; a constant sigma does not shrink as
      the number of metadata samples increases.
    - Add the SEM under uncertainty_key; an existing component with that name is
      a configuration conflict. The Stage 0 producer rule folds a global-only
      propagate_to_all component into that SEM with hypot(SEM, mean sigma),
      replaces the global key, and logs the legacy fold at INFO.

    Notes
    -----
    SEM here uses:
      - weighted mean
      - weighted variance (about mean)
      - effective sample size n_eff = (sum(w)^2) / sum(w^2)
      - sem = sqrt(var) / sqrt(n_eff)
    """
    _validate_uncertainties(bd)
    if not bd.units.is_compatible_with(require_units):
        raise ValueError(f"Value must be in {require_units}, got {bd.units}")

    # scalar passthrough
    if np.size(bd.signal) == 1:
        return BaseData.from_scalar(
            bd.signal.squeeze(), bd.units, uncertainties=bd.uncertainties, weights=bd.weights.squeeze()
        )

    if uncertainty_key in bd.uncertainties:
        raise ValueError(f"Uncertainty component {uncertainty_key!r} conflicts with the scatter SEM key.")

    mean, sem = static_weighted_mean_sem(bd.signal, bd.weights)
    components = {
        name: static_weighted_mean_sem(np.broadcast_to(sigma, bd.signal.shape), bd.weights)[0]
        for name, sigma in bd.uncertainties.items()
    }

    return BaseData.from_scalar(
        signal=np.array(mean, dtype=float),
        units=bd.units,
        uncertainties=with_uncertainty_estimates(components, {uncertainty_key: sem}),
        rank_of_data=0,
    )
