# SPDX-License-Identifier: BSD-3-Clause
# /usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

__coding__ = "utf-8"
__authors__ = ["Brian R. Pauw"]  # add names to the list as appropriate
__copyright__ = "Copyright 2025, The MoDaCor team"
__date__ = "16/11/2025"
__status__ = "Development"  # "Development", "Production"

__all__ = ["WeightedAverage"]
__version__ = "20251116.1"

from pathlib import Path
from typing import Any

import numpy as np

from modacor.dataclasses.basedata import BaseData
from modacor.dataclasses.databundle import DataBundle
from modacor.dataclasses.process_step import ProcessStep
from modacor.dataclasses.process_step_describer import ProcessStepDescriber


class WeightedAverage(ProcessStep):
    """
    Compute a (possibly weighted) average of a BaseData signal over one or more axes,
    propagating uncertainties.

    For each uncertainty key `k`, assumes uncorrelated errors:

        μ = Σ w_i x_i / Σ w_i
        σ_μ^2 = Σ (w_i^2 σ_i^2) / (Σ w_i)^2

    NaN handling:
        - If nan_policy == 'omit', NaNs in `signal` (and their σ) are ignored.
        - If nan_policy == 'propagate', NaNs behave like in plain numpy: if any NaN is
          present along the reduced axes, the result becomes NaN.
    """

    documentation = ProcessStepDescriber(
        calling_name="Weighted average over axes",
        calling_id="WeightedAverage",
        calling_module_path=Path(__file__),
        calling_version=__version__,
        required_data_keys=["signal"],
        works_on={"signal": ["signal", "uncertainties", "units", "weights"]},
        calling_arguments={
            # Axis or axes to reduce. Can be int, list/tuple of ints, or None (reduce all).
            "axes": None,
            # Use BaseData.weights as weights (default) or do unweighted mean (False → equal weights).
            "use_weights": True,
            # 'omit' → ignore NaNs (nanmean-style); 'propagate' → NaNs propagate (mean-style).
            "nan_policy": "omit",
        },
        step_keywords=["average", "mean", "weighted", "nanmean", "reduce", "axis"],
        step_doc=(
            "Compute (default weighted) mean of the BaseData signal over the given axes, "
            "with proper uncertainty propagation."
        ),
        step_reference="DOI 10.1088/0953-8984/25/38/383201",
        step_note=(
            "This step reduces the dimensionality of the signal by averaging over one or more axes. "
            "Units are preserved; axes metadata is currently not adjusted and is left empty on the result."
        ),
    )

    # ---------------------------- helpers ---------------------------------

    @staticmethod
    def _normalize_axes(axes: Any) -> int | tuple[int, ...] | None:
        """
        Normalize configuration 'axes' into a numpy-compatible axis argument.
        Allowed:
          - None           → reduce over all axes
          - int            → single axis
          - list/tuple[int] → tuple of axes
        """
        if axes is None:
            return None
        if isinstance(axes, int):
            return axes
        # list/tuple of ints
        return tuple(int(a) for a in axes)

    @staticmethod
    def _weighted_mean_with_uncertainty(
        bd: BaseData,
        axis: int | tuple[int, ...] | None,
        use_weights: bool,
        nan_policy: str,
    ) -> BaseData:
        x = np.asarray(bd.signal, dtype=float)

        if use_weights:
            w = np.asarray(bd.weights, dtype=float)
            w = np.broadcast_to(w, x.shape)
        else:
            w = np.ones_like(x, dtype=float)

        # NaN handling
        if nan_policy == "omit":
            mask = np.isnan(x) | np.isnan(w)
            x_eff = np.where(mask, 0.0, x)
            w_eff = np.where(mask, 0.0, w)
        elif nan_policy == "propagate":
            mask = np.zeros_like(x, dtype=bool)
            x_eff = x
            w_eff = w
        else:
            raise ValueError(f"Invalid nan_policy: {nan_policy!r}. Use 'omit' or 'propagate'.")

        # Weighted sums
        w_sum = np.sum(w_eff, axis=axis)
        wx_sum = np.sum(w_eff * x_eff, axis=axis)

        # Safe denominator: NaN where w_sum == 0
        denom = np.where(w_sum == 0, np.nan, w_sum)

        # μ = Σ w_i x_i / Σ w_i
        mean = wx_sum / denom

        uncertainties_out: dict[str, np.ndarray] = {}
        for key, err in bd.uncertainties.items():
            err_arr = np.asarray(err, dtype=float)
            err_arr = np.broadcast_to(err_arr, x.shape)

            if nan_policy == "omit":
                err_arr_eff = np.where(mask, 0.0, err_arr)
            else:
                err_arr_eff = err_arr

            var_sum = np.sum((w_eff**2) * (err_arr_eff**2), axis=axis)

            # σ_μ = sqrt(Σ w_i^2 σ_i^2) / Σ w_i, same denom as mean
            sigma = np.sqrt(var_sum) / denom

            uncertainties_out[key] = sigma

        return BaseData(
            signal=mean,
            units=bd.units,
            uncertainties=uncertainties_out,
            weights=np.array(1.0),
        )

    # ---------------------------- main API ---------------------------------

    def calculate(self) -> dict[str, DataBundle]:
        """
        Perform the weighted average over configured axes for each selected DataBundle.
        """
        axis = self._normalize_axes(self.configuration.get("axes"))
        use_weights = bool(self.configuration.get("use_weights", True))
        nan_policy = self.configuration.get("nan_policy", "omit")

        output: dict[str, DataBundle] = {}

        for key in self.configuration["with_processing_keys"]:
            databundle: DataBundle = self.processing_data.get(key)
            bd: BaseData = databundle["signal"]

            averaged = self._weighted_mean_with_uncertainty(
                bd=bd,
                axis=axis,
                use_weights=use_weights,
                nan_policy=nan_policy,
            )

            # Replace the signal with the reduced BaseData
            databundle["signal"] = averaged
            output[key] = databundle

        return output
