# SPDX-License-Identifier: BSD-3-Clause
# /usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Dict, List, Tuple

__coding__ = "utf-8"
__authors__ = ["Brian R. Pauw"]
__copyright__ = "Copyright 2025, The MoDaCor team"
__date__ = "30/11/2025"
__status__ = "Development"  # "Development", "Production"

__version__ = "20251130.1"
__all__ = ["IndexedAverager"]

from pathlib import Path

import numpy as np

from modacor import ureg
from modacor.dataclasses.basedata import BaseData
from modacor.dataclasses.databundle import DataBundle
from modacor.dataclasses.messagehandler import MessageHandler
from modacor.dataclasses.process_step import ProcessStep
from modacor.dataclasses.process_step_describer import ProcessStepDescriber

logger = MessageHandler(name=__name__)


class IndexedAverager(ProcessStep):
    """
    Perform averaging of signal using precomputed per-pixel bin indices.

    This module expects that a previous step (e.g. IndexPixels) has produced a
    "pixel_index" BaseData map containing, for each pixel, the bin index it
    belongs to (or -1 for "not participating").

    It then:
      - Combines the per-pixel signal into bin-averaged values using optional
        weights.
      - Computes a weighted mean Q per bin.
      - Computes a weighted circular mean Psi per bin.
      - Propagates per-pixel uncertainties on signal and Q to the bin-mean.
      - Estimates a bin-level SEM ("SEM" key) from the scatter of the signal for Q, Psi and signal.

    Inputs (from the databundle selected via with_processing_keys)
    --------------------------------------------------------------
    - "signal": BaseData
    - "Q": BaseData
    - "Psi": BaseData
    - "pixel_index": BaseData
         same spatial rank and shape as (at least) the signal data.
    - "Mask": BaseData (optional)
         boolean mask, True meaning "masked" (pixel ignored).

    Configuration
    -------------
    with_processing_keys : list[str]
        Databundle key(s) to work on. If None and there is exactly one
        databundle, that one is used.

    averaging_direction : {"radial", "azimuthal"}, default "azimuthal"
        Only used to decide which axis is attached to the output signal:
        - "azimuthal": signal.axes[0] will reference Q_1d.
        - "radial":    signal.axes[0] will reference Psi_1d.
        The underlying binning is entirely determined by "pixel_index".

    use_signal_weights : bool, default True
        If True, multiply per-pixel weights into the total weight.

    use_signal_uncertainty_weights : bool, default False
        If True, use 1 / sigma^2 (for the specified key) as an additional
        factor in the weights.

    uncertainty_weight_key : str | None, default None
        Uncertainty key in signal.uncertainties to use when
        use_signal_uncertainty_weights is True. Must be provided and present
        in signal.uncertainties in that case.

    Outputs (returned from calculate())
    -----------------------------------
    For each key in with_processing_keys, the corresponding databundle will
    be updated with 1D BaseData:

    - "signal": BaseData
        Bin-averaged signal as 1D array (length n_bins).
        Units: same as input signal.units.
        uncertainties:
          * For each original signal uncertainty key 'k', a propagated sigma
            for the bin mean under that key.
          * An additional key "SEM" with a bin-level standard error on the
            mean derived from the weighted scatter of the signal values.

    - "Q": BaseData
        Weighted mean Q per bin (length n_bins).
        units: same as input Q.units.
        uncertainties:
          * For each original Q uncertainty key 'k', propagated sigma on the
            bin mean for that key.

    - "Psi": BaseData
        Weighted circular mean of Psi per bin (length n_bins).
        units: same as input Psi.units.
        uncertainties:
          * For each original Psi uncertainty key 'k', propagated sigma on the
            bin mean for that key (using linear propagation on angles).

    The original 2D/1D "pixel_index" and optional "Mask" remain present in
    the databundle, enabling further inspection or reuse.
    """

    documentation = ProcessStepDescriber(
        calling_name="Indexed Averager",
        calling_id="IndexedAverager",
        calling_module_path=Path(__file__),
        calling_version=__version__,
        required_data_keys=["signal", "Q", "Psi", "pixel_index"],
        required_arguments=[
            "with_processing_keys",
            "output_processing_key",
            "averaging_direction",
            "use_signal_weights",
            "use_signal_uncertainty_weights",
            "uncertainty_weight_key",
        ],
        default_configuration={
            "with_processing_keys": None,
            "output_processing_key": None,  # currently unused
            "averaging_direction": "azimuthal",
            "use_signal_weights": True,
            "use_signal_uncertainty_weights": False,
            "uncertainty_weight_key": None,
        },
        argument_specs={
            "with_processing_keys": {
                "type": (str, list, type(None)),
                "required": False,
                "doc": "ProcessingData key or list of keys to average.",
            },
            "output_processing_key": {
                "type": (str, type(None)),
                "required": False,
                "doc": "Optional output key override (currently unused).",
            },
            "averaging_direction": {
                "type": str,
                "required": True,
                "doc": "Averaging direction: 'radial' or 'azimuthal'.",
            },
            "use_signal_weights": {
                "type": bool,
                "required": False,
                "doc": "Use BaseData weights when averaging signal.",
            },
            "use_signal_uncertainty_weights": {
                "type": bool,
                "required": False,
                "doc": "Use signal uncertainty as weights.",
            },
            "uncertainty_weight_key": {
                "type": (str, type(None)),
                "required": False,
                "doc": "Uncertainty key to use as weights if enabled.",
            },
        },
        modifies={
            # We overwrite 'signal', 'Q', 'Psi' with their 1D binned versions.
            "signal": ["signal", "uncertainties"],
            "Q": ["signal", "uncertainties"],
            "Psi": ["signal", "uncertainties"],
        },
        step_keywords=[
            "radial",
            "azimuthal",
            "averaging",
            "binning",
            "scattering",
        ],
        step_doc="Average signal and geometry using precomputed pixel bin indices.",
        step_reference="DOI 10.1088/0953-8984/25/38/383201",
        step_note=(
            "IndexedAverager expects a 'pixel_index' map from a previous "
            "IndexPixels step and performs per-bin weighted means of signal, "
            "Q and Psi, including uncertainty propagation."
        ),
    )

    def __attrs_post_init__(self) -> None:
        super().__attrs_post_init__()

    # ------------------------------------------------------------------
    # Helper: normalise with_processing_keys to a list
    # ------------------------------------------------------------------
    def _normalised_keys(self) -> List[str]:
        """
        Normalise with_processing_keys into a non-empty list of strings.

        If configuration value is None and exactly one databundle is present
        in processing_data, that key is returned as the single entry.
        """
        if self.processing_data is None:
            raise RuntimeError("IndexedAverager: processing_data is None in _normalised_keys.")

        cfg_value = self.configuration.get("with_processing_keys", None)

        # None → use only key if exactly one available
        if cfg_value is None:
            if len(self.processing_data) == 0:
                raise ValueError("IndexedAverager: with_processing_keys is None and processing_data is empty.")
            if len(self.processing_data) == 1:
                only_key = next(iter(self.processing_data.keys()))
                logger.info(
                    "IndexedAverager: with_processing_keys not set; using the only key %r.",
                    only_key,
                )
                return [only_key]
            raise ValueError(
                "IndexedAverager: with_processing_keys is None but multiple "
                "databundles are present. Please specify a key or list of keys."
            )

        # Single string
        if isinstance(cfg_value, str):
            return [cfg_value]

        # Iterable of strings
        try:
            keys = list(cfg_value)
        except TypeError as exc:  # not iterable
            raise ValueError(
                "IndexedAverager: with_processing_keys must be a string, an iterable of strings, or None."
            ) from exc

        if not keys:
            raise ValueError("IndexedAverager: with_processing_keys is an empty iterable.")

        for k in keys:
            if not isinstance(k, str):
                raise ValueError(f"IndexedAverager: all entries in with_processing_keys must be strings, got {k!r}.")
        return keys

    # ------------------------------------------------------------------
    # Helper: validate geometry, signal and pixel_index for a databundle
    # ------------------------------------------------------------------
    def _validate_inputs(
        self,
        databundle: DataBundle,
    ) -> Tuple[BaseData, BaseData, BaseData, BaseData, BaseData | None, Tuple[int, ...]]:
        """
        Validate presence and shapes of signal, Q, Psi, pixel_index
        (and optional Mask) for a given databundle.

        Returns:
            signal_bd, q_bd, psi_bd, pix_bd, mask_bd_or_None, spatial_shape
        """
        try:
            signal_bd: BaseData = databundle["signal"]
            q_bd: BaseData = databundle["Q"]
            psi_bd: BaseData = databundle["Psi"]
            pix_bd: BaseData = databundle["pixel_index"]
        except KeyError as exc:
            raise KeyError(
                "IndexedAverager: databundle missing required keys 'signal', 'Q', 'Psi', or 'pixel_index'."
            ) from exc

        spatial_shape: Tuple[int, ...] = tuple(signal_bd.shape)
        if q_bd.shape != spatial_shape:
            raise ValueError(f"IndexedAverager: Q shape {q_bd.shape} does not match signal shape {spatial_shape}.")
        if psi_bd.shape != spatial_shape:
            raise ValueError(f"IndexedAverager: Psi shape {psi_bd.shape} does not match signal shape {spatial_shape}.")
        if pix_bd.shape != spatial_shape:
            raise ValueError(
                f"IndexedAverager: pixel_index shape {pix_bd.shape} does not match signal shape {spatial_shape}."
            )

        mask_bd: BaseData | None = None
        # Optional mask: we accept 'Mask' or 'mask'
        if "Mask" in databundle:
            mask_bd = databundle["Mask"]
        elif "mask" in databundle:
            mask_bd = databundle["mask"]

        if mask_bd is not None and mask_bd.shape != spatial_shape:
            raise ValueError(
                f"IndexedAverager: Mask shape {mask_bd.shape} does not match signal shape {spatial_shape}."
            )

        return signal_bd, q_bd, psi_bd, pix_bd, mask_bd, spatial_shape

    # ------------------------------------------------------------------
    # Helper: core binning/averaging logic
    # ------------------------------------------------------------------
    @staticmethod
    def _compute_bin_averages(  # noqa: C901 -- complexity # TODO: reduce complexity after testing with index_pixels
        signal_bd: BaseData,
        q_bd: BaseData,
        psi_bd: BaseData,
        pix_bd: BaseData,
        mask_bd: BaseData | None,
        use_signal_weights: bool,
        use_signal_uncertainty_weights: bool,
        uncertainty_weight_key: str | None,
    ) -> Tuple[BaseData, BaseData, BaseData]:
        """
        Core binning logic: produce 1D BaseData for signal, Q, Psi.

        All inputs are assumed to have identical shapes.
        """

        # Flatten arrays
        sig_full = np.asarray(signal_bd.signal, dtype=float).ravel()
        q_full = np.asarray(q_bd.signal, dtype=float).ravel()
        psi_full = np.asarray(psi_bd.signal, dtype=float).ravel()

        pix_flat = np.asarray(pix_bd.signal, dtype=float).ravel().astype(int)

        if sig_full.size == 0:
            raise ValueError("IndexedAverager: signal array is empty.")

        if not np.all(np.isfinite(pix_flat) | (pix_flat == -1)):
            logger.warning("IndexedAverager: pixel_index contains non-finite entries; treating them as -1.")
            pix_flat[~np.isfinite(pix_flat)] = -1

        # Base validity: a pixel participates if index >= 0
        valid = pix_flat >= 0

        # Apply optional mask: True means "masked" → exclude
        if mask_bd is not None:
            mask_flat = np.asarray(mask_bd.signal, dtype=bool).ravel()
            if mask_flat.shape != pix_flat.shape:
                raise ValueError("IndexedAverager: Mask shape does not match pixel_index.")
            valid &= ~mask_flat

        # Exclude non-finite signal / Q / Psi
        valid &= np.isfinite(sig_full) & np.isfinite(q_full) & np.isfinite(psi_full)

        if not np.any(valid):
            raise ValueError("IndexedAverager: no valid pixels to average.")

        bin_idx = pix_flat[valid]
        sig_valid = sig_full[valid]
        q_valid = q_full[valid]
        psi_valid = psi_full[valid]

        n_bins = int(bin_idx.max()) + 1
        if n_bins <= 0:
            raise ValueError("IndexedAverager: inferred n_bins is non-positive.")

        # ------------------------------------------------------------------
        # 1. Combined weights
        # ------------------------------------------------------------------
        w = np.ones_like(sig_full, dtype=float)

        if use_signal_weights:
            w_bd = np.asarray(signal_bd.weights, dtype=float)
            try:
                w_bd_full = np.broadcast_to(w_bd, signal_bd.shape).ravel()
            except ValueError as exc:
                raise ValueError("IndexedAverager: could not broadcast signal.weights to signal shape.") from exc
            w *= w_bd_full

        if use_signal_uncertainty_weights:
            if uncertainty_weight_key is None:
                raise ValueError(
                    "IndexedAverager: use_signal_uncertainty_weights=True but uncertainty_weight_key is None."
                )
            if uncertainty_weight_key not in signal_bd.uncertainties:
                raise KeyError(
                    f"IndexedAverager: uncertainty key {uncertainty_weight_key!r} not found in signal.uncertainties."  # noqa: E713
                )

            sigma_u = np.asarray(signal_bd.uncertainties[uncertainty_weight_key], dtype=float)
            try:
                sigma_full = np.broadcast_to(sigma_u, signal_bd.shape).ravel()
            except ValueError as exc:
                raise ValueError(
                    "IndexedAverager: could not broadcast chosen uncertainty array to signal shape."
                ) from exc

            var_full = sigma_full**2
            # Only accept strictly positive finite variances for weighting
            valid_sigma = np.isfinite(var_full) & (var_full > 0.0)
            if not np.any(valid_sigma & valid):
                raise ValueError(
                    "IndexedAverager: no pixels have positive finite variance under the chosen uncertainty_weight_key."
                )

            # Pixels with non-positive/NaN variance are effectively dropped
            valid &= valid_sigma
            var_full[~valid_sigma] = np.inf  # avoid division by zero
            w *= 1.0 / var_full

        # Recompute valid slice after potential tightening due to uncertainty weights
        valid_idx = np.where(valid)[0]
        bin_idx = pix_flat[valid_idx]
        sig_valid = sig_full[valid_idx]
        q_valid = q_full[valid_idx]
        psi_valid = psi_full[valid_idx]
        w_valid = w[valid_idx]

        if not np.any(w_valid > 0.0):
            raise ValueError("IndexedAverager: all weights are zero; cannot compute averages.")

        # Clamp negative weights to zero (should not happen, but be robust)
        w_valid = np.clip(w_valid, a_min=0.0, a_max=None)

        # ------------------------------------------------------------------
        # 2. Weighted sums for signal and Q
        # ------------------------------------------------------------------
        sum_w = np.bincount(bin_idx, weights=w_valid, minlength=n_bins)
        sum_wx = np.bincount(bin_idx, weights=w_valid * sig_valid, minlength=n_bins)
        sum_wq = np.bincount(bin_idx, weights=w_valid * q_valid, minlength=n_bins)

        with np.errstate(divide="ignore", invalid="ignore"):
            mean_signal = np.full(n_bins, np.nan, dtype=float)
            mean_q = np.full(n_bins, np.nan, dtype=float)

            positive = sum_w > 0.0
            mean_signal[positive] = sum_wx[positive] / sum_w[positive]
            mean_q[positive] = sum_wq[positive] / sum_w[positive]

        # ------------------------------------------------------------------
        # 3. Weighted circular mean for Psi
        # ------------------------------------------------------------------
        psi_unit = psi_bd.units

        # Convert Psi to radians for trigonometric operations
        cf_to_rad = ureg.radian.m_from(psi_unit)
        psi_rad_valid = psi_valid * cf_to_rad

        cos_psi = np.cos(psi_rad_valid)
        sin_psi = np.sin(psi_rad_valid)

        sum_wcos = np.bincount(bin_idx, weights=w_valid * cos_psi, minlength=n_bins)
        sum_wsin = np.bincount(bin_idx, weights=w_valid * sin_psi, minlength=n_bins)

        with np.errstate(divide="ignore", invalid="ignore"):
            mean_cos = np.full(n_bins, np.nan, dtype=float)
            mean_sin = np.full(n_bins, np.nan, dtype=float)
            mean_psi_rad = np.full(n_bins, np.nan, dtype=float)

            positive = sum_w > 0.0
            mean_cos[positive] = sum_wcos[positive] / sum_w[positive]
            mean_sin[positive] = sum_wsin[positive] / sum_w[positive]

            mean_psi_rad[positive] = np.arctan2(mean_sin[positive], mean_cos[positive])

        # Convert back to original Psi units
        cf_from_rad = psi_unit.m_from(ureg.radian)
        mean_psi = mean_psi_rad * cf_from_rad

        # ------------------------------------------------------------------
        # 4. Propagate uncertainties on signal, Q, Psi
        # ------------------------------------------------------------------
        sig_unc_binned: Dict[str, np.ndarray] = {}
        q_unc_binned: Dict[str, np.ndarray] = {}
        psi_unc_binned: Dict[str, np.ndarray] = {}

        # Helper for propagation: sigma_mean = sqrt(sum (w^2 * sigma^2)) / sum_w
        def _propagate_uncertainties(unc_dict: Dict[str, np.ndarray], ref_bd: BaseData) -> Dict[str, np.ndarray]:
            result: Dict[str, np.ndarray] = {}
            for key, arr in unc_dict.items():
                arr_full = np.asarray(arr, dtype=float)
                try:
                    arr_full = np.broadcast_to(arr_full, ref_bd.shape).ravel()
                except ValueError as exc:
                    raise ValueError(
                        f"IndexedAverager: could not broadcast uncertainty[{key!r}] to reference shape."
                    ) from exc

                arr_valid = arr_full[valid_idx]
                var_valid = arr_valid**2

                sum_w2_var = np.bincount(bin_idx, weights=(w_valid**2) * var_valid, minlength=n_bins)

                sigma = np.full(n_bins, np.nan, dtype=float)
                with np.errstate(divide="ignore", invalid="ignore"):
                    positive = sum_w > 0.0
                    sigma[positive] = np.sqrt(sum_w2_var[positive]) / sum_w[positive]

                result[key] = sigma
            return result

        if signal_bd.uncertainties:
            sig_unc_binned.update(_propagate_uncertainties(signal_bd.uncertainties, signal_bd))
        if q_bd.uncertainties:
            q_unc_binned.update(_propagate_uncertainties(q_bd.uncertainties, q_bd))
        if psi_bd.uncertainties:
            psi_unc_binned.update(_propagate_uncertainties(psi_bd.uncertainties, psi_bd))

        # ------------------------------------------------------------------
        # 5. SEM from scatter of signal ("SEM" key)
        # ------------------------------------------------------------------
        # Effective sample size:
        sum_w2 = np.bincount(bin_idx, weights=w_valid**2, minlength=n_bins)
        with np.errstate(divide="ignore", invalid="ignore"):
            N_eff = np.full(n_bins, np.nan, dtype=float)
            positive = sum_w2 > 0.0
            N_eff[positive] = (sum_w[positive] ** 2) / sum_w2[positive]

        # Weighted variance around mean
        # dev_i = x_i - mean_signal[bin_idx_i]
        mean_signal_per_pixel = mean_signal[bin_idx]
        dev_valid = sig_valid - mean_signal_per_pixel

        sum_w_dev2 = np.bincount(bin_idx, weights=w_valid * (dev_valid**2), minlength=n_bins)

        with np.errstate(divide="ignore", invalid="ignore"):
            var_spread = np.full(n_bins, np.nan, dtype=float)
            sem_spread = np.full(n_bins, np.nan, dtype=float)

            valid_bins = (sum_w > 0.0) & np.isfinite(N_eff) & (N_eff > 1.0)

            var_spread[valid_bins] = sum_w_dev2[valid_bins] / sum_w[valid_bins]
            sem_spread[valid_bins] = np.sqrt(var_spread[valid_bins] / N_eff[valid_bins])

        # Add SEM as a dedicated uncertainty key on the binned signal
        sig_unc_binned["SEM"] = sem_spread

        # ------------------------------------------------------------------
        # 6. Build output BaseData objects
        # ------------------------------------------------------------------
        # 1D signal
        signal_1d = BaseData(
            signal=mean_signal,
            units=signal_bd.units,
            uncertainties=sig_unc_binned,
            weights=np.ones_like(mean_signal, dtype=float),
            axes=[],  # will be filled based on averaging_direction in caller
            rank_of_data=1,
        )

        # 1D Q
        Q_1d = BaseData(
            signal=mean_q,
            units=q_bd.units,
            uncertainties=q_unc_binned,
            weights=np.ones_like(mean_q, dtype=float),
            axes=[],
            rank_of_data=1,
        )

        # 1D Psi
        Psi_1d = BaseData(
            signal=mean_psi,
            units=psi_bd.units,
            uncertainties=psi_unc_binned,
            weights=np.ones_like(mean_psi, dtype=float),
            axes=[],
            rank_of_data=1,
        )

        return signal_1d, Q_1d, Psi_1d

    # ------------------------------------------------------------------
    # prepare_execution: nothing heavy here for now
    # ------------------------------------------------------------------
    def prepare_execution(self) -> None:
        """
        For IndexedAverager, there is no heavy geometry to precompute.

        All binning work depends on the per-frame signal, so we perform the
        averaging inside calculate(). This method only validates that the
        configuration is at least minimally sensible.
        """
        # ensure configuration keys are present; defaults are already set by __attrs_post_init__
        direction = str(self.configuration.get("averaging_direction", "azimuthal")).lower()
        if direction not in ("radial", "azimuthal"):
            raise ValueError(
                f"IndexedAverager: averaging_direction must be 'radial' or 'azimuthal', got {direction!r}."
            )

    # ------------------------------------------------------------------
    # calculate: perform per-key averaging using pixel_index
    # ------------------------------------------------------------------
    def calculate(self) -> Dict[str, DataBundle]:
        """
        For each databundle in with_processing_keys, perform the binning /
        averaging using the precomputed pixel_index map and return updated
        DataBundles containing 1D 'signal', 'Q', and 'Psi' BaseData.
        """
        output: Dict[str, DataBundle] = {}

        if self.processing_data is None:
            logger.warning("IndexedAverager: processing_data is None in calculate; nothing to do.")
            return output

        keys = self._normalised_keys()
        use_signal_weights = bool(self.configuration.get("use_signal_weights", True))
        use_unc_w = bool(self.configuration.get("use_signal_uncertainty_weights", False))
        uncertainty_weight_key = self.configuration.get("uncertainty_weight_key", None)
        direction = str(self.configuration.get("averaging_direction", "azimuthal")).lower()

        for key in keys:
            if key not in self.processing_data:
                logger.warning(
                    "IndexedAverager: processing_data has no entry for key=%r; skipping.",
                    key,
                )
                continue

            databundle = self.processing_data[key]

            (
                signal_bd,
                q_bd,
                psi_bd,
                pix_bd,
                mask_bd,
                _spatial_shape,
            ) = self._validate_inputs(databundle)

            # Compute binned 1D BaseData
            signal_1d, Q_1d, Psi_1d = self._compute_bin_averages(
                signal_bd=signal_bd,
                q_bd=q_bd,
                psi_bd=psi_bd,
                pix_bd=pix_bd,
                mask_bd=mask_bd,
                use_signal_weights=use_signal_weights,
                use_signal_uncertainty_weights=use_unc_w,
                uncertainty_weight_key=uncertainty_weight_key,
            )

            # Attach axis: Q for azimuthal, Psi for radial (convention)
            if direction == "azimuthal":
                signal_1d.axes = [Q_1d]
            else:  # "radial"
                signal_1d.axes = [Psi_1d]

            db_out = DataBundle(
                {
                    "signal": signal_1d,
                    "Q": Q_1d,
                    "Psi": Psi_1d,
                    # pixel_index, Mask, etc. remain in the original databundle
                }
            )

            output[key] = db_out

        return output
