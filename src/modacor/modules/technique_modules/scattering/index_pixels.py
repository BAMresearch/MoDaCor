# SPDX-License-Identifier: BSD-3-Clause
# /usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

__coding__ = "utf-8"
__authors__ = ["Brian R. Pauw"]
__copyright__ = "Copyright 2025, The MoDaCor team"
__date__ = "29/11/2025"
__status__ = "Development"  # "Development", "Production"

__version__ = "20251130.1"
__all__ = ["IndexPixels"]

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from modacor import ureg
from modacor.dataclasses.basedata import BaseData
from modacor.dataclasses.databundle import DataBundle
from modacor.dataclasses.messagehandler import MessageHandler
from modacor.dataclasses.process_step import ProcessStep
from modacor.dataclasses.process_step_describer import ProcessStepDescriber

logger = MessageHandler(name=__name__)


class IndexPixels(ProcessStep):
    """
    Compute pixel bin indices for a single dataset, for subsequent 1D averaging.

    Depending on `averaging_direction`, this step can prepare indices for:
      - azimuthal averaging (bin along Q, normal for 1D X-ray scattering curves).
      - radial averaging (bin along Psi, usually for getting orientation information);

    This step:
      - Interprets Q limits in user-specified units (q_limits_unit).
      - Interprets Psi limits in user-specified units (psi_limits_unit).
      - Builds bin edges internally (Q or Psi depending on averaging_direction).
      - For each pixel, decides which bin it belongs to, or -1 if it does
        not participate in any bin (out of range / outside ROI / non-finite).

    Inputs (from the databundle selected via with_processing_keys)
    --------------------------------------------------------------
    - "signal": BaseData   (together with its rank_of_data used for data shape)
    - "Q": BaseData        (modulus of scattering vector)
    - "Psi": BaseData      (azimuthal angle)

    This step does *not* apply the Mask. Mask is left to downstream modules
    (e.g., the averaging step), so that it can vary per frame for dynamic masking.

    Configuration
    -------------
    with_processing_keys : str | list[str] | None
        Databundle key(s) to work on. The pixel index map is computed from
        the first key and attached to all specified keys.
        If None and there is exactly one databundle, that one is used.

    averaging_direction : {"radial", "azimuthal"}, default "azimuthal"
        - "azimuthal": bins along Q, using q_min/q_max and bin_type;
        - "radial": bins along Psi (linear bins), using psi_min/psi_max.
          In this case q_min/q_max define a radial ROI mask (optional).

    q_min, q_max : float, optional
        Q limits expressed in units given by q_limits_unit.
        If omitted:
          - For "radial" + "log" binning: q_min = smallest positive finite Q;
          - Otherwise: q_min = min(Q), q_max = max(Q).
        q_min may be negative if not using "log" binning. Useful for e.g. USAXS scans.

    q_limits_unit : str or pint.Unit, optional
        Units in which q_min/q_max are defined, e.g. "1/nm".
        Defaults to the Q.units of the dataset.

    n_bins : int, default 100
        Number of bins along the averaging direction (Q or Psi).

    bin_type : {"log", "linear"}, default "log"
        - For averaging_direction="radial":
            "log" uses geometric spacing (np.geomspace);
            "linear" uses np.linspace.
        - For averaging_direction="azimuthal":
            Must be "linear" (logarithmic psi is not implemented).

    psi_min, psi_max : float, optional
        Azimuth limits expressed in psi_limits_unit.
        For averaging_direction="azimuthal":
          - These define an azimuthal mask (ROI).
          - Defaults to a full circle:
              * 0 .. 360 if psi_limits_unit is degree;
              * 0 .. 2π  if psi_limits_unit is radian.
        For averaging_direction="radial":
          - These also define the binning range along Psi.

    psi_limits_unit : str or pint.Unit, optional
        Units in which psi_min/psi_max are defined (i.e. "degree" or "radian").
        Defaults to the Psi.units of the dataset.

    Outputs (returned from calculate())
    -----------------------------------
    One DataBundle per key in with_processing_keys, each containing:

    - "pixel_index": BaseData
        signal : ndarray with same shape as the last rank_of_data ndims
                 of the chosen "signal" BaseData.
                 Each entry is an integer bin index (stored as float in
                 BaseData; will be cast back to int when used).
                 -1 means "this pixel does not participate in any bin".
        units  : dimensionless
        uncertainties : empty dict
        axes   : copied from the *last* rank_of_data axes of the original signal
        rank_of_data : same as the original signal BaseData
    """

    documentation = ProcessStepDescriber(
        calling_name="Index Pixels",
        calling_id="IndexPixels",
        calling_module_path=Path(__file__),
        calling_version=__version__,
        required_data_keys=["signal", "Q", "Psi"],
        required_arguments=[
            "with_processing_keys",
            "averaging_direction",
        ],
        default_configuration={
            "with_processing_keys": None,
            "averaging_direction": "radial",
            "q_min": None,
            "q_max": None,
            "q_limits_unit": None,
            "n_bins": 100,
            "bin_type": "log",
            "psi_min": None,
            "psi_max": None,
            "psi_limits_unit": None,
        },
        argument_specs={
            "with_processing_keys": {
                "type": (str, list, type(None)),
                "required": False,
                "doc": "ProcessingData key or list of keys to index.",
            },
            "averaging_direction": {
                "type": str,
                "required": True,
                "doc": "Averaging direction: 'radial' or 'azimuthal'.",
            },
            "q_min": {
                "type": (float, int, type(None)),
                "required": False,
                "doc": "Minimum Q value for binning.",
            },
            "q_max": {
                "type": (float, int, type(None)),
                "required": False,
                "doc": "Maximum Q value for binning.",
            },
            "q_limits_unit": {
                "type": (str, type(None)),
                "required": False,
                "doc": "Units for q_min/q_max if provided.",
            },
            "n_bins": {
                "type": int,
                "required": False,
                "doc": "Number of bins.",
            },
            "bin_type": {
                "type": str,
                "required": False,
                "doc": "Binning type: 'linear' or 'log'.",
            },
            "psi_min": {
                "type": (float, int, type(None)),
                "required": False,
                "doc": "Minimum Psi value for binning.",
            },
            "psi_max": {
                "type": (float, int, type(None)),
                "required": False,
                "doc": "Maximum Psi value for binning.",
            },
            "psi_limits_unit": {
                "type": (str, type(None)),
                "required": False,
                "doc": "Units for psi_min/psi_max if provided.",
            },
        },
        modifies={},  # nothing, we only add.
        step_keywords=[
            "radial",
            "azimuthal",
            "pixel indexing",
            "binning",
            "scattering",
        ],
        step_doc="Compute per-pixel bin indices (radial or azimuthal) for later 1D averaging.",
        step_reference="DOI 10.1088/0953-8984/25/38/383201",
        step_note=(
            "IndexPixels computes bin indices purely from geometry (Q, Psi) and "
            "user-defined limits; Mask is not used here so it can be applied "
            "per frame in downstream steps."
        ),
    )

    def __attrs_post_init__(self) -> None:
        super().__attrs_post_init__()
        # Prepared state lives in self._prepared_data.

    # ------------------------------------------------------------------
    # internal helper: normalise with_processing_keys
    # ------------------------------------------------------------------
    def _normalised_keys(self) -> Tuple[str, List[str]]:
        """
        Return (primary_key, keys_to_update).

        primary_key: the key used to compute the pixel index map.
        keys_to_update: all keys that should receive the map.
        """
        if self.processing_data is None:
            raise RuntimeError("IndexPixels: processing_data is None in _normalised_keys.")

        cfg_value = self.configuration.get("with_processing_keys", None)

        # None → use only key if exactly one available
        if cfg_value is None:
            if len(self.processing_data) == 0:
                raise ValueError("IndexPixels: with_processing_keys is None and processing_data is empty.")
            if len(self.processing_data) == 1:
                only_key = next(iter(self.processing_data.keys()))
                logger.info(
                    f"IndexPixels: with_processing_keys not set; using the only key {only_key}.",  # noqa: E702
                )
                return only_key, [only_key]
            raise ValueError(
                "IndexPixels: with_processing_keys is None but multiple "
                "databundles are present. Please specify a key or list of keys."
            )

        # Single string
        if isinstance(cfg_value, str):
            return cfg_value, [cfg_value]

        # Iterable of strings
        try:
            keys = list(cfg_value)
        except TypeError as exc:  # not iterable
            raise ValueError(
                "IndexPixels: with_processing_keys must be a string, an iterable of strings, or None."
            ) from exc

        if not keys:
            raise ValueError("IndexPixels: with_processing_keys is an empty iterable.")

        for k in keys:
            if not isinstance(k, str):
                raise ValueError("IndexPixels: all entries in with_processing_keys must be strings, got %r." % (k,))

        primary_key = keys[0]
        if len(keys) > 1:
            logger.warning(
                (
                    "IndexPixels: multiple with_processing_keys given; "
                    "pixel index map will be computed from the first (%r) and "
                    "attached to all %r."
                ),
                primary_key,
                keys,
            )
        return primary_key, keys

    # ------------------------------------------------------------------
    # internal helper: geometry / shape validation
    # ------------------------------------------------------------------
    def _validate_and_get_geometry(
        self,
        databundle: DataBundle,
    ) -> Tuple[BaseData, BaseData, BaseData, int, Tuple[int, ...], List[BaseData | None]]:
        """
        Validate signal/Q/Psi for azimuthal geometry and return:

            signal_bd, q_bd, psi_bd, RoD, spatial_shape, spatial_axes
        """
        signal_bd: BaseData = databundle["signal"]
        q_bd: BaseData = databundle["Q"]
        psi_bd: BaseData = databundle["Psi"]

        RoD: int = int(signal_bd.rank_of_data)
        if RoD not in (1, 2):
            raise ValueError(f"IndexPixels: rank_of_data must be 1 or 2 for azimuthal geometry, got {RoD}.")

        spatial_shape: Tuple[int, ...] = signal_bd.shape[-RoD:] if RoD > 0 else ()

        if q_bd.shape != spatial_shape:
            raise ValueError(f"IndexPixels: Q shape {q_bd.shape} does not match spatial shape {spatial_shape}.")
        if psi_bd.shape != spatial_shape:
            raise ValueError(f"IndexPixels: Psi shape {psi_bd.shape} does not match spatial shape {spatial_shape}.")

        if signal_bd.axes:
            spatial_axes: List[BaseData | None] = list(signal_bd.axes[-RoD:])
        else:
            spatial_axes = []

        return signal_bd, q_bd, psi_bd, RoD, spatial_shape, spatial_axes

    # ------------------------------------------------------------------
    # prepare_execution: all geometry + array work happens here
    # ------------------------------------------------------------------
    def prepare_execution(self) -> None:  # noqa: C901 # complexity issue / separation of concerns TODO: fix this later.
        """
        Prepare the pixel index map for the selected databundle.

        All heavy computations and array manipulations are done here.
        calculate() only wraps the prepared BaseData into DataBundles.
        """
        if self._prepared_data.get("pixel_index_bd") is not None:
            return

        if self.processing_data is None:
            raise RuntimeError("IndexPixels: processing_data is None in prepare_execution.")

        primary_key, keys_to_update = self._normalised_keys()
        self._prepared_data["keys_to_update"] = keys_to_update

        if primary_key not in self.processing_data:
            raise KeyError(f"IndexPixels: key {primary_key!r} not found in processing_data.")  # noqa: E713

        databundle: DataBundle = self.processing_data[primary_key]
        (
            signal_bd,
            q_bd,
            psi_bd,
            RoD,
            spatial_shape,
            spatial_axes,
        ) = self._validate_and_get_geometry(databundle)

        # Direction of averaging: "radial" or "azimuthal"
        direction = str(self.configuration.get("averaging_direction", "azimuthal")).lower()
        if direction not in ("radial", "azimuthal"):
            raise ValueError(f"IndexPixels: averaging_direction must be 'radial' or 'azimuthal', got {direction!r}.")

        # ------------------------------------------------------------------
        # 1. Resolve Q limits (mask +, for radial, binning)
        # ------------------------------------------------------------------
        q_min_cfg = self.configuration.get("q_min", None)
        q_max_cfg = self.configuration.get("q_max", None)
        n_bins = int(self.configuration.get("n_bins", 100))
        bin_type = str(self.configuration.get("bin_type", "log")).lower()

        if n_bins <= 0:
            raise ValueError(f"IndexPixels: n_bins must be positive, got {n_bins}.")

        q_limits_unit_cfg = self.configuration.get("q_limits_unit", None)
        if q_limits_unit_cfg is None:
            q_limits_unit = q_bd.units
        else:
            q_limits_unit = ureg.Unit(q_limits_unit_cfg)

        q_full = np.asarray(q_bd.signal, dtype=float)
        try:
            q_flat = q_full.ravel()
        except Exception as exc:  # noqa: BLE001
            raise ValueError("IndexPixels: could not flatten Q array.") from exc

        finite_q = q_flat[np.isfinite(q_flat)]
        if finite_q.size == 0:
            raise ValueError("IndexPixels: Q array has no finite values.")

        data_q_min = float(np.nanmin(finite_q))
        data_q_max = float(np.nanmax(finite_q))

        if direction == "azimuthal":
            # q_min/q_max define both mask and bin range
            if q_min_cfg is not None:
                q_min_val = (float(q_min_cfg) * q_limits_unit).to(q_bd.units).magnitude
            else:
                if bin_type == "log":
                    positive = finite_q[finite_q > 0.0]
                    if positive.size == 0:
                        raise ValueError("IndexPixels: cannot determine positive q_min for log binning.")
                    q_min_val = float(np.nanmin(positive))
                else:
                    q_min_val = data_q_min

            if q_max_cfg is not None:
                q_max_val = (float(q_max_cfg) * q_limits_unit).to(q_bd.units).magnitude
            else:
                q_max_val = data_q_max
        else:
            # radial: q_min/q_max are optional ROI only; ignore bin_type here
            if q_min_cfg is not None:
                q_min_val = (float(q_min_cfg) * q_limits_unit).to(q_bd.units).magnitude
            else:
                q_min_val = data_q_min

            if q_max_cfg is not None:
                q_max_val = (float(q_max_cfg) * q_limits_unit).to(q_bd.units).magnitude
            else:
                q_max_val = data_q_max

        if q_max_val <= q_min_val or not np.isfinite(q_min_val) or not np.isfinite(q_max_val):
            raise ValueError(f"IndexPixels: invalid Q range q_min={q_min_val}, q_max={q_max_val}.")

        # ------------------------------------------------------------------
        # 2. Resolve Psi limits (mask +, for azimuthal, binning)
        # ------------------------------------------------------------------
        psi_limits_unit_cfg = self.configuration.get("psi_limits_unit", None)
        if psi_limits_unit_cfg is None:
            psi_limits_unit = psi_bd.units
        else:
            psi_limits_unit = ureg.Unit(psi_limits_unit_cfg)

        psi_min_cfg = self.configuration.get("psi_min", None)
        psi_max_cfg = self.configuration.get("psi_max", None)

        if psi_min_cfg is None:
            psi_min_cfg = 0.0

        if psi_max_cfg is None:
            # Choose a default full-circle depending on psi_limits_unit
            if psi_limits_unit == ureg.degree:
                psi_max_cfg = 360.0
            elif psi_limits_unit == ureg.radian:
                psi_max_cfg = 2.0 * np.pi
            else:
                raise ValueError(
                    "IndexPixels: psi_limits_unit is neither degree nor radian "
                    "and no psi_max is specified; cannot infer a full-circle default."
                )

        psi_min_val = (float(psi_min_cfg) * psi_limits_unit).to(psi_bd.units).magnitude
        psi_max_val = (float(psi_max_cfg) * psi_limits_unit).to(psi_bd.units).magnitude

        psi_full = np.asarray(psi_bd.signal, dtype=float)
        try:
            psi_flat = psi_full.ravel()
        except Exception as exc:  # noqa: BLE001
            raise ValueError("IndexPixels: could not flatten Psi array.") from exc

        # ------------------------------------------------------------------
        # 3. Build masks
        # ------------------------------------------------------------------
        finite_mask = np.isfinite(q_flat) & np.isfinite(psi_flat)

        # Radial mask from Q limits
        q_range_mask = (q_flat >= q_min_val) & (q_flat <= q_max_val)

        # Azimuthal mask from Psi limits
        if np.isclose(psi_min_val, psi_max_val):
            # Full circle
            psi_mask = np.ones_like(psi_flat, dtype=bool)
        elif psi_min_val < psi_max_val:
            psi_mask = (psi_flat >= psi_min_val) & (psi_flat <= psi_max_val)
        else:
            # Wrap-around (e.g. 350° .. 10° converted to Psi.units)
            psi_mask = (psi_flat >= psi_min_val) | (psi_flat <= psi_max_val)

        valid_geom = q_range_mask & psi_mask & finite_mask

        # ------------------------------------------------------------------
        # 4. Build bin edges and assign indices
        # ------------------------------------------------------------------
        if direction == "azimuthal":
            coord_flat = q_flat
            if bin_type == "log":
                if q_min_val <= 0.0:
                    raise ValueError("IndexPixels: q_min must be > 0 for log binning.")
                bin_edges = np.geomspace(q_min_val, q_max_val, num=n_bins + 1, dtype=float)
            elif bin_type == "linear":
                bin_edges = np.linspace(q_min_val, q_max_val, num=n_bins + 1, dtype=float)
            else:
                raise ValueError(
                    f"IndexPixels: unknown bin_type {bin_type!r} for radial averaging. Expected 'log' or 'linear'."
                )
        else:  # radial
            # direction == "radial": bin along Psi, require linear spacing
            coord_flat = psi_flat
            if bin_type != "linear":
                raise ValueError("IndexPixels: for averaging_direction='radial', only bin_type='linear' is supported.")
            bin_edges = np.linspace(psi_min_val, psi_max_val, num=n_bins + 1, dtype=float)

        bin_idx = np.searchsorted(bin_edges, coord_flat, side="right") - 1
        out_of_range = (bin_idx < 0) | (bin_idx >= n_bins)
        valid_idx = valid_geom & ~out_of_range

        # Pixels that are not valid for any reason get index -1
        bin_idx[~valid_idx] = -1

        # Reshape to the spatial shape
        bin_idx_reshaped = bin_idx.reshape(spatial_shape)

        pixel_index_bd = BaseData(
            signal=bin_idx_reshaped,
            units=ureg.dimensionless,
            uncertainties={},
            weights=np.array(1.0),
            axes=spatial_axes,
            rank_of_data=signal_bd.rank_of_data,
        )

        self._prepared_data["pixel_index_bd"] = pixel_index_bd

    # ------------------------------------------------------------------
    # calculate: only wraps the prepared BaseData into DataBundles
    # ------------------------------------------------------------------
    def calculate(self) -> Dict[str, DataBundle]:
        """
        Add the pixel index as BaseData to the databundles specified in
        'with_processing_keys'. If multiple keys are given, the same pixel
        index map (computed from the first) is added to all.
        """
        output: Dict[str, DataBundle] = {}

        if self.processing_data is None:
            logger.warning("IndexPixels: processing_data is None in calculate; nothing to do.")
            return output

        if self._prepared_data.get("pixel_index_bd") is None:
            self.prepare_execution()

        pixel_index_bd: BaseData = self._prepared_data["pixel_index_bd"]
        _primary, keys_to_update = self._normalised_keys()

        logger.info(f"IndexPixels: adding pixel indices to keys={keys_to_update}")

        for key in keys_to_update:
            databundle = self.processing_data.get(key)
            if databundle is None:
                logger.warning(
                    "IndexPixels: processing_data has no entry for key=%r; skipping.",
                    key,
                )
                continue

            # Use a copy so each databundle has its own BaseData instance
            databundle["pixel_index"] = pixel_index_bd.copy(with_axes=True)
            output[key] = databundle

        logger.info(f"IndexPixels: pixel indices attached for {len(output)} keys.")
        return output
