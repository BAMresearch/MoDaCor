# SPDX-License-Identifier: BSD-3-Clause
# /usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

__coding__ = "utf-8"
__authors__ = ["Brian R. Pauw", "Anja F. Hörmann"]
__copyright__ = "Copyright 2025, The MoDaCor team"
__date__ = "23/07/2026"
__status__ = "Development"  # "Development", "Production"

__version__ = "20260723.1"
__all__ = ["IndexLinecutPixels"]

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


class IndexLinecutPixels(ProcessStep):
    """
    Compute pixel bin indices for a single dataset, for subsequent 1D averaging.

    Depending on `averaging_direction`, this step can prepare indices for:
      - linecut parallel to the surface (bin along Q_par, to compare with 1D X-ray scattering curves).
      - linecut perpendicular to the surface (bin along Q_per, for example to identify the Yoneda). 

    This step:
      - Interprets Qpar/Qper limits in user-specified units (q_limits_unit).
      - Builds bin edges internally (Qpar or Qper depending on averaging_direction).
      - For each pixel, decides which bin it belongs to, or -1 if it does
        not participate in any bin (out of range / outside ROI / non-finite).

    Inputs (from the databundle selected via with_processing_keys)
    --------------------------------------------------------------
    - "signal": BaseData   (together with its rank_of_data used for data shape)
    - "Qpar": BaseData     (component of the scattering vector parallel to the surface)
    - "Qper": BaseData     (component of the scattering vector perpendicular to the surface)

    This step does *not* apply the Mask. Mask is left to other modules
    (e.g., the averaging step), so that it can vary per frame for dynamic masking.

    Configuration
    -------------
    with_processing_keys : str | list[str] | None
        Databundle key(s) to work on. The pixel index map is computed from
        the first key and attached to all specified keys.
        If None and there is exactly one databundle, that one is used.

    averaging_direction : {"parallel", "perpendicular"}, default "parallel"
        - "parallel": bins along Qpar, using q_min/q_max and bin_type;
        - "perpendicular": bins along Qper, using q_min/q_max and bin_type;

    q_min, q_max : float, optional
        Q limits expressed in units given by q_unit.
        If omitted:
          - For "radial" + "log" binning: q_min = smallest positive finite Q;
          - Otherwise: q_min = min(Q), q_max = max(Q).
        q_min may be negative if not using "log" binning. 

    q_unit : str or pint.Unit, optional
        Units in which q_min/q_max and q_slice_width are defined, e.g. "1/nm".
        Defaults to the Q.units of the dataset.

    n_bins : int, default 100
        Number of bins along the averaging direction (Q or Psi).

    bin_type : {"log", "linear"}, default "linear"
        - "log" uses geometric spacing (np.geomspace);
        - "linear" uses np.linspace.

    q_slice_loc : float
        Location of the center of the linecut expressed in q_unit. 

    q_slice_width : float, optional
        Width of the linecut expressed in q_unit.
        Defines the binning range along the opposite of averaging_direction.

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
        calling_name="Index Linecut Pixels",
        calling_id="IndexLinecutPixels",
        calling_module_path=Path(__file__),
        calling_version=__version__,
        required_data_keys=["signal", "Qpar", "Qper"],
        arguments={
            "with_processing_keys": {
                "type": (str, list, type(None)),
                "required": True,
                "default": None,
                "doc": "ProcessingData key or list of keys to index.",
            },
            "averaging_direction": {
                "type": str,
                "required": True,
                "default": "parallel",
                "doc": "Averaging direction: 'parallel' or 'perpendicular'.",
            },
            "q_min": {
                "type": (float, int, type(None)),
                "default": None,
                "doc": "Minimum Q value for binning.",
            },
            "q_max": {
                "type": (float, int, type(None)),
                "default": None,
                "doc": "Maximum Q value for binning.",
            },
            "q_unit": {
                "type": (str, type(None)),
                "default": None,
                "doc": "Units for q_min/q_max if provided.",
            },
            "n_bins": {
                "type": int,
                "default": 100,
                "doc": "Number of bins.",
            },
            "bin_type": {
                "type": str,
                "default": "linear",
                "doc": "Binning type: 'linear' or 'log'.",
            },
            "q_slice_loc": {
                "type": (float, int, type(None)),
                "default": None,
                "doc": "Location of the center of the linecut.",
            },
            "q_slice_width": {
                "type": (float, int, type(None)),
                "default": None,
                "doc": "Width of the linecut.",
            },
        },
        modifies={},  # nothing, we only add.
        step_keywords=[
            "parallel",
            "perpendicular",
            "pixel indexing",
            "binning",
            "linecut",
            "grazing incidence",
        ],
        step_doc="Compute per-pixel bin indices (radial or azimuthal) for later 1D averaging.",
        step_reference="DOI 10.1088/0953-8984/25/38/383201",
        step_note=(
            "IndexPixels computes bin indices purely from geometry (Qpar, Qper) and "
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
        keys = self._normalised_processing_keys()
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

            signal_bd, qpar_bd, qper_bd, RoD, spatial_shape, spatial_axes
        """
        signal_bd: BaseData = databundle["signal"]
        qpar_bd: BaseData = databundle["Qpar"]
        qper_bd: BaseData = databundle["Qper"]

        RoD: int = int(signal_bd.rank_of_data)
        if RoD != 2:
            raise ValueError(f"IndexPixels: rank_of_data must be 2 for linecut indexing, got {RoD}.")

        spatial_shape: Tuple[int, ...] = signal_bd.shape[-RoD:] if RoD > 0 else ()

        # we expect Qpar/Qper to be 1-dimensional after remapping
        # have to actually use axes here to match which one belongs to rows/columns
        # if qpar_bd.shape != spatial_shape[1]:
        #     raise ValueError(f"IndexPixels: Qpar shape {q_bd.shape} does not match spatial shape {spatial_shape}.")
        # if qper_bd.shape != spatial_shape[0]:
        #     raise ValueError(f"IndexPixels: Psi shape {psi_bd.shape} does not match spatial shape {spatial_shape}.")

        if signal_bd.axes:
            spatial_axes: List[BaseData | None] = list(signal_bd.axes[-RoD:])
        else:
            spatial_axes = []

        return signal_bd, qpar_bd, qper_bd, RoD, spatial_shape, spatial_axes

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
            qpar_bd,
            qper_bd,
            RoD,
            spatial_shape,
            spatial_axes,
        ) = self._validate_and_get_geometry(databundle)

        # Direction of averaging: "radial" or "azimuthal"
        direction = str(self.configuration.get("averaging_direction", "parallel")).lower()
        if direction not in ("parallel", "perpendicular"):
            raise ValueError(f"IndexPixels: averaging_direction must be 'parallel' or 'perpendicular', got {direction!r}.")

        # ------------------------------------------------------------------
        # 1. Resolve Q limits (mask +, for radial, binning)
        # ------------------------------------------------------------------
        qpar, qper = np.meshgrid(qpar_bd.signal, qper_bd.signal) # convert to 2d
        q_min_cfg = self.configuration.get("q_min", None)
        q_max_cfg = self.configuration.get("q_max", None)
        n_bins = int(self.configuration.get("n_bins", 100))
        bin_type = str(self.configuration.get("bin_type", "linear")).lower()

        if n_bins <= 0:
            raise ValueError(f"IndexPixels: n_bins must be positive, got {n_bins}.")

        q_unit_cfg = self.configuration.get("q_unit", None)
        if q_unit_cfg is None:
            q_unit = qpar_bd.units # assume Q units are the same in both directions
        else:
            q_unit = ureg.Unit(q_unit_cfg)

        if direction == "parallel":
            # q_min/q_max define both mask and bin range
            
            try:
                q_flat = qpar.ravel()
            except Exception as exc:  # noqa: BLE001
                raise ValueError("IndexPixels: could not flatten Qpar array.") from exc

            finite_q = q_flat[np.isfinite(q_flat)]
            if finite_q.size == 0:
                raise ValueError("IndexPixels: Q array has no finite values.")

            data_q_min = float(np.nanmin(finite_q))
            data_q_max = float(np.nanmax(finite_q))

            if q_min_cfg is not None:
                q_min_val = (float(q_min_cfg) * q_unit).to(qpar_bd.units).magnitude
            else:
                if bin_type == "log":
                    positive = finite_q[finite_q > 0.0]
                    if positive.size == 0:
                        raise ValueError("IndexPixels: cannot determine positive q_min for log binning.")
                    q_min_val = float(np.nanmin(positive))
                else:
                    q_min_val = data_q_min

            if q_max_cfg is not None:
                q_max_val = (float(q_max_cfg) * q_unit).to(qpar_bd.units).magnitude
            else:
                q_max_val = data_q_max

            q_width = qper.ravel()
            q_loc_cfg = self.configuration.get("q_slice_loc", 0)
            q_width_cfg = self.configuration.get("q_slice_width", 0.1)
            q_width_min_val = ((float(q_loc_cfg) - float(q_width_cfg)/2.)* q_unit).to(qper_bd.units).magnitude
            q_width_max_val = ((float(q_loc_cfg) + float(q_width_cfg)/2.)* q_unit).to(qper_bd.units).magnitude
        else:
            # radial: q_min/q_max are optional ROI only; ignore bin_type here
                        
            try:
                q_flat = qper.ravel()
            except Exception as exc:  # noqa: BLE001
                raise ValueError("IndexPixels: could not flatten Qpar array.") from exc

            finite_q = q_flat[np.isfinite(q_flat)]
            if finite_q.size == 0:
                raise ValueError("IndexPixels: Q array has no finite values.")

            data_q_min = float(np.nanmin(finite_q))
            data_q_max = float(np.nanmax(finite_q))

            if q_min_cfg is not None:
                q_min_val = (float(q_min_cfg) * q_unit).to(qper_bd.units).magnitude
            else:
                q_min_val = data_q_min

            if q_max_cfg is not None:
                q_max_val = (float(q_max_cfg) * q_unit).to(qper_bd.units).magnitude
            else:
                q_max_val = data_q_max

            
            q_width = qpar.ravel()
            q_loc_cfg = self.configuration.get("q_slice_loc", 0)
            q_width_cfg = self.configuration.get("q_slice_width", 0.1)
            q_width_min_val = ((float(q_loc_cfg) - float(q_width_cfg)/2.)* q_unit).to(qpar_bd.units).magnitude
            q_width_max_val = ((float(q_loc_cfg) + float(q_width_cfg)/2.)* q_unit).to(qpar_bd.units).magnitude


        if q_max_val <= q_min_val or not np.isfinite(q_min_val) or not np.isfinite(q_max_val):
            raise ValueError(f"IndexPixels: invalid Q range q_min={q_min_val}, q_max={q_max_val}.")

        # ------------------------------------------------------------------
        # 2. Build masks
        # ------------------------------------------------------------------
        finite_mask = np.isfinite(q_flat) & np.isfinite(q_width)

        # Radial mask from Q limits
        q_range_mask = (q_flat >= q_min_val) & (q_flat <= q_max_val)

        # Azimuthal mask from slice width limits
        q_width_mask = (q_width >= q_width_min_val) & (q_width <= q_width_max_val)

        valid_geom = q_range_mask & q_width_mask & finite_mask

        # ------------------------------------------------------------------
        # 4. Build bin edges and assign indices
        # ------------------------------------------------------------------
        
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
