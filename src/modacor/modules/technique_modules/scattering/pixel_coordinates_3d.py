# SPDX-License-Identifier: BSD-3-Clause
# /usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path

from modacor.dataclasses.helpers import basedata_from_sources
from modacor.dataclasses.process_step_describer import ProcessStepDescriber

__coding__ = "utf-8"
__authors__ = ["Brian R. Pauw"]  # add names to the list as appropriate
__copyright__ = "Copyright 2026, The MoDaCor team"
__date__ = "04/01/2026"
__status__ = "Development"

__version__ = "20260103.1"
__all__ = ["CanonicalDetectorFrame", "PixelCoordinates3D"]

from typing import Dict, Tuple

import numpy as np

# import pint
from attrs import define

from modacor import ureg
from modacor.dataclasses.basedata import BaseData
from modacor.dataclasses.messagehandler import MessageHandler
from modacor.dataclasses.process_step import ProcessStep
from modacor.modules.technique_modules.scattering.geometry_helpers import (
    prepare_static_scalar,
    require_scalar,
    unit_vec3,
)

logger = MessageHandler(name=__name__)


@define(frozen=True, slots=True)
class CanonicalDetectorFrame:
    """
    Canonical detector frame for pixel coordinate calculation.

    Coordinates are in lab-frame NeXus axes (x, y, z), z= along beam, y=up, x=left when looking downstream from source.
    Beam center is implicitly encoded as the lab-frame cartesian offset of the detector origin to the beam center in length units.
    - det_coord_z: lab-frame z-coordinate of the beam intersection with the detector plane (x=y=0 on that plane). units of length
    - det_coord_x: lab-frame x-coordinate of the detector origin. This indicates the offset of the detector origin to the beam center in the lab frame. units of length.
    - det_coord_y: lab-frame y-coordinate of the detector origin. This indicates the offset of the detector origin to the beam center in the lab frame. units of length.
    - e_fast/e_slow/e_normal: unit vectors in lab frame (shape (3,)), defining detector orientation.
    - pixel_pitch_{slow,fast}: scalar length/pixel

    Notes:
    Tilt support will be integrated when needed following the NeXus pitch, yaw, roll for rotations around x, y, z.
    This implementation assumes a non-moving, planar detector.
    For other, instrument-specific implementations, subclass PixelCoordinates3D and replace _load_canonical_frame().
    Origin of the detector is at pixel with index (0,0).
    """

    det_coord_z: BaseData
    det_coord_x: BaseData
    det_coord_y: BaseData

    e_fast: np.ndarray
    e_slow: np.ndarray
    e_normal: np.ndarray

    pixel_pitch_slow: BaseData
    pixel_pitch_fast: BaseData

    def __attrs_post_init__(self):
        object.__setattr__(self, "det_coord_z", prepare_static_scalar(self.det_coord_z, require_units=ureg.m))
        object.__setattr__(self, "det_coord_x", prepare_static_scalar(self.det_coord_x, require_units=ureg.m))
        object.__setattr__(self, "det_coord_y", prepare_static_scalar(self.det_coord_y, require_units=ureg.m))

        object.__setattr__(
            self,
            "pixel_pitch_slow",
            prepare_static_scalar(self.pixel_pitch_slow, require_units=ureg.m / ureg.pixel),
        )
        object.__setattr__(
            self,
            "pixel_pitch_fast",
            prepare_static_scalar(self.pixel_pitch_fast, require_units=ureg.m / ureg.pixel),
        )


class PixelCoordinates3D(ProcessStep):
    """
    Primary arrays module: compute 3D pixel center coordinates in lab-frame NeXus-like axes.

    Outputs (BaseData, length units, detector shape):
      - coord_x
      - coord_y
      - coord_z

    Notes:
      - output coordinate ndim is clamped to RoD (which can never be larger than signal.ndim), so we never produce arrays larger than the detector.
      - Planar detector assumed; tilt support will be implemented (following NeXus pitch, yaw, roll for rotations around x, y, z) in the future as needed.
      - no sensor thickness offset applied, it is assumed the photon detection happens at the coordinates computed.
    """

    documentation = ProcessStepDescriber(
        calling_name="Add 3D pixel coordinates (generic)",
        calling_id="PixelCoordinates3D",
        calling_module_path=Path(__file__),
        calling_version=__version__,
        required_data_keys=["signal"],
        arguments={
            "det_coord_z_source": {
                "type": (str, type(None)),
                "required": True,
                "default": None,
                "doc": "IoSources key for detector z-coordinate signal.",
            },
            "det_coord_z_units_source": {
                "type": (str, type(None)),
                "default": None,
                "doc": "IoSources key for detector z-coordinate units.",
            },
            "det_coord_z_uncertainties_sources": {
                "type": dict,
                "default": {},
                "doc": "Uncertainty sources for detector z-coordinate.",
            },
            "det_coord_x_source": {
                "type": (str, type(None)),
                "required": True,
                "default": None,
                "doc": "IoSources key for detector x-coordinate signal.",
            },
            "det_coord_x_units_source": {
                "type": (str, type(None)),
                "default": None,
                "doc": "IoSources key for detector x-coordinate units.",
            },
            "det_coord_x_uncertainties_sources": {
                "type": dict,
                "default": {},
                "doc": "Uncertainty sources for detector x-coordinate.",
            },
            "det_coord_y_source": {
                "type": (str, type(None)),
                "required": True,
                "default": None,
                "doc": "IoSources key for detector y-coordinate signal.",
            },
            "det_coord_y_units_source": {
                "type": (str, type(None)),
                "default": None,
                "doc": "IoSources key for detector y-coordinate units.",
            },
            "det_coord_y_uncertainties_sources": {
                "type": dict,
                "default": {},
                "doc": "Uncertainty sources for detector y-coordinate.",
            },
            "pixel_pitch_slow_source": {
                "type": (str, type(None)),
                "required": True,
                "default": None,
                "doc": "IoSources key for slow-axis pixel pitch signal.",
            },
            "pixel_pitch_slow_units_source": {
                "type": (str, type(None)),
                "default": None,
                "doc": "IoSources key for slow-axis pixel pitch units.",
            },
            "pixel_pitch_slow_uncertainties_sources": {
                "type": dict,
                "default": {},
                "doc": "Uncertainty sources for slow-axis pixel pitch.",
            },
            "pixel_pitch_fast_source": {
                "type": (str, type(None)),
                "required": True,
                "default": None,
                "doc": "IoSources key for fast-axis pixel pitch signal.",
            },
            "pixel_pitch_fast_units_source": {
                "type": (str, type(None)),
                "default": None,
                "doc": "IoSources key for fast-axis pixel pitch units.",
            },
            "pixel_pitch_fast_uncertainties_sources": {
                "type": dict,
                "default": {},
                "doc": "Uncertainty sources for fast-axis pixel pitch.",
            },
            "basis_fast": {
                "type": tuple,
                "default": (1.0, 0.0, 0.0),
                "doc": "Basis vector for the fast detector axis.",
            },
            "basis_slow": {
                "type": tuple,
                "default": (0.0, 1.0, 0.0),
                "doc": "Basis vector for the slow detector axis.",
            },
            "basis_normal": {
                "type": tuple,
                "default": (0.0, 0.0, 1.0),
                "doc": "Basis vector for the detector normal.",
            },
        },
        modifies={
            "coord_x": ["signal", "uncertainties"],
            "coord_y": ["signal", "uncertainties"],
            "coord_z": ["signal", "uncertainties"],
        },
        step_keywords=["geometry", "coordinates", "detector"],
        step_doc="Computes 3D pixel center coordinates in lab-frame axes.",
    )

    def _load_from_sources(self, key: str) -> BaseData:
        return basedata_from_sources(
            io_sources=self.io_sources,
            signal_source=self.configuration.get(f"{key}_source"),
            units_source=self.configuration.get(f"{key}_units_source", None),
            uncertainty_sources=self.configuration.get(f"{key}_uncertainties_sources", {}),
        )

    def _load_canonical_frame(
        self,
        *,
        RoD: int,
        detector_shape: Tuple[int, ...],
        reference_signal: BaseData,
    ) -> CanonicalDetectorFrame:
        det_coord_z = prepare_static_scalar(
            self._load_from_sources("det_coord_z"), require_units=ureg.m, uncertainty_key="detector_position_jitter"
        )  # scalar length
        det_coord_x = prepare_static_scalar(
            self._load_from_sources("det_coord_x"), require_units=ureg.m, uncertainty_key="detector_position_jitter"
        )  # scalar length
        det_coord_y = prepare_static_scalar(
            self._load_from_sources("det_coord_y"), require_units=ureg.m, uncertainty_key="detector_position_jitter"
        )  # scalar length

        pitch_slow = prepare_static_scalar(
            self._load_from_sources("pixel_pitch_slow"),
            require_units=ureg.m / ureg.pixel,
            uncertainty_key="pixel_pitch_jitter",
        )  # scalar length/pixel
        pitch_fast = prepare_static_scalar(
            self._load_from_sources("pixel_pitch_fast"),
            require_units=ureg.m / ureg.pixel,
            uncertainty_key="pixel_pitch_jitter",
        )  # scalar length/pixel

        e_fast = unit_vec3(self.configuration.get("basis_fast", (1.0, 0.0, 0.0)), name="basis_fast")
        e_slow = unit_vec3(self.configuration.get("basis_slow", (0.0, 1.0, 0.0)), name="basis_slow")
        e_norm = unit_vec3(self.configuration.get("basis_normal", (0.0, 0.0, 1.0)), name="basis_normal")

        return CanonicalDetectorFrame(
            det_coord_z=det_coord_z,
            det_coord_x=det_coord_x,
            det_coord_y=det_coord_y,
            e_fast=e_fast,
            e_slow=e_slow,
            e_normal=e_norm,
            pixel_pitch_slow=pitch_slow,
            pixel_pitch_fast=pitch_fast,
        )

    # ----------------------------
    # rank/shape helpers
    # ----------------------------

    @staticmethod
    def _detector_shape(signal_bd: BaseData, RoD: int) -> Tuple[int, ...]:
        return () if RoD <= 0 else tuple(signal_bd.signal.shape[-RoD:])

    @staticmethod
    def _require_scalar(name: str, bd: BaseData) -> None:
        if np.size(bd.signal) != 1:
            raise ValueError(f"{name} must be scalar (size==1). Got shape={np.shape(bd.signal)}.")

    @staticmethod
    def _unit(v: np.ndarray | Tuple[float, float, float]) -> np.ndarray:
        v = np.asarray(v, dtype=float).reshape(3)
        n = float(np.linalg.norm(v))
        if n == 0.0:
            raise ValueError("basis vector must be non-zero")
        return v / n

    # ----------------------------
    # broadcast-friendly pixel indices (pixel-center convention)
    # ----------------------------

    @staticmethod
    def _idx_fast_1d(n_fast: int) -> BaseData:
        sig = np.arange(n_fast, dtype=float) + 0.5
        return BaseData(signal=sig, units=ureg.pixel, uncertainties={"pixel_index_fast": np.full_like(sig, 0.5)})

    @staticmethod
    def _idx_slow_2d(n_slow: int) -> BaseData:
        sig = (np.arange(n_slow, dtype=float) + 0.5)[:, None]
        return BaseData(signal=sig, units=ureg.pixel, uncertainties={"pixel_index_slow": np.full_like(sig, 0.5)})

    @staticmethod
    def _idx_fast_2d(n_fast: int) -> BaseData:
        sig = (np.arange(n_fast, dtype=float) + 0.5)[None, :]
        return BaseData(signal=sig, units=ureg.pixel, uncertainties={"pixel_index_fast": np.full_like(sig, 0.5)})

    # ----------------------------
    # core compute
    # ----------------------------

    def _compute_pixel_positions(
        self,
        *,
        RoD: int,
        detector_shape: Tuple[int, ...],
        frame: CanonicalDetectorFrame,
    ) -> Dict[str, BaseData]:
        """
        here:
        det_coord_x/y/z represent the lab-frame position of the detector “pixel grid origin” (i.e. the corner before applying the +0.5 pixel-center shift).
        Pixel centers are at (i+0.5, j+0.5) which matches the current _idx_* methods and the ±0.5 px index uncertainty.
        """

        # Scalars in length units (already averaged + SEM in CanonicalDetectorFrame.__attrs_post_init__)
        ox = require_scalar("det_coord_x", frame.det_coord_x)
        oy = require_scalar("det_coord_y", frame.det_coord_y)
        oz = require_scalar("det_coord_z", frame.det_coord_z)
        pitch_fast = require_scalar("pixel_pitch_fast", frame.pixel_pitch_fast)
        pitch_slow = require_scalar("pixel_pitch_slow", frame.pixel_pitch_slow)

        e_fast = unit_vec3(frame.e_fast, name="e_fast")
        e_slow = unit_vec3(frame.e_slow, name="e_slow")
        # e_normal kept for future tilt support

        # RoD==0: no detector axes, just return the detector origin position as scalars
        if RoD == 0:
            return {"coord_x": ox, "coord_y": oy, "coord_z": oz}

        # RoD==1: one detector axis ("fast")
        if RoD == 1:
            (n_fast,) = detector_shape
            i_fast_px = self._idx_fast_1d(n_fast)  # (n_fast,), centers at i+0.5

            off_fast = i_fast_px * pitch_fast  # length along fast axis

            coord_x = ox + (off_fast * e_fast[0])
            coord_y = oy + (off_fast * e_fast[1])
            coord_z = oz + (off_fast * e_fast[2])
            return {"coord_x": coord_x, "coord_y": coord_y, "coord_z": coord_z}

        # RoD==2: (slow, fast)
        if RoD != 2:
            raise NotImplementedError(
                f"PixelCoordinates3D: only RoD in (0, 1, 2) supported; got RoD={RoD}."  # noqa: E702
            )

        n_slow, n_fast = detector_shape
        j_slow_px = self._idx_slow_2d(n_slow)  # (n_slow, 1)
        i_fast_px = self._idx_fast_2d(n_fast)  # (1, n_fast)

        off_slow = j_slow_px * pitch_slow  # (n_slow, 1) length
        off_fast = i_fast_px * pitch_fast  # (1, n_fast) length

        # Broadcast to (n_slow, n_fast) automatically via BaseData arithmetic
        coord_x = ox + (off_slow * e_slow[0]) + (off_fast * e_fast[0])
        coord_y = oy + (off_slow * e_slow[1]) + (off_fast * e_fast[1])
        coord_z = oz + (off_slow * e_slow[2]) + (off_fast * e_fast[2])

        return {"coord_x": coord_x, "coord_y": coord_y, "coord_z": coord_z}

    # ----------------------------
    # ProcessStep lifecycle
    # ----------------------------

    def prepare_execution(self):
        super().prepare_execution()

        with_keys = self.configuration.get("with_processing_keys") or []
        if not with_keys:
            raise ValueError("PixelCoordinates3D: configuration.with_processing_keys is empty.")

        ref_signal: BaseData = self.processing_data[with_keys[0]]["signal"]

        RoD = ref_signal.rank_of_data
        if RoD not in (0, 1, 2):
            raise NotImplementedError(
                f"PixelCoordinates3D: only RoD in (0, 1, 2) supported; got RoD={RoD}."  # noqa: E702
            )

        detector_shape = self._detector_shape(ref_signal, RoD)
        frame = self._load_canonical_frame(RoD=RoD, detector_shape=detector_shape, reference_signal=ref_signal)
        outputs = self._compute_pixel_positions(RoD=RoD, detector_shape=detector_shape, frame=frame)

        for bd in outputs.values():
            bd.rank_of_data = min(RoD, int(np.ndim(bd.signal)))

        self._prepared_data = {k: outputs[k] for k in ("coord_x", "coord_y", "coord_z")}

    def calculate(self):
        with_keys = self.configuration.get("with_processing_keys") or []
        if not with_keys:
            logger.warning("PixelCoordinates3D: no with_processing_keys specified; nothing to do.")
            return {}

        out: Dict[str, object] = {}
        for key in with_keys:
            bundle = self.processing_data.get(key)
            if bundle is None:
                logger.warning(
                    f"PixelCoordinates3D: processing_data has no entry for key={key!r}; skipping."  # noqa: E702
                )  # noqa: E702
                continue
            for out_key, bd in self._prepared_data.items():
                bundle[out_key] = bd
            out[key] = bundle

        return out
