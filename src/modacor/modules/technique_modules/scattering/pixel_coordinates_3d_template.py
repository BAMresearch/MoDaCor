# SPDX-License-Identifier: BSD-3-Clause
# /usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

__coding__ = "utf-8"
__authors__ = ["Brian R. Pauw"]  # add names to the list as appropriate
__copyright__ = "Copyright 2026, The MoDaCor team"
__date__ = "03/01/2026"
__status__ = "Development"

__version__ = "20260103.1"
__all__ = ["CanonicalDetectorFrame", "PixelCoordinates3DTemplate"]

from abc import ABC, abstractmethod
from typing import Dict, Tuple

import numpy as np
from attrs import define

from modacor import ureg
from modacor.dataclasses.basedata import BaseData
from modacor.dataclasses.messagehandler import MessageHandler
from modacor.dataclasses.process_step import ProcessStep

logger = MessageHandler(name=__name__)


@define(frozen=True, slots=True)
class CanonicalDetectorFrame:
    """
    Canonical detector frame for pixel coordinate calculation.

    Coordinates are in lab-frame NeXus axes (x, y, z).

    - origin: position of the beam intersection with the detector plane (x=y=0 on that plane), shape (3,)
    - e_fast/e_slow/e_normal: unit vectors in lab frame (shape (3,)), defining detector orientation
    - pixel_pitch_{slow,fast}: scalar length/pixel
    - beam_center_{slow,fast}_px: scalar pixel coordinate
    """

    origin: BaseData

    e_fast: np.ndarray
    e_slow: np.ndarray
    e_normal: np.ndarray  # for future tilt support

    pixel_pitch_slow: BaseData
    pixel_pitch_fast: BaseData
    beam_center_slow_px: BaseData
    beam_center_fast_px: BaseData


class PixelCoordinates3DTemplate(ProcessStep, ABC):
    """
    Primary arrays module: compute 3D pixel center coordinates in lab-frame NeXus axes.

    Outputs (BaseData, length units, detector shape):
      - coord_x
      - coord_y
      - coord_z

    Notes:
      - RoD is clamped to signal.ndim, so we never produce arrays larger than the detector.
      - Planar detector assumed; tilt is supported by changing e_* basis vectors in the loader.
      - no sensor thickness offset applied, it is assumed the photon detection happens at the coordinates computed.
    """

    signal_key: str = "signal"
    output_keys: Tuple[str, ...] = ("coord_x", "coord_y", "coord_z")
    supported_rank_of_data: Tuple[int, ...] = (1, 2)

    @abstractmethod
    def _load_canonical_frame(
        self,
        *,
        RoD: int,
        detector_shape: Tuple[int, ...],
        reference_signal: BaseData,
    ) -> CanonicalDetectorFrame:
        """Instrument-specific decoding goes here."""

    # ----------------------------
    # rank/shape helpers
    # ----------------------------

    @staticmethod
    def _effective_rank_of_data(signal_bd: BaseData) -> int:
        rod = int(signal_bd.rank_of_data)
        ndim = int(np.ndim(signal_bd.signal))
        if rod > ndim:
            logger.warning(
                f"PixelCoordinates3D: rank_of_data={rod} > signal.ndim={ndim}; clamping to {ndim}."  # noqa: E702
            )  # noqa: E702
        return min(rod, ndim)

    @staticmethod
    def _detector_shape(signal_bd: BaseData, RoD: int) -> Tuple[int, ...]:
        return () if RoD <= 0 else tuple(signal_bd.signal.shape[-RoD:])

    @staticmethod
    def _require_scalar(name: str, bd: BaseData) -> None:
        if np.size(bd.signal) != 1:
            raise ValueError(f"{name} must be scalar (size==1). Got shape={np.shape(bd.signal)}.")

    @staticmethod
    def _require_vec3(name: str, bd: BaseData) -> None:
        if tuple(np.shape(bd.signal)) != (3,):
            raise ValueError(f"{name} must have shape (3,). Got shape={np.shape(bd.signal)}.")

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
        self._require_vec3("origin", frame.origin)
        for name in (
            "pixel_pitch_slow",
            "pixel_pitch_fast",
            "beam_center_slow_px",
            "beam_center_fast_px",
        ):
            self._require_scalar(name, getattr(frame, name))

        if not frame.beam_center_fast_px.units.is_compatible_with(ureg.pixel):
            raise ValueError(f"beam_center_fast_px must be in pixels, got {frame.beam_center_fast_px.units}")

        if not frame.pixel_pitch_fast.units.is_compatible_with(ureg.m / ureg.pixel):
            raise ValueError(f"pixel_pitch_fast must be length/pixel, got {frame.pixel_pitch_fast.units}")

        e_fast = self._unit(frame.e_fast)
        e_slow = self._unit(frame.e_slow)
        # keep e_normal in the frame for future tilt; intentionally unused right now

        ox, oy, oz = (frame.origin.indexed(i, rank_of_data=0) for i in range(3))

        if RoD == 1:
            (n_fast,) = detector_shape
            d_fast_px = self._idx_fast_1d(n_fast) - frame.beam_center_fast_px
            offset_fast_len = d_fast_px * frame.pixel_pitch_fast

            coord_x = ox + (offset_fast_len * e_fast[0])
            coord_y = oy + (offset_fast_len * e_fast[1])
            coord_z = oz + (offset_fast_len * e_fast[2])
            return {"coord_x": coord_x, "coord_y": coord_y, "coord_z": coord_z}

        # RoD == 2
        if not frame.beam_center_slow_px.units.is_compatible_with(ureg.pixel):
            raise ValueError(f"beam_center_slow_px must be in pixels, got {frame.beam_center_slow_px.units}")
        if not frame.pixel_pitch_slow.units.is_compatible_with(ureg.m / ureg.pixel):
            raise ValueError(f"pixel_pitch_slow must be length/pixel, got {frame.pixel_pitch_slow.units}")

        n_slow, n_fast = detector_shape
        d_slow_px = self._idx_slow_2d(n_slow) - frame.beam_center_slow_px
        d_fast_px = self._idx_fast_2d(n_fast) - frame.beam_center_fast_px

        offset_slow_len = d_slow_px * frame.pixel_pitch_slow
        offset_fast_len = d_fast_px * frame.pixel_pitch_fast

        coord_x = ox + (offset_slow_len * e_slow[0]) + (offset_fast_len * e_fast[0])
        coord_y = oy + (offset_slow_len * e_slow[1]) + (offset_fast_len * e_fast[1])
        coord_z = oz + (offset_slow_len * e_slow[2]) + (offset_fast_len * e_fast[2])

        return {"coord_x": coord_x, "coord_y": coord_y, "coord_z": coord_z}

    # ----------------------------
    # ProcessStep lifecycle
    # ----------------------------

    def prepare_execution(self):
        super().prepare_execution()

        with_keys = self.configuration.get("with_processing_keys") or []
        if not with_keys:
            raise ValueError("PixelCoordinates3D: configuration.with_processing_keys is empty.")

        ref_signal: BaseData = self.processing_data[with_keys[0]][self.signal_key]

        RoD = self._effective_rank_of_data(ref_signal)
        if RoD not in self.supported_rank_of_data:
            raise NotImplementedError(
                f"PixelCoordinates3D: only RoD in {self.supported_rank_of_data} supported; got RoD={RoD}."  # noqa: E702
            )

        detector_shape = self._detector_shape(ref_signal, RoD)
        frame = self._load_canonical_frame(RoD=RoD, detector_shape=detector_shape, reference_signal=ref_signal)
        outputs = self._compute_pixel_positions(RoD=RoD, detector_shape=detector_shape, frame=frame)

        for bd in outputs.values():
            bd.rank_of_data = min(RoD, int(np.ndim(bd.signal)))

        self._prepared_data = {k: outputs[k] for k in self.output_keys}

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
