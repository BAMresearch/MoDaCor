# SPDX-License-Identifier: BSD-3-Clause
# /usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from modacor import ureg
from modacor.dataclasses.basedata import BaseData
from modacor.dataclasses.helpers import basedata_from_sources
from modacor.dataclasses.messagehandler import MessageHandler
from modacor.dataclasses.process_step import ProcessStep
from modacor.dataclasses.process_step_describer import ProcessStepDescriber
from modacor.modules.technique_modules.scattering.geometry_helpers import (
    prepare_static_scalar,
    require_scalar,
    unit_vec3,
)

logger = MessageHandler(name=__name__)

__version__ = "20260106.1"
__all__ = ["XSGeometryFromPixelCoordinates"]


class XSGeometryFromPixelCoordinates(ProcessStep):
    """
    Compute scattering geometry from precomputed lab-frame pixel coordinates.

    Inputs in each databundle:
      - coord_x, coord_y, coord_z  (BaseData arrays, length units)

    Inputs from configuration sources:
      - sample_z: scalar length (sample is at (0,0,sample_z))
      - wavelength: scalar length
      - pixel_pitch_fast, pixel_pitch_slow: scalar length/pixel (for Omega)

    Outputs:
      - Q0, Q1, Q2, Q, Psi, TwoTheta, Omega
    """

    documentation = ProcessStepDescriber(
        calling_name="Add Q, Psi, TwoTheta, Omega from pixel coordinates",
        calling_id="XSGeometryFromPixelCoordinates",
        calling_module_path=Path(__file__),
        calling_version=__version__,
        required_data_keys=["coord_x", "coord_y", "coord_z"],
        required_arguments=[
            "sample_z_source",
            "sample_z_units_source",
            "sample_z_uncertainties_sources",
            "wavelength_source",
            "wavelength_units_source",
            "wavelength_uncertainties_sources",
            "pixel_pitch_slow_source",
            "pixel_pitch_slow_units_source",
            "pixel_pitch_slow_uncertainties_sources",
            "pixel_pitch_fast_source",
            "pixel_pitch_fast_units_source",
            "pixel_pitch_fast_uncertainties_sources",
        ],
        default_configuration={
            "sample_z_source": None,
            "sample_z_units_source": None,
            "sample_z_uncertainties_sources": {},
            "wavelength_source": None,
            "wavelength_units_source": None,
            "wavelength_uncertainties_sources": {},
            "pixel_pitch_slow_source": None,
            "pixel_pitch_slow_units_source": None,
            "pixel_pitch_slow_uncertainties_sources": {},
            "pixel_pitch_fast_source": None,
            "pixel_pitch_fast_units_source": None,
            "pixel_pitch_fast_uncertainties_sources": {},
            # optional detector normal (lab frame)
            "detector_normal": (0.0, 0.0, 1.0),
        },
        argument_specs={
            "sample_z_source": {
                "type": (str, type(None)),
                "required": True,
                "doc": "IoSources key for sample z-position signal.",
            },
            "sample_z_units_source": {
                "type": (str, type(None)),
                "required": False,
                "doc": "IoSources key for sample z-position units.",
            },
            "sample_z_uncertainties_sources": {
                "type": dict,
                "required": False,
                "doc": "Uncertainty sources for sample z-position.",
            },
            "wavelength_source": {
                "type": (str, type(None)),
                "required": True,
                "doc": "IoSources key for wavelength signal.",
            },
            "wavelength_units_source": {
                "type": (str, type(None)),
                "required": False,
                "doc": "IoSources key for wavelength units.",
            },
            "wavelength_uncertainties_sources": {
                "type": dict,
                "required": False,
                "doc": "Uncertainty sources for wavelength.",
            },
            "pixel_pitch_slow_source": {
                "type": (str, type(None)),
                "required": True,
                "doc": "IoSources key for slow-axis pixel pitch signal.",
            },
            "pixel_pitch_slow_units_source": {
                "type": (str, type(None)),
                "required": False,
                "doc": "IoSources key for slow-axis pixel pitch units.",
            },
            "pixel_pitch_slow_uncertainties_sources": {
                "type": dict,
                "required": False,
                "doc": "Uncertainty sources for slow-axis pixel pitch.",
            },
            "pixel_pitch_fast_source": {
                "type": (str, type(None)),
                "required": True,
                "doc": "IoSources key for fast-axis pixel pitch signal.",
            },
            "pixel_pitch_fast_units_source": {
                "type": (str, type(None)),
                "required": False,
                "doc": "IoSources key for fast-axis pixel pitch units.",
            },
            "pixel_pitch_fast_uncertainties_sources": {
                "type": dict,
                "required": False,
                "doc": "Uncertainty sources for fast-axis pixel pitch.",
            },
            "detector_normal": {
                "type": tuple,
                "required": False,
                "doc": "Detector normal unit vector in lab frame.",
            },
        },
        modifies={
            "Q0": ["signal", "uncertainties"],
            "Q1": ["signal", "uncertainties"],
            "Q2": ["signal", "uncertainties"],
            "Q": ["signal", "uncertainties"],
            "Psi": ["signal"],  # computed from nominal x/y only
            "TwoTheta": ["signal", "uncertainties"],
            "Omega": ["signal", "uncertainties"],
        },
        step_keywords=["geometry", "Q", "Psi", "TwoTheta", "Solid Angle", "Omega", "scattering"],
        step_doc="Compute Q-vector components and angles from lab-frame pixel coordinates.",
    )

    output_keys: Tuple[str, ...] = ("Q0", "Q1", "Q2", "Q", "Psi", "TwoTheta", "Omega")

    # ----------------------------
    # loading helpers
    # ----------------------------

    def _load_from_sources(self, key: str) -> BaseData:
        return basedata_from_sources(
            io_sources=self.io_sources,
            signal_source=self.configuration.get(f"{key}_source"),
            units_source=self.configuration.get(f"{key}_units_source", None),
            uncertainty_sources=self.configuration.get(f"{key}_uncertainties_sources", {}),
        )

    # ----------------------------
    # core compute
    # ----------------------------

    def _compute(
        self,
        *,
        coord_x: BaseData,
        coord_y: BaseData,
        coord_z: BaseData,
        sample_z: BaseData,
        wavelength: BaseData,
        pitch_slow: BaseData,
        pitch_fast: BaseData,
        detector_normal: np.ndarray,
    ) -> Dict[str, BaseData]:
        # sample position is (0,0,sample_z)
        dz = coord_z - sample_z
        dx = coord_x
        dy = coord_y

        # ray length
        R = ((dx**2) + (dy**2) + (dz**2)).sqrt()

        # angles
        r_perp = ((dx**2) + (dy**2)).sqrt()
        TwoTheta = (r_perp / dz).arctan()  # radians

        # k = 2π/λ
        two_pi = float(2.0 * np.pi)
        k = two_pi / wavelength  # 1/length

        # unit direction to pixel
        rhat_x = dx / R
        rhat_y = dy / R
        rhat_z = dz / R

        # q = k_out - k_in, with k_in along +z: (0,0,k)
        Q0 = k * rhat_x
        Q1 = k * rhat_y
        Q2 = k * (rhat_z - 1.0)

        Q = ((Q0**2) + (Q1**2) + (Q2**2)).sqrt()

        # Psi from NOMINAL geometry only (matches your earlier approach)
        psi_signal = np.arctan2(dy.signal, dx.signal)
        Psi = BaseData(signal=psi_signal, units=ureg.radian)

        # Solid angle per pixel:
        # dΩ ≈ A * cos(alpha) / R^2, with cos(alpha)=n·rhat
        # Here A from pitches (length/pixel)×(length/pixel).
        n = detector_normal
        cos_alpha = (rhat_x * n[0]) + (rhat_y * n[1]) + (rhat_z * n[2])
        cos_alpha_clipped = cos_alpha.copy()
        cos_alpha_clipped.signal = np.clip(cos_alpha.signal, 0.0, None)
        cos_alpha = cos_alpha_clipped

        one_px = BaseData(signal=np.array(1.0, dtype=float), units=ureg.pixel, rank_of_data=0)
        area_pixel = (pitch_fast * one_px) * (pitch_slow * one_px)  # -> m^2
        Omega = (area_pixel * cos_alpha) / (R**2)
        Omega.units = ureg.steradian  # (steradian is dimensionless, but explicit is fine)

        return {"Q0": Q0, "Q1": Q1, "Q2": Q2, "Q": Q, "Psi": Psi, "TwoTheta": TwoTheta, "Omega": Omega}

    # ----------------------------
    # ProcessStep lifecycle
    # ----------------------------

    def prepare_execution(self):
        super().prepare_execution()

        with_keys = self.configuration.get("with_processing_keys") or []
        if not with_keys:
            raise ValueError("XSGeometryFromPixelCoordinates: configuration.with_processing_keys is empty.")

        # reference bundle
        ref = self.processing_data[with_keys[0]]
        coord_x: BaseData = ref["coord_x"]
        coord_y: BaseData = ref["coord_y"]
        coord_z: BaseData = ref["coord_z"]

        RoD = int(
            getattr(coord_x, "rank_of_data", ref["signal"].rank_of_data if "signal" in ref else np.ndim(coord_x.signal))
        )

        sample_z = prepare_static_scalar(
            self._load_from_sources("sample_z"), require_units=ureg.m, uncertainty_key="sample_position_jitter"
        )
        wavelength = prepare_static_scalar(
            self._load_from_sources("wavelength"), require_units=ureg.m, uncertainty_key="wavelength_jitter"
        )
        pitch_slow = prepare_static_scalar(
            self._load_from_sources("pixel_pitch_slow"),
            require_units=ureg.m / ureg.pixel,
            uncertainty_key="pixel_pitch_jitter",
        )
        pitch_fast = prepare_static_scalar(
            self._load_from_sources("pixel_pitch_fast"),
            require_units=ureg.m / ureg.pixel,
            uncertainty_key="pixel_pitch_jitter",
        )

        detector_normal = unit_vec3(self.configuration.get("detector_normal", (0.0, 0.0, 1.0)), name="detector_normal")

        # (optional) enforce scalar-ness right before compute:
        sample_z = require_scalar("sample_z", sample_z)
        wavelength = require_scalar("wavelength", wavelength)
        pitch_slow = require_scalar("pixel_pitch_slow", pitch_slow)
        pitch_fast = require_scalar("pixel_pitch_fast", pitch_fast)

        out = self._compute(
            coord_x=coord_x,
            coord_y=coord_y,
            coord_z=coord_z,
            sample_z=sample_z,
            wavelength=wavelength,
            pitch_slow=pitch_slow,
            pitch_fast=pitch_fast,
            detector_normal=detector_normal,
        )

        for bd in out.values():
            bd.rank_of_data = min(RoD, int(np.ndim(bd.signal)))

        self._prepared_data = {k: out[k] for k in self.output_keys}

    def calculate(self):
        with_keys = self.configuration.get("with_processing_keys") or []
        if not with_keys:
            logger.warning("XSGeometryFromPixelCoordinates: no with_processing_keys specified; nothing to do.")
            return {}

        out: Dict[str, object] = {}
        for key in with_keys:
            bundle = self.processing_data.get(key)
            if bundle is None:
                logger.warning(
                    f"XSGeometryFromPixelCoordinates: no processing_data entry for key={key!r}; skipping."  # noqa: E702
                )
                continue
            for out_key, bd in self._prepared_data.items():
                bundle[out_key] = bd
            out[key] = bundle
        return out
