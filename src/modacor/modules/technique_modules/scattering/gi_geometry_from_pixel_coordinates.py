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
from modacor.modules.helpers import attach_prepared_data, normalize_str_list
from modacor.modules.technique_modules.scattering.geometry_helpers import (
    prepare_static_scalar,
    require_scalar,
    unit_vec3,
)

logger = MessageHandler(name=__name__)

__version__ = "20260717.1"
__all__ = ["GIGeometryFromPixelCoordinates"]


class GIGeometryFromPixelCoordinates(ProcessStep):
    """
    Compute scattering geometry from precomputed lab-frame pixel coordinates.

    Inputs in each databundle:
      - coord_x, coord_y, coord_z  (BaseData arrays, length units)

    Inputs from configuration sources:
      - sample_z: scalar length (sample is at (0,0,sample_z))
      - wavelength: scalar length
      - pixel_pitch_fast, pixel_pitch_slow: scalar detector element size in length units (for Omega);
        legacy forms such as mm/pixel are accepted because pixel is dimensionless

    Outputs:
      - Q0, Q1, Q2, Q, Psi, TwoTheta, Omega
    """

    documentation = ProcessStepDescriber(
        calling_name="Add Q, Psi, TwoTheta, Omega from pixel coordinates",
        calling_id="GIGeometryFromPixelCoordinates",
        calling_module_path=Path(__file__),
        calling_version=__version__,
        required_data_keys=["coord_x", "coord_y", "coord_z"],
        arguments={
            "sample_z_source": {
                "type": (str, type(None)),
                "required": True,
                "default": None,
                "doc": "IoSources key for sample z-position signal.",
            },
            "sample_z_units_source": {
                "type": (str, type(None)),
                "default": None,
                "doc": "IoSources key for sample z-position units.",
            },
            "sample_z_uncertainties_sources": {
                "type": dict,
                "default": {},
                "doc": "Uncertainty sources for sample z-position.",
            },
            "wavelength_source": {
                "type": (str, type(None)),
                "required": True,
                "default": None,
                "doc": "IoSources key for wavelength signal.",
            },
            "wavelength_units_source": {
                "type": (str, type(None)),
                "default": None,
                "doc": "IoSources key for wavelength units.",
            },
            "wavelength_uncertainties_sources": {
                "type": dict,
                "default": {},
                "doc": "Uncertainty sources for wavelength.",
            },
            "pixel_pitch_slow_source": {
                "type": (str, type(None)),
                "required": True,
                "default": None,
                "doc": "IoSources key for slow-axis detector element size signal.",
            },
            "pixel_pitch_slow_units_source": {
                "type": (str, type(None)),
                "default": None,
                "doc": "IoSources key for slow-axis detector element size units; prefer length units such as m or mm.",
            },
            "pixel_pitch_slow_uncertainties_sources": {
                "type": dict,
                "default": {},
                "doc": "Uncertainty sources for slow-axis detector element size.",
            },
            "pixel_pitch_fast_source": {
                "type": (str, type(None)),
                "required": True,
                "default": None,
                "doc": "IoSources key for fast-axis detector element size signal.",
            },
            "pixel_pitch_fast_units_source": {
                "type": (str, type(None)),
                "default": None,
                "doc": "IoSources key for fast-axis detector element size units; prefer length units such as m or mm.",
            },
            "pixel_pitch_fast_uncertainties_sources": {
                "type": dict,
                "default": {},
                "doc": "Uncertainty sources for fast-axis detector element size.",
            },
            "detector_normal": {
                "type": tuple,
                "default": (0.0, 0.0, 1.0),
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
            "signal": ["signal", "uncertainties"],
        },
        step_keywords=["geometry", "Q", "Psi", "TwoTheta", "Solid Angle", "Omega", "scattering"],
        step_doc="Compute Q-vector components and angles from lab-frame pixel coordinates.",
    )

    output_keys: Tuple[str, ...] = ("Q0", "Q1", "Q2", "Q", "Qpar", "Qper", "signal", "Psi", "TwoTheta", "Omega")

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

        # unit direction to detector element center
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

        # Solid angle per detector element:
        # dΩ ≈ A * cos(alpha) / R^2, with cos(alpha)=n·rhat
        # Here A is computed from detector element sizes in length units.
        n = detector_normal
        cos_alpha = (rhat_x * n[0]) + (rhat_y * n[1]) + (rhat_z * n[2])
        cos_alpha_clipped = cos_alpha.copy()
        cos_alpha_clipped.signal = np.clip(cos_alpha.signal, 0.0, None)
        cos_alpha = cos_alpha_clipped

        area_pixel = pitch_fast * pitch_slow
        Omega = (area_pixel * cos_alpha) / (R**2)
        Omega.units = ureg.steradian

        return {"Q0": Q0, "Q1": Q1, "Q2": Q2, "Q": Q, "Psi": Psi, "TwoTheta": TwoTheta, "Omega": Omega}

    def _mask_missing_wedge(
            self,
            Q0_bd: BaseData,
            Q1_bd: BaseData,
            Q2_bd: BaseData,
            Q_bd: BaseData,
            wavelength_bd: BaseData,
            signal_bd: BaseData,
            ) -> BaseData:
        """
        Mask the inaccessible area in q space with zeros
        """

        qpar = np.where(Q0_bd.signal > 0, np.sqrt(Q0_bd.signal**2 + Q2_bd.signal**2), -np.sqrt(Q0_bd.signal**2 + Q2_bd.signal**2))
        # it seems q_par in dawn is continuous, so we need to find a row where it is and map
        # the other rows to those values (with binning)
        qpar_row = np.where((Q1_bd.signal)**2 < 1e-3)
        q_par = qpar[qpar_row]
        
        #hist, q_par = np.histogram(q_par, bins = q_par.shape[0] - 1)
        q_par = np.linspace(q_par.min(), q_par.max(), q_par.shape[0])
        print(qpar_row, q_par.shape)

        
        Qpar_bd = BaseData(signal = -1*q_par[::-1], units = "1/nm", rank_of_data = 1)
        qper_col = np.where(Q0_bd.signal**2 == np.abs(Q0_bd.signal).min()**2)
        qper = np.where(Q1_bd.signal > 0, np.sqrt(Q_bd.signal**2 - qpar**2), -np.sqrt(Q_bd.signal**2 - qpar**2))
        q_per = qper[:,qper_col[1]].flatten()
        #hist, q_per = np.histogram(q_per, bins = q_per.shape[0] - 1)
        q_per = np.linspace(q_per.min(), q_per.max(), q_per.shape[0])
        
        for i in range(qpar.shape[0]):  # to do: modify binning - both directions (histogram2d?)
            digitized = np.digitize(qpar[i].astype(float), q_par[::-1].astype(float), right = True)
            bin_means = [signal_bd.signal[i][digitized == j].mean() for j in range(0, len(q_par))]
            signal_bd.signal[i] = np.array(bin_means)
        
        for i in range(qper.shape[1]):
            digitized_y = np.digitize(qper[:,i].astype(float), q_per.astype(float), right = True)
            bin_means_y = [signal_bd.signal[:,i][digitized_y == j].mean() for j in range(0, len(q_per))]
            signal_bd.signal[:,i] = np.array(bin_means_y)
            
        
        Qper_bd = BaseData(signal = q_per, units = "1/nm", rank_of_data = 1)
        return signal_bd, Qpar_bd, Qper_bd


    # ----------------------------
    # ProcessStep lifecycle
    # ----------------------------

    def prepare_execution(self):
        super().prepare_execution()

        with_keys = normalize_str_list(self.configuration.get("with_processing_keys", None)) or []
        if not with_keys:
            raise ValueError("GIGeometryFromPixelCoordinates: configuration.with_processing_keys is empty.")

        # reference bundle
        ref = self.processing_data[with_keys[0]]
        coord_x: BaseData = ref["coord_x"]
        coord_y: BaseData = ref["coord_y"]
        coord_z: BaseData = ref["coord_z"]
        signal_bd : BaseData = ref["signal"]

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
            require_units=ureg.m,
            uncertainty_key="pixel_pitch_jitter",
        )
        pitch_fast = prepare_static_scalar(
            self._load_from_sources("pixel_pitch_fast"),
            require_units=ureg.m,
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

        
        signal, Qpar_bd, Qper_bd = self._mask_missing_wedge(
            Q0_bd = out["Q0"],
            Q1_bd = out["Q1"],
            Q2_bd = out["Q2"],
            Q_bd = out["Q"],
            wavelength_bd = wavelength, 
            signal_bd = signal_bd)

        signal.axes = [Qper_bd, Qpar_bd]
        out["signal"] = signal
        out["Qpar"] = Qpar_bd
        out["Qper"] = Qper_bd



        for bd in out.values():
            bd.rank_of_data = min(RoD, int(np.ndim(bd.signal)))

        self._prepared_data = {k: out[k] for k in self.output_keys}

    def calculate(self):
        with_keys = normalize_str_list(self.configuration.get("with_processing_keys", None)) or []
        if not with_keys:
            logger.warning("GIGeometryFromPixelCoordinates: no with_processing_keys specified; nothing to do.")
            return {}

        return attach_prepared_data(
            self.processing_data,
            with_keys,
            self._prepared_data,
            logger=logger,
            module_name="GIGeometryFromPixelCoordinates",
        )
