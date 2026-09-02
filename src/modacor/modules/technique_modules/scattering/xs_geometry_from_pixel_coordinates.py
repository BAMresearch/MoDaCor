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
from modacor.modules.base_modules.nexus_transformations import (
    load_nexus_detector_frame_inputs,
    resolve_nexus_transform_chain,
)
from modacor.modules.helpers import attach_prepared_data, normalize_str_list
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
      - pixel_pitch_fast, pixel_pitch_slow: scalar detector element size in length units (for Omega);
        legacy forms such as mm/pixel are accepted because pixel is dimensionless

    Outputs:
      - Q0, Q1, Q2, Q, Psi, TwoTheta, Omega
    """

    documentation = ProcessStepDescriber(
        calling_name="Add Q, Psi, TwoTheta, Omega from pixel coordinates",
        calling_id="XSGeometryFromPixelCoordinates",
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
            "detector_frame": {
                "type": (dict, type(None)),
                "default": None,
                "doc": (
                    "Optional detector-frame adapter. Use {'type': 'nexus', 'source': '<ref>', "
                    "'detector_path': '/entry1/instrument/detector'} to load pixel pitch and "
                    "detector normal from a NeXus NXdetector/NXdetector_module transformation chain."
                ),
            },
            "sample_z_override": {
                "type": (dict, int, float, type(None)),
                "default": None,
                "doc": (
                    "Optional sample z-position override. Mapping form accepts either value/units "
                    "or {'type': 'nexus', 'source': '<ref>', 'transform_path': '<path>'}."
                ),
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

    @staticmethod
    def _translation_component_index(value) -> int:
        if value is None:
            return 2
        if isinstance(value, int):
            if value in (0, 1, 2):
                return value
            raise ValueError("NeXus sample_z_override component index must be 0, 1, or 2.")
        component = str(value).strip().lower()
        component_indices = {"x": 0, "0": 0, "y": 1, "1": 1, "z": 2, "2": 2}
        if component not in component_indices:
            raise ValueError("NeXus sample_z_override component must be one of x, y, z, 0, 1, or 2.")
        return component_indices[component]

    def _load_nexus_sample_z(self, override: dict) -> BaseData:
        source_reference = override.get("source", override.get("source_reference"))
        transform_path = override.get("transform_path", override.get("path", override.get("nexus_path")))
        if source_reference is None or transform_path is None:
            raise ValueError("NeXus sample_z_override requires 'source' and 'transform_path'.")

        transform = resolve_nexus_transform_chain(
            self.io_sources,
            str(source_reference),
            str(transform_path),
        )
        component_index = self._translation_component_index(override.get("component", "z"))
        return BaseData(
            signal=np.asarray(float(transform.translation[component_index]), dtype=float),
            units=ureg.m,
            rank_of_data=0,
        )

    def _load_sample_z(self) -> BaseData:
        override = self.configuration.get("sample_z_override")
        if override is None:
            return self._load_from_sources("sample_z")

        if isinstance(override, dict):
            override_type = str(override.get("type", "value")).strip().lower()
            if override_type == "nexus":
                return self._load_nexus_sample_z(override)
            if override_type != "value":
                raise ValueError(f"Unsupported sample_z_override type: {override_type!r}.")
            value = override.get("value", 0.0)
            units = override.get("units", "m")
        else:
            value = override
            units = "m"
        return BaseData(signal=np.asarray(value, dtype=float), units=ureg.Unit(str(units)), rank_of_data=0)

    def _load_nexus_detector_frame_inputs(self):
        detector_frame_cfg = self.configuration.get("detector_frame")
        if detector_frame_cfg is None:
            return None
        if not isinstance(detector_frame_cfg, dict):
            raise TypeError("XSGeometryFromPixelCoordinates detector_frame configuration must be a mapping.")
        frame_type = str(detector_frame_cfg.get("type", "")).strip().lower()
        if frame_type != "nexus":
            raise ValueError(f"Unsupported XSGeometryFromPixelCoordinates detector_frame type: {frame_type!r}.")
        return load_nexus_detector_frame_inputs(
            self.io_sources,
            source_reference=str(detector_frame_cfg["source"]),
            detector_path=str(detector_frame_cfg["detector_path"]),
            detector_module_name=str(detector_frame_cfg.get("detector_module_name", "detector_module")),
            module_origin=str(detector_frame_cfg.get("module_origin", "corner")),
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

    # ----------------------------
    # ProcessStep lifecycle
    # ----------------------------

    def prepare_execution(self):
        super().prepare_execution()

        with_keys = normalize_str_list(self.configuration.get("with_processing_keys", None)) or []
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
            self._load_sample_z(), require_units=ureg.m, uncertainty_key="sample_position_jitter"
        )
        wavelength = prepare_static_scalar(
            self._load_from_sources("wavelength"), require_units=ureg.m, uncertainty_key="wavelength_jitter"
        )
        nexus_frame_inputs = self._load_nexus_detector_frame_inputs()
        if nexus_frame_inputs is None:
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
            detector_normal = unit_vec3(
                self.configuration.get("detector_normal", (0.0, 0.0, 1.0)), name="detector_normal"
            )
        else:
            pitch_slow = nexus_frame_inputs.pixel_pitch_slow
            pitch_fast = nexus_frame_inputs.pixel_pitch_fast
            detector_normal = unit_vec3(
                self.configuration.get("detector_normal", nexus_frame_inputs.basis_normal), name="detector_normal"
            )

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
        with_keys = normalize_str_list(self.configuration.get("with_processing_keys", None)) or []
        if not with_keys:
            logger.warning("XSGeometryFromPixelCoordinates: no with_processing_keys specified; nothing to do.")
            return {}

        return attach_prepared_data(
            self.processing_data,
            with_keys,
            self._prepared_data,
            logger=logger,
            module_name="XSGeometryFromPixelCoordinates",
        )
