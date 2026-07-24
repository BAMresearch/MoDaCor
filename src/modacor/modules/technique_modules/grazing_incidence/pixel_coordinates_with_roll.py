# SPDX-License-Identifier: BSD-3-Clause
# /usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path

from modacor.dataclasses.helpers import basedata_from_sources
from modacor.dataclasses.process_step_describer import ProcessStepDescriber

__coding__ = "utf-8"
__authors__ = ["Anja F. Hörmann"]  # add names to the list as appropriate
__copyright__ = "Copyright 2026, The MoDaCor team"
__date__ = "24/07/2026"
__status__ = "Development"

__version__ = "20260724.1"
__all__ = ["PixelCoordinatesWithRoll"]

from typing import Dict, Tuple

import numpy as np

# import pint
from attrs import define

from modacor import ureg
from modacor.dataclasses.basedata import BaseData
from modacor.dataclasses.messagehandler import MessageHandler
from modacor.dataclasses.process_step import ProcessStep
from modacor.modules.helpers import attach_prepared_data, normalize_str_list
from modacor.modules.technique_modules.scattering.geometry_helpers import (
    detector_index_basedata,
    prepare_static_scalar,
    require_scalar,
    unit_vec3,
)
from modacor.modules.technique_modules.scattering.pixel_coordinates_3d import PixelCoordinates3D, CanonicalDetectorFrame

logger = MessageHandler(name=__name__)




class PixelCoordinatesWithRoll(PixelCoordinates3D):
    """
    Primary arrays module: compute 3D detector element center coordinates in lab-frame NeXus-like axes.

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
        calling_id="PixelCoordinatesWithRoll",
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
            "sample_roll": {
                "type": float,
                "default": 0.0,
                "doc": "Roll angle of the sample, translated to a detector roll to compensate.",
            },
        },
        modifies={
            "coord_x": ["signal", "uncertainties"],
            "coord_y": ["signal", "uncertainties"],
            "coord_z": ["signal", "uncertainties"],
        },
        step_keywords=["geometry", "coordinates", "detector"],
        step_doc="Computes 3D detector element center coordinates in lab-frame axes.",
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
            require_units=ureg.m,
            uncertainty_key="pixel_pitch_jitter",
        )  # scalar length
        pitch_fast = prepare_static_scalar(
            self._load_from_sources("pixel_pitch_fast"),
            require_units=ureg.m,
            uncertainty_key="pixel_pitch_jitter",
        )  # scalar length

        e_fast = unit_vec3(self.configuration.get("basis_fast", (1.0, 0.0, 0.0)), name="basis_fast")
        e_slow = unit_vec3(self.configuration.get("basis_slow", (0.0, 1.0, 0.0)), name="basis_slow")
        e_norm = unit_vec3(self.configuration.get("basis_normal", (0.0, 0.0, 1.0)), name="basis_normal")

        sample_roll = np.deg2rad(self.configuration.get("sample_roll", 0.0))
        detector_roll = -sample_roll

        # rotate the unit vectors
        rot_matrix = np.array([[np.cos(detector_roll), -np.sin(detector_roll), 0],
                              [np.sin(detector_roll), np.cos(detector_roll), 0],
                              [0, 0, 1],
                              ])
        e_fast = np.dot(rot_matrix,e_fast)
        e_slow = np.dot(rot_matrix,e_slow)

        # rotate the beam center
        rotated_coordinate = np.dot(rot_matrix, np.array([det_coord_x, det_coord_y, det_coord_z])
                                    )
        det_coord_x = rotated_coordinate[0]
        det_coord_y = rotated_coordinate[1]
        det_coord_z = rotated_coordinate[2]

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

