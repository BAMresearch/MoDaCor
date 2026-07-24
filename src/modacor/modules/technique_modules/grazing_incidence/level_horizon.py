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
__all__ = ["LevelHorizon"]

from typing import Dict, Tuple

import numpy as np
from scipy.optimize import minimize_scalar, dual_annealing, Bounds


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
from modacor.modules.technique_modules.grazing_incidence.pixel_coordinates_with_roll import PixelCoordinatesWithRoll
logger = MessageHandler(name=__name__)



class LevelHorizon(PixelCoordinatesWithRoll):
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
        calling_id="LevelHorizon",
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
        step_keywords=["geometry", "roll", "coordinates", "detector"],
        step_doc="Optimizes the roll angle such that the data is symmetric.",
    )

    def to_min(self, sample_roll):
        self.configuration["sample_roll"] = sample_roll.flatten()[0]
        with_keys = normalize_str_list(self.configuration.get("with_processing_keys", None)) or []
        if not with_keys:
            raise ValueError("LevelHorizon: configuration.with_processing_keys is empty.")
        ref_signal: BaseData = self.processing_data[with_keys[0]]["signal"]
        RoD = ref_signal.rank_of_data
        detector_shape = self._detector_shape(ref_signal, RoD)

        frame = self._load_canonical_frame(RoD=RoD, detector_shape=detector_shape, reference_signal=ref_signal)
        outputs = self._compute_pixel_positions(RoD=RoD, detector_shape=detector_shape, frame=frame)

        coord_x = outputs["coord_x"]
        coord_y = outputs["coord_y"]

        # take two vertical cuts near the edge to determine the location of the horizon
        x_cut = np.quantile(np.abs(coord_x.signal), 0.75)

        condition_left = np.where((coord_x.signal > -1.1*x_cut) & (coord_x.signal < -0.9*x_cut))[1] 
        condition_right = np.where((coord_x.signal < 1.1*x_cut) & (coord_x.signal > 0.9*x_cut))[1]

        left = ref_signal.signal[500:,condition_left].mean(axis = 1)
        left_y = coord_y.signal[500:,condition_left].mean(axis = 1)
        right_y = coord_y.signal[500:,condition_right].mean(axis = 1)
        right = ref_signal.signal[500:,condition_right].mean(axis = 1)

        def step(x, loc, level_low, level_high):
            return (level_high - level_low) * 0.5 * (np.sign(x - loc) + 1)

        def edge_to_min(loc, x, data, level_high, level_low):
            functionvalues = step(x, loc = loc, level_high = level_high, level_low = level_low)
            return np.sum((data - functionvalues)**2)

        edge_res_l = minimize_scalar(edge_to_min, bounds = [-0.01, 0.03],
                                     args = (left_y, left, np.nanmax(left), np.nanmin(left)),
                                     )
        edge_res_r = minimize_scalar(edge_to_min, bounds = [-0.01, 0.03],
                                     args = (right_y, right, np.nanmax(right), np.nanmin(right)),
                                     )
        edge_left = edge_res_l.x
        edge_right = edge_res_r.x
        return np.sum((edge_left - edge_right)**2)

        average_left = ref_signal.signal[(coord_x.signal < 0) & (coord_y.signal > 0) & (coord_y.signal < 0.005)].mean()
        average_right = ref_signal.signal[(coord_x.signal > 0) & (coord_y.signal > 0) & (coord_y.signal < 0.005)].mean()
        return (average_left - average_right)**2
        

    # ----------------------------
    # ProcessStep lifecycle
    # ----------------------------

    def prepare_execution(self):
        super().prepare_execution()

        with_keys = normalize_str_list(self.configuration.get("with_processing_keys", None)) or []
        if not with_keys:
            raise ValueError("LevelHorizon: configuration.with_processing_keys is empty.")

        ref_signal: BaseData = self.processing_data[with_keys[0]]["signal"]

        RoD = ref_signal.rank_of_data
        if RoD not in (0, 1, 2):
            raise NotImplementedError(
                f"LevelHorizon: only RoD in (0, 1, 2) supported; got RoD={RoD}."  # noqa: E702
            )

        detector_shape = self._detector_shape(ref_signal, RoD)

        #res = minimize_scalar(lambda x: self.to_min(x), bounds = [-1.0, 1.0],                                     tol = 75e-6,)
        res = dual_annealing(lambda x: self.to_min(x), bounds = Bounds(lb = -1, ub = 1))
        self.configuration["sample_roll"] = res.x.flatten()[0]
        print("found optimal roll angle:", res.x)

        
        frame = self._load_canonical_frame(RoD=RoD, detector_shape=detector_shape, reference_signal=ref_signal)
        outputs = self._compute_pixel_positions(RoD=RoD, detector_shape=detector_shape, frame=frame)

        for bd in outputs.values():
            bd.rank_of_data = min(RoD, int(np.ndim(bd.signal)))

        self._prepared_data = {k: outputs[k] for k in ("coord_x", "coord_y", "coord_z")}

    def calculate(self):
        with_keys = normalize_str_list(self.configuration.get("with_processing_keys", None)) or []
        if not with_keys:
            logger.warning("LevelHorizon: no with_processing_keys specified; nothing to do.")
            return {}

        return attach_prepared_data(
            self.processing_data,
            with_keys,
            self._prepared_data,
            logger=logger,
            module_name="LevelHorizon",
        )
