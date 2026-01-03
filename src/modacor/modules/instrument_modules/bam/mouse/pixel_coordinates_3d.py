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
__all__ = ["PixelCoordinates3D"]

from pathlib import Path
from typing import Tuple

from modacor.dataclasses.basedata import BaseData
from modacor.dataclasses.helpers import basedata_from_sources
from modacor.dataclasses.process_step_describer import ProcessStepDescriber
from modacor.modules.technique_modules.scattering.pixel_coordinates_3d_template import (
    CanonicalDetectorFrame,
    PixelCoordinates3DTemplate,
)

# import numpy as np


class PixelCoordinates3D(PixelCoordinates3DTemplate):
    """
    MOUSE implementation: loads explicit slow/fast pixel pitch + beam center sources.

    Tilt-ready:
      - basis vectors can be configured (later: compute them from a NeXus depends_on chain)
    """

    documentation = ProcessStepDescriber(
        calling_name="Add 3D pixel coordinates (MOUSE)",
        calling_id="PixelCoordinates3D_MOUSE",
        calling_module_path=Path(__file__),
        calling_version=__version__,
        required_data_keys=["signal"],
        required_arguments=[
            "origin_source",
            "origin_units_source",
            "origin_uncertainties_sources",
            "pixel_pitch_slow_source",
            "pixel_pitch_slow_units_source",
            "pixel_pitch_slow_uncertainties_sources",
            "pixel_pitch_fast_source",
            "pixel_pitch_fast_units_source",
            "pixel_pitch_fast_uncertainties_sources",
            "beam_center_slow_px_source",
            "beam_center_slow_px_units_source",
            "beam_center_slow_px_uncertainties_sources",
            "beam_center_fast_px_source",
            "beam_center_fast_px_units_source",
            "beam_center_fast_px_uncertainties_sources",
        ],
        default_configuration={
            "origin_source": None,
            "origin_units_source": None,
            "origin_uncertainties_sources": {},
            "pixel_pitch_slow_source": None,
            "pixel_pitch_slow_units_source": None,
            "pixel_pitch_slow_uncertainties_sources": {},
            "pixel_pitch_fast_source": None,
            "pixel_pitch_fast_units_source": None,
            "pixel_pitch_fast_uncertainties_sources": {},
            "beam_center_slow_px_source": None,
            "beam_center_slow_px_units_source": None,
            "beam_center_slow_px_uncertainties_sources": {},
            "beam_center_fast_px_source": None,
            "beam_center_fast_px_units_source": None,
            "beam_center_fast_px_uncertainties_sources": {},
            "basis_fast": (1.0, 0.0, 0.0),
            "basis_slow": (0.0, 1.0, 0.0),
            "basis_normal": (0.0, 0.0, 1.0),
        },
        modifies={
            "coord_x": ["signal", "uncertainties"],
            "coord_y": ["signal", "uncertainties"],
            "coord_z": ["signal", "uncertainties"],
        },
        step_keywords=["geometry", "coordinates", "MOUSE", "detector"],
        step_doc="Computes 3D pixel center coordinates in lab-frame axes for MOUSE data.",
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
        origin = self._load_from_sources("origin")  # (3,) length

        pitch_slow = self._load_from_sources("pixel_pitch_slow")  # scalar length/pixel
        pitch_fast = self._load_from_sources("pixel_pitch_fast")

        bc_slow = self._load_from_sources("beam_center_slow_px")  # scalar pixel
        bc_fast = self._load_from_sources("beam_center_fast_px")

        e_fast = self._unit(self.configuration.get("basis_fast", (1.0, 0.0, 0.0)))
        e_slow = self._unit(self.configuration.get("basis_slow", (0.0, 1.0, 0.0)))
        e_norm = self._unit(self.configuration.get("basis_normal", (0.0, 0.0, 1.0)))

        return CanonicalDetectorFrame(
            origin=origin,
            e_fast=e_fast,
            e_slow=e_slow,
            e_normal=e_norm,
            pixel_pitch_slow=pitch_slow,
            pixel_pitch_fast=pitch_fast,
            beam_center_slow_px=bc_slow,
            beam_center_fast_px=bc_fast,
        )
