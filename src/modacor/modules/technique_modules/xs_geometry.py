# SPDX-License-Identifier: BSD-3-Clause
# /usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

__coding__ = "utf-8"
__authors__ = ["Brian R. Pauw"]  # add names to the list as appropriate
__copyright__ = "Copyright 2025, The MoDaCor team"
__date__ = "22/11/2025"
__status__ = "Development"  # "Development", "Production"
# end of header and standard imports

__version__ = "20251122.1"
__all__ = ["XSGeometry"]

from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from modacor import ureg
from modacor.dataclasses.basedata import BaseData
from modacor.dataclasses.helpers import basedata_from_sources
from modacor.dataclasses.process_step import ProcessStep
from modacor.dataclasses.process_step_describer import ProcessStepDescriber

# from typing import Any


class XSGeometry(ProcessStep):
    """
    Calculates the geometric information Q, Q0, Q1, Q2, Psi, TwoTheta, and Omega (solid angle)
    for X-ray scattering data and adds them to the databundle.

        Geometry model
    --------------
    * The last `rank_of_data` dimensions of `signal` are the detector dimensions,
    ordered as (..., y, x) for 2D and (..., y) for 1D.
    * `beam_center.signal` is given in [y, x] pixel coordinates for 2D,
    and [y] for 1D.
    * `pixel_size.signal` is [pixel_size_y, pixel_size_x] in length units / pixel.
    * `pixel_size` is a BaseData vector of length 2 or 3 (length units):
        - first component = pixel size along the "Q0" axis,
        - second component = pixel size along the "Q1" axis,
        - third component (if present) is currently unused.
    * `detector_distance` and `wavelength` are BaseData scalars with length units.

    All computed outputs (Q, Q0, Q1, Q2, Psi, TwoTheta, Omega) are BaseData objects.
    """

    documentation = ProcessStepDescriber(
        calling_name="Add Q, Psi, TwoTheta, Omega",  # Omega is Solid Angle
        calling_id="XSGeometry",
        calling_module_path=Path(__file__),
        calling_version=__version__,
        required_data_keys=["signal"],
        works_on={},
        step_keywords=[
            "geometry",
            "Q",
            "Psi",
            "TwoTheta",
            "Solid Angle",
            "Omega",
            "X-ray scattering",
        ],
        step_doc="Add geometric information Q, Psi, TwoTheta, and Solid Angle to the data",
        step_reference="DOI 10.1088/0953-8984/25/38/383201",
        step_note="This calculates geometric factors relevant for X-ray scattering data",
        calling_arguments={
            "detector_distance_source": None,
            "detector_distance_units_source": None,
            "detector_distance_uncertainties_sources": {},
            "pixel_size_source": None,
            "pixel_size_units_source": None,
            "pixel_size_uncertainties_sources": {},
            "beam_center_source": None,
            "beam_center_units_source": None,
            "beam_center_uncertainties_sources": {},
            "wavelength_source": None,
            "wavelength_units_source": None,
            "wavelength_uncertainties_sources": {},
        },
    )

    # ------------------------------------------------------------------
    # Small helpers: geometry loading & shape utilities
    # ------------------------------------------------------------------

    def _load_geometry(self) -> Dict[str, BaseData]:
        """
        Load all required geometry parameters as BaseData objects.

        Expected configuration keys:
        - detector_distance
        - pixel_size
        - beam_center
        - wavelength
        for each their *_source / *_units_source / *_uncertainties_sources.
        """
        geom: Dict[str, BaseData] = {}
        for key in ["detector_distance", "pixel_size", "beam_center", "wavelength"]:
            for subkey in [f"{key}_source", f"{key}_units_source", f"{key}_uncertainties_sources"]:
                if subkey not in self.configuration:
                    raise ValueError(f"Missing required configuration parameter: {subkey}")
            geom[key] = basedata_from_sources(
                io_sources=self.io_sources,
                signal_source=self.configuration.get(f"{key}_source"),
                units_source=self.configuration.get(f"{key}_units_source", None),
                uncertainty_sources=self.configuration.get(f"{key}_uncertainties_sources", {}),
            )
        return geom

    def _validate_geometry(
        self,
        geom: Dict[str, BaseData],
        RoD: int,
        spatial_shape: tuple[int, ...],
    ) -> None:
        """
        Validate that geometry inputs are consistent with the detector rank.
        """
        beam_center_bd = geom["beam_center"]
        pixel_size_bd = geom["pixel_size"]

        if RoD not in (0, 1, 2):
            raise NotImplementedError(f"XSGeometry supports RoD 0, 1, or 2; got RoD={RoD}.")  # noqa: E702

        # Beam center: for RoD>0, we expect a vector of length RoD.
        if RoD > 0:
            if beam_center_bd.signal.size != RoD:
                raise ValueError(
                    f"Beam center must have {RoD} components for RoD={RoD}, got size={beam_center_bd.signal.size}."
                )

        # Pixel size: vector of 2 or 3 components.
        if pixel_size_bd.shape not in ((2,), (3,)):
            raise ValueError(f"Pixel size should be a 2D or 3D vector, got shape={pixel_size_bd.shape}.")

        # Sanity check on spatial_shape vs RoD
        if RoD == 1 and len(spatial_shape) != 1:
            raise ValueError(f"RoD=1 expects 1D spatial shape, got {spatial_shape}.")
        if RoD == 2 and len(spatial_shape) != 2:
            raise ValueError(f"RoD=2 expects 2D spatial shape, got {spatial_shape}.")

    def _make_index_basedata(
        self,
        shape: tuple[int, ...],
        axis: int,
        uncertainty_key: str = "pixel_index",
    ) -> BaseData:
        """
        Create a BaseData representing pixel indices along a given axis.

        Each index gets an uncertainty of ±0.5 pixel to reflect the
        pixel-center assumption.
        """
        if len(shape) == 0:
            signal = np.array(0.0, dtype=float)
        else:
            grids = np.meshgrid(
                *[np.arange(n, dtype=float) for n in shape],
                indexing="ij",
            )
            signal = grids[axis]

        # always add half-pixel uncertainty estimate to pixel indices
        uncertainties: Dict[str, np.ndarray] = {uncertainty_key: np.full_like(signal, 0.5, dtype=float)}

        return BaseData(
            signal=signal,
            units=ureg.pixel,
            uncertainties=uncertainties,
        )

    # ------------------------------------------------------------------
    # Coordinate calculation per dimensionality
    # ------------------------------------------------------------------

    def _compute_coordinates(
        self,
        RoD: int,
        spatial_shape: tuple[int, ...],
        beam_center_bd: BaseData,
        px0_bd: BaseData,
        px1_bd: BaseData,
        detector_distance_bd: BaseData,
    ) -> Tuple[BaseData, BaseData, BaseData, BaseData]:
        """
        Compute detector-plane coordinates (x0, x1), in-plane radius r_perp,
        and distance R from sample to pixel center, all as BaseData.

        Returns
        -------
        x0_bd, x1_bd, r_perp_bd, R_bd
        """
        if RoD == 0:
            # 0D: no spatial axes, use the detector distance directly.
            x0_bd = BaseData(signal=np.array(0.0), units=px0_bd.units)
            x1_bd = BaseData(signal=np.array(0.0), units=px1_bd.units)
            r_perp_bd = BaseData(signal=np.array(0.0), units=px0_bd.units)
            R_bd = detector_distance_bd
            return x0_bd, x1_bd, r_perp_bd, R_bd

        if RoD == 1:
            (n0,) = spatial_shape
            idx0_bd = self._make_index_basedata(shape=(n0,), axis=0)

            rel_idx0_bd = idx0_bd - beam_center_bd.indexed(0, rank_of_data=0)
            x0_bd = rel_idx0_bd * px0_bd
            x1_bd = BaseData(
                signal=np.zeros_like(x0_bd.signal),
                units=x0_bd.units,
            )

        else:  # RoD == 2
            # image dimensions
            n0, n1 = spatial_shape
            # Axis 1 (columns) → x0, Axis 0 (rows) → x1
            idx0_bd = self._make_index_basedata(shape=(n0, n1), axis=0)
            idx1_bd = self._make_index_basedata(shape=(n0, n1), axis=1)

            rel_idx0_bd = idx0_bd - beam_center_bd.indexed(0, rank_of_data=0)
            rel_idx1_bd = idx1_bd - beam_center_bd.indexed(1, rank_of_data=0)

            x0_bd = rel_idx0_bd * px0_bd
            x1_bd = rel_idx1_bd * px1_bd

        # Common for RoD = 1, 2
        r_perp_bd = ((x0_bd**2) + (x1_bd**2)).sqrt()
        R_bd = ((r_perp_bd**2) + (detector_distance_bd**2)).sqrt()

        return x0_bd, x1_bd, r_perp_bd, R_bd

    # ------------------------------------------------------------------
    # Derived quantities: angles, Q, Psi, solid angle
    # ------------------------------------------------------------------

    def _compute_angles(
        self,
        r_perp_bd: BaseData,
        detector_distance_bd: BaseData,
    ) -> Tuple[BaseData, BaseData, BaseData]:
        """
        Compute 2θ, θ, and sin(θ) as BaseData.
        """
        ratio_bd = r_perp_bd / detector_distance_bd  # dimensionless
        two_theta_bd = ratio_bd.arctan()  # radians
        theta_bd = 0.5 * two_theta_bd  # radians
        sin_theta_bd = theta_bd.sin()  # dimensionless
        return two_theta_bd, theta_bd, sin_theta_bd

    def _compute_Q(
        self,
        sin_theta_bd: BaseData,
        wavelength_bd: BaseData,
    ) -> BaseData:
        """
        Compute scattering vector magnitude Q as BaseData.

        Q = (4π / λ) * sin(θ)
        """
        four_pi = 4.0 * np.pi
        return (four_pi * sin_theta_bd) / wavelength_bd

    def _compute_Q_components(
        self,
        Q_bd: BaseData,
        x0_bd: BaseData,
        x1_bd: BaseData,
        r_perp_bd: BaseData,
    ) -> Tuple[BaseData, BaseData, BaseData]:
        """
        Compute Q0, Q1, Q2 components as BaseData. Q2 is set to zero for a flat detector.
        """
        safe_signal = np.where(r_perp_bd.signal == 0.0, 1.0, r_perp_bd.signal)

        r_perp_safe_bd = BaseData(
            signal=safe_signal,
            units=r_perp_bd.units,
        )

        dir0_bd = x0_bd / r_perp_safe_bd  # cos Psi
        dir1_bd = x1_bd / r_perp_safe_bd  # sin Psi

        Q0_bd = Q_bd * dir0_bd
        Q1_bd = Q_bd * dir1_bd
        Q2_bd = BaseData(
            signal=np.zeros_like(Q_bd.signal),
            units=Q_bd.units,
        )

        return Q0_bd, Q1_bd, Q2_bd

    def _compute_psi(
        self,
        x0_bd: BaseData,
        x1_bd: BaseData,
    ) -> BaseData:
        """
        Compute azimuthal angle Psi from nominal coordinates only (no propagated uncertainty).

        Psi = atan2(x1, x0)
        """
        psi_signal = np.arctan2(x1_bd.signal, x0_bd.signal)
        return BaseData(
            signal=psi_signal,
            units=ureg.radian,
        )

    def _compute_solid_angle(
        self,
        R_bd: BaseData,
        px0_bd: BaseData,
        px1_bd: BaseData,
        detector_distance_bd: BaseData,
    ) -> BaseData:
        """
        Compute solid angle per pixel (Omega) as BaseData.

        Approximation:
            dΩ ≈ A * D / R³
        with A = pixel area (px0 * px1), D = detector distance, R = ray length.
        """
        area_bd = px0_bd * px1_bd
        R3_bd = R_bd**3
        Omega_bd = (area_bd * detector_distance_bd) / R3_bd  # dimensionless (sr)
        return Omega_bd

    # ------------------------------------------------------------------
    # Main execution methods
    # ------------------------------------------------------------------

    def prepare_execution(self):
        """
        Precalculate Q, Q0, Q1, Q2, Psi, TwoTheta, and Omega as BaseData objects and
        store them in self._prepared_data.
        """
        super().prepare_execution()

        # 1. Signal & detector dimensions
        pkey = self.configuration.get("with_processing_keys")
        signal_bd: BaseData = self.processing_data[pkey[0]]["signal"]
        RoD = signal_bd.rank_of_data
        spatial_shape: tuple[int, ...] = signal_bd.shape[-RoD:] if RoD > 0 else ()

        # 2. Load and validate geometry
        geom = self._load_geometry()
        self._validate_geometry(geom, RoD, spatial_shape)

        detector_distance_bd = geom["detector_distance"]
        pixel_size_bd = geom["pixel_size"]
        beam_center_bd = geom["beam_center"]
        wavelength_bd = geom["wavelength"]

        # 3. Extract pixel pitches along Q0/Q1
        px0_bd = pixel_size_bd.indexed(0, rank_of_data=0)
        px1_bd = pixel_size_bd.indexed(1, rank_of_data=0)

        # 4. Coordinates (x0, x1, r_perp, R)
        x0_bd, x1_bd, r_perp_bd, R_bd = self._compute_coordinates(
            RoD=RoD,
            spatial_shape=spatial_shape,
            beam_center_bd=beam_center_bd,
            px0_bd=px0_bd,
            px1_bd=px1_bd,
            detector_distance_bd=detector_distance_bd,
        )

        # 5. Angles: 2θ, θ, sin θ
        two_theta_bd, theta_bd, sin_theta_bd = self._compute_angles(
            r_perp_bd=r_perp_bd,
            detector_distance_bd=detector_distance_bd,
        )

        # 6. Q magnitude
        Q_bd = self._compute_Q(
            sin_theta_bd=sin_theta_bd,
            wavelength_bd=wavelength_bd,
        )

        # 7. Q components
        Q0_bd, Q1_bd, Q2_bd = self._compute_Q_components(
            Q_bd=Q_bd,
            x0_bd=x0_bd,
            x1_bd=x1_bd,
            r_perp_bd=r_perp_bd,
        )

        # 8. Psi
        Psi_bd = self._compute_psi(
            x0_bd=x0_bd,
            x1_bd=x1_bd,
        )

        # 9. Solid angle (Omega)
        Omega_bd = self._compute_solid_angle(
            R_bd=R_bd,
            px0_bd=px0_bd,
            px1_bd=px1_bd,
            detector_distance_bd=detector_distance_bd,
        )

        # 10. Set rank_of_data on outputs and stash in prepared_data
        for bd in (Q_bd, Q0_bd, Q1_bd, Q2_bd, Psi_bd, theta_bd, Omega_bd):
            bd.rank_of_data = RoD

        self._prepared_data = {
            "Q": Q_bd,
            "Q0": Q0_bd,
            "Q1": Q1_bd,
            "Q2": Q2_bd,
            "Psi": Psi_bd,
            "TwoTheta": two_theta_bd,
            "Omega": Omega_bd,
        }

    def calculate(self):
        """
        Add Q, Q0, Q1, Q2, Psi, TwoTheta, and Omega (solid angle) as BaseData objects
        to the databundles specified in 'with_processing_keys'.
        """
        data = self.processing_data
        output: Dict[str, object] = {}

        with_keys = self.configuration.get("with_processing_keys", [])
        if not with_keys:
            return output

        for key in with_keys:
            databundle = data.get(key)
            if databundle is None:
                continue

            databundle["Q"] = self._prepared_data["Q"]
            databundle["Q0"] = self._prepared_data["Q0"]
            databundle["Q1"] = self._prepared_data["Q1"]
            databundle["Q2"] = self._prepared_data["Q2"]
            databundle["Psi"] = self._prepared_data["Psi"]
            databundle["TwoTheta"] = self._prepared_data["TwoTheta"]
            databundle["Omega"] = self._prepared_data["Omega"]

            output[key] = databundle

        return output
