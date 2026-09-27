# SPDX-License-Identifier: BSD-3-Clause

"""Construct USAXS scattering geometry from an analyser-crystal rocking angle."""

from __future__ import annotations

__all__ = ["XSGeometryFromAnalyserAngle"]
__version__ = "20260927.1"

from pathlib import Path

import numpy as np

from modacor import ureg
from modacor.dataclasses.basedata import BaseData
from modacor.dataclasses.databundle import DataBundle
from modacor.dataclasses.helpers import basedata_from_sources
from modacor.dataclasses.process_step import (
    ProcessStep,
    ProcessStepDependencies,
    processing_key_patterns,
    source_refs_from_references,
)
from modacor.dataclasses.process_step_describer import ProcessStepDescriber
from modacor.modules.helpers.scattering.detector_data import (
    prepare_static_scalar,
    require_scalar,
)


class XSGeometryFromAnalyserAngle(ProcessStep):
    """Apply an analyser zero and angular scale, then calculate USAXS q."""

    documentation = ProcessStepDescriber(
        calling_name="Construct USAXS geometry from an analyser angle",
        calling_id="XSGeometryFromAnalyserAngle",
        calling_module_path=Path(__file__),
        calling_version=__version__,
        required_data_keys=[],
        modifies={
            "TwoTheta": ["signal", "uncertainties", "units"],
            "Psi": ["signal", "units"],
            "signed_q": ["signal", "uncertainties", "units"],
            "Q": ["signal", "uncertainties", "units"],
            "applied_angular_zero": ["signal", "uncertainties", "units"],
            "applied_angular_multiplier": ["signal", "uncertainties"],
        },
        arguments={
            "with_processing_keys": {
                "type": list,
                "required": True,
                "default": None,
                "doc": "DataBundles receiving geometry calculated from their analyser angle.",
            },
            "angle_key": {
                "type": str,
                "default": "analyser_angle",
                "doc": "BaseData key containing the raw analyser-crystal rocking angle.",
                "dependency_role": "processing_read_basedata_key",
            },
            "angular_zero_key": {
                "type": (str, type(None)),
                "default": None,
                "doc": "Optional scalar BaseData key containing the analyser zero.",
            },
            "angular_zero_processing_key": {
                "type": (str, type(None)),
                "default": None,
                "doc": (
                    "Optional DataBundle containing angular_zero_key. When omitted, each "
                    "target bundle supplies its own zero."
                ),
            },
            "angular_multiplier_key": {
                "type": (str, type(None)),
                "default": None,
                "doc": "Optional scalar BaseData key containing the angular calibration multiplier.",
                "dependency_role": "processing_read_basedata_key",
            },
            "angular_multiplier_source": {
                "type": (str, type(None)),
                "default": None,
                "doc": "Optional IoSources reference for the angular calibration multiplier.",
            },
            "angular_multiplier_units_source": {
                "type": (str, type(None)),
                "default": None,
                "doc": "Optional IoSources reference for angular multiplier units.",
            },
            "angular_multiplier_uncertainties_sources": {
                "type": dict,
                "default": {},
                "doc": "Optional mapping of angular multiplier uncertainty names to IoSources references.",
            },
            "wavelength_source": {
                "type": str,
                "required": True,
                "default": None,
                "doc": "IoSources reference for a scalar wavelength.",
            },
            "wavelength_units_source": {
                "type": (str, type(None)),
                "default": None,
                "doc": "Optional IoSources reference for wavelength units.",
            },
            "wavelength_uncertainties_sources": {
                "type": dict,
                "default": {},
                "doc": "Optional mapping of wavelength uncertainty names to IoSources references.",
            },
            "psi_key": {
                "type": (str, type(None)),
                "default": None,
                "doc": "Optional scalar BaseData key containing the nominal trajectory azimuth.",
                "dependency_role": "processing_read_basedata_key",
            },
            "psi_source": {
                "type": (str, type(None)),
                "default": None,
                "doc": "Optional IoSources reference for the nominal trajectory azimuth.",
            },
            "psi_units_source": {
                "type": (str, type(None)),
                "default": None,
                "doc": "Optional IoSources reference for nominal-azimuth units.",
            },
            "two_theta_units": {
                "type": str,
                "default": "radian",
                "doc": "Output units for signed TwoTheta.",
            },
            "q_units": {
                "type": str,
                "default": "1 / angstrom",
                "doc": "Output units for signed_q and Q.",
            },
        },
        step_keywords=["USAXS", "analyser", "geometry", "Q", "TwoTheta", "Psi"],
        step_doc=(
            "Construct the one-dimensional USAXS scattering geometry from a raw "
            "analyser-crystal rocking angle and its angular-zero calibration."
        ),
        step_note=(
            "This one-angle mapping is specific to an analyser-crystal USAXS geometry; "
            "it is not a general multi-axis diffractometer transformation."
        ),
    )

    output_keys: tuple[str, ...] = (
        "TwoTheta",
        "Psi",
        "signed_q",
        "Q",
        "applied_angular_zero",
        "applied_angular_multiplier",
    )

    def _load_from_sources(self, key: str) -> BaseData:
        """Load one BaseData using the standard geometry source convention."""
        return basedata_from_sources(
            io_sources=self.io_sources,
            signal_source=self.configuration.get(f"{key}_source"),
            units_source=self.configuration.get(f"{key}_units_source"),
            uncertainty_sources=self.configuration.get(f"{key}_uncertainties_sources", {}),
        )

    def dependency_contract(self) -> ProcessStepDependencies:
        """Declare exact scan metadata reads and generated geometry writes."""
        cfg = self.configuration or {}
        processing_keys = cfg.get("with_processing_keys")
        reads = processing_key_patterns(
            processing_keys,
            basedata_key=cfg.get("angle_key", "analyser_angle"),
        )

        for config_key in (
            "angular_multiplier_key",
            "psi_key",
        ):
            basedata_key = cfg.get(config_key)
            if basedata_key is not None:
                reads |= processing_key_patterns(
                    processing_keys,
                    basedata_key=basedata_key,
                )

        zero_key = cfg.get("angular_zero_key")
        if zero_key is not None:
            zero_processing_key = cfg.get("angular_zero_processing_key") or processing_keys
            reads |= processing_key_patterns(
                zero_processing_key,
                basedata_key=zero_key,
            )

        writes = frozenset(
            pattern
            for output_key in self.output_keys
            for pattern in processing_key_patterns(
                processing_keys,
                basedata_key=output_key,
            )
        )
        return ProcessStepDependencies(
            source_refs=source_refs_from_references(cfg),
            processing_reads=reads,
            processing_writes=writes,
        )

    def _load_wavelength(self) -> BaseData:
        """Load and validate the wavelength from its IoSources reference."""
        wavelength = self._load_from_sources("wavelength")
        wavelength = prepare_static_scalar(
            wavelength,
            require_units=ureg.m,
            uncertainty_key="wavelength_jitter",
        )
        wavelength = require_scalar("wavelength", wavelength)
        wavelength.to_units("meter")
        value_m = float(np.asarray(wavelength.signal))
        if not np.isfinite(value_m) or value_m <= 0.0:
            raise ValueError("Wavelength must be finite and positive.")
        return wavelength

    def _angular_zero(self, processing_key: str, angle: BaseData) -> BaseData:
        """Resolve the optional scalar zero for one target bundle."""
        cfg = self.configuration
        zero_key = cfg.get("angular_zero_key")
        if zero_key is None:
            return BaseData(signal=np.asarray(0.0), units=angle.units, rank_of_data=0)
        source_key = cfg.get("angular_zero_processing_key") or processing_key
        source_bundle = self.processing_data[str(source_key)]
        if str(zero_key) not in source_bundle:
            raise KeyError(f"DataBundle {source_key!r} has no angular zero {zero_key!r}.")
        zero = require_scalar("angular_zero", source_bundle[str(zero_key)].copy(with_axes=False))
        zero.to_units(angle.units)
        return zero

    def _angular_multiplier(self, processing_key: str) -> BaseData:
        """Load the angular calibration from data or instrument metadata."""
        cfg = self.configuration
        key = cfg.get("angular_multiplier_key")
        source = cfg.get("angular_multiplier_source")
        if key is not None and source is not None:
            raise ValueError("Configure at most one of angular_multiplier_key and angular_multiplier_source.")
        if source is not None:
            multiplier = self._load_from_sources("angular_multiplier")
        elif key is not None:
            multiplier = self.processing_data[processing_key][str(key)].copy(with_axes=False)
        else:
            multiplier = BaseData(signal=np.asarray(1.0), units=ureg.dimensionless, rank_of_data=0)
        multiplier = prepare_static_scalar(
            multiplier,
            require_units=ureg.dimensionless,
            uncertainty_key="angular_multiplier_jitter",
        )
        multiplier = require_scalar("angular_multiplier", multiplier)
        multiplier.to_units("dimensionless")
        value = float(np.asarray(multiplier.signal))
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError("angular_multiplier must be finite and positive.")
        return multiplier

    def _psi_radian(self, processing_key: str) -> float:
        """Load the nominal USAXS trajectory azimuth from data or metadata."""
        cfg = self.configuration
        key = cfg.get("psi_key")
        source = cfg.get("psi_source")
        if (key is None) == (source is None):
            raise ValueError("Configure exactly one of psi_key or psi_source.")
        if source is not None:
            psi = self._load_from_sources("psi")
        else:
            psi = self.processing_data[processing_key][str(key)].copy(with_axes=False)
        psi = prepare_static_scalar(
            psi,
            require_units=ureg.radian,
            uncertainty_key="psi_jitter",
        )
        psi = require_scalar("psi", psi)
        psi.to_units("radian")
        return float(np.asarray(psi.signal))

    def _compute(
        self,
        *,
        angle: BaseData,
        angular_zero: BaseData,
        angular_multiplier: BaseData,
        wavelength: BaseData,
        psi_radian: float,
    ) -> dict[str, BaseData]:
        """Calculate USAXS geometry from prepared BaseData inputs."""
        cfg = self.configuration
        raw_angle_units = angle.units
        angle = angle.copy()
        angle.to_units("radian")
        angular_zero = angular_zero.copy(with_axes=False)
        angular_zero.to_units("radian")

        two_theta_radian = (angle - angular_zero) * angular_multiplier
        signed_q = (two_theta_radian / 2.0).sin() * (4.0 * np.pi) / wavelength
        signed_q.to_units(str(cfg.get("q_units", "1 / angstrom")))
        q = signed_q.copy()
        q.signal = np.abs(q.signal)

        two_theta = two_theta_radian.copy()
        two_theta.to_units(str(cfg.get("two_theta_units", "radian")))
        psi = BaseData(
            signal=np.full(angle.signal.shape, psi_radian),
            units=ureg.radian,
            weights=np.array(angle.weights, copy=True),
            axes=list(angle.axes),
            rank_of_data=angle.rank_of_data,
        )

        applied_zero = angular_zero.copy(with_axes=False)
        applied_zero.to_units(raw_angle_units)
        return {
            "TwoTheta": two_theta,
            "Psi": psi,
            "signed_q": signed_q,
            "Q": q,
            "applied_angular_zero": applied_zero,
            "applied_angular_multiplier": angular_multiplier.copy(with_axes=False),
        }

    def prepare_execution(self) -> None:
        """Prepare geometry for each scan, following the shared XSGeometry lifecycle."""
        super().prepare_execution()
        processing_keys = self._normalised_processing_keys()

        cfg = self.configuration
        angle_key = str(cfg.get("angle_key", "analyser_angle"))
        prepared: dict[str, dict[str, BaseData]] = {}
        for processing_key in processing_keys:
            if processing_key not in self.processing_data:
                raise KeyError(f"ProcessingData has no entry {processing_key!r}.")
            bundle = self.processing_data[processing_key]
            if angle_key not in bundle:
                raise KeyError(f"DataBundle {processing_key!r} has no analyser angle {angle_key!r}.")
            angle = bundle[angle_key]
            zero = self._angular_zero(processing_key, angle)
            wavelength = self._load_wavelength()
            multiplier = self._angular_multiplier(processing_key)
            psi_radian = self._psi_radian(processing_key)
            outputs = self._compute(
                angle=angle,
                angular_zero=zero,
                angular_multiplier=multiplier,
                wavelength=wavelength,
                psi_radian=psi_radian,
            )
            rank = int(angle.rank_of_data)
            for name in ("TwoTheta", "Psi", "signed_q", "Q"):
                outputs[name].rank_of_data = min(rank, outputs[name].signal.ndim)
            prepared[processing_key] = outputs
        self._prepared_data = prepared

    def calculate(self) -> dict[str, DataBundle]:
        """Attach the separately prepared geometry to each target DataBundle."""
        processing_keys = self._normalised_processing_keys()

        output: dict[str, DataBundle] = {}
        for processing_key in processing_keys:
            bundle = self.processing_data[processing_key]
            bundle.update(self._prepared_data[processing_key])
            output[processing_key] = bundle
        return output
