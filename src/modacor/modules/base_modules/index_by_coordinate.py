# SPDX-License-Identifier: BSD-3-Clause
"""Assign one-dimensional coordinate values to bins."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from modacor import ureg
from modacor.dataclasses.basedata import BaseData
from modacor.dataclasses.databundle import DataBundle
from modacor.dataclasses.process_step import ProcessStep
from modacor.dataclasses.process_step_describer import ProcessStepDescriber
from modacor.models.binning import assign_bin_indices, generate_bin_edges, validate_bin_edges

__all__ = ["IndexByCoordinate"]
__version__ = "20261001.1"


class IndexByCoordinate(ProcessStep):
    """Assign an integer bin index from one arbitrary ``BaseData`` coordinate.

    The step is deliberately independent of signal values and masks. It writes
    both the index map and the physical edges used to construct it. The edges
    are diagnostic output; indexed reducers need only the index map.
    """

    documentation = ProcessStepDescriber(
        calling_name="Index by Coordinate",
        calling_id="IndexByCoordinate",
        calling_module_path=Path(__file__),
        calling_version=__version__,
        required_data_keys=[],
        arguments={
            "with_processing_keys": {
                "type": (str, list, type(None)),
                "default": None,
                "doc": "ProcessingData key(s) whose coordinate should be indexed.",
            },
            "coordinate_key": {
                "type": str,
                "default": "Q",
                "doc": "BaseData coordinate used to assign bins.",
                "dependency_role": "processing_read_basedata_key",
            },
            "index_key": {
                "type": str,
                "default": "bin_index",
                "doc": "Output BaseData key for the integer index map.",
                "dependency_role": "processing_write_basedata_key",
            },
            "edges_key": {
                "type": str,
                "default": "bin_edges",
                "doc": "Output BaseData key recording the physical bin edges.",
                "dependency_role": "processing_write_basedata_key",
            },
            "bin_edges": {
                "type": (list, tuple, type(None)),
                "default": None,
                "doc": "Explicit bin edges in bin_units; mutually exclusive with generated-edge settings.",
            },
            "bin_min": {
                "type": (float, int, type(None)),
                "default": None,
                "doc": "Generated lower edge in bin_units; inferred when omitted.",
            },
            "bin_max": {
                "type": (float, int, type(None)),
                "default": None,
                "doc": "Generated upper edge in bin_units; inferred when omitted.",
            },
            "bin_units": {
                "type": (str, type(None)),
                "default": None,
                "doc": "Units for configured limits or edges; defaults to coordinate units.",
            },
            "n_bins": {
                "type": (int, type(None)),
                "default": None,
                "doc": "Generated bin count; defaults to 100 when omitted.",
            },
            "spacing": {
                "type": (str, type(None)),
                "default": None,
                "doc": "Generated edge spacing, 'linear' or 'log'; defaults to 'linear'.",
            },
        },
        modifies={
            "bin_index": ["signal", "units", "axes"],
            "bin_edges": ["signal", "units"],
        },
        step_keywords=["indexing", "binning", "coordinate", "reduction"],
        step_doc="Assign one coordinate value per input point to a one-dimensional bin index.",
        step_note=(
            "Bins are left-inclusive and right-exclusive except for the included final right edge. "
            "Non-finite and out-of-range coordinates receive index -1. Masks are applied by downstream reducers."
        ),
    )

    @staticmethod
    def _validated_key(configuration: dict, name: str) -> str:
        value = str(configuration[name]).strip()
        if not value:
            raise ValueError(f"IndexByCoordinate: {name} must be a non-empty string.")
        return value

    def _edges_for(self, coordinate: BaseData) -> np.ndarray:
        units_config = self.configuration.get("bin_units")
        configured_units = coordinate.units if units_config is None else ureg.Unit(units_config)
        try:
            conversion = coordinate.units.m_from(configured_units)
        except Exception as exc:  # Pint exposes several dimensionality error types
            raise ValueError(
                f"IndexByCoordinate: bin_units {configured_units!s} are incompatible with "
                f"coordinate units {coordinate.units!s}."
            ) from exc

        explicit_edges = self.configuration.get("bin_edges")
        generated_settings = {
            "bin_min": self.configuration.get("bin_min"),
            "bin_max": self.configuration.get("bin_max"),
            "n_bins": self.configuration.get("n_bins"),
            "spacing": self.configuration.get("spacing"),
        }
        if explicit_edges is not None:
            configured = [name for name, value in generated_settings.items() if value is not None]
            if configured:
                raise ValueError(
                    "IndexByCoordinate: bin_edges is mutually exclusive with generated-edge settings: "
                    + ", ".join(configured)
                    + "."
                )
            return validate_bin_edges(np.asarray(explicit_edges, dtype=float) * conversion)

        n_bins = generated_settings["n_bins"]
        if n_bins is None:
            n_bins = 100
        spacing = generated_settings["spacing"]
        if spacing is None:
            spacing = "linear"
        lower = generated_settings["bin_min"]
        upper = generated_settings["bin_max"]
        lower_value = None if lower is None else float(lower) * conversion
        upper_value = None if upper is None else float(upper) * conversion
        return generate_bin_edges(
            coordinate.signal,
            lower=lower_value,
            upper=upper_value,
            n_bins=int(n_bins),
            spacing=str(spacing),
        )

    def calculate(self) -> dict[str, DataBundle]:
        output: dict[str, DataBundle] = {}
        coordinate_key = self._validated_key(self.configuration, "coordinate_key")
        index_key = self._validated_key(self.configuration, "index_key")
        edges_key = self._validated_key(self.configuration, "edges_key")
        if len({coordinate_key, index_key, edges_key}) != 3:
            raise ValueError("IndexByCoordinate: coordinate_key, index_key, and edges_key must be distinct.")

        for processing_key in self._normalised_processing_keys():
            bundle = self.processing_data.get(processing_key)
            if bundle is None:
                raise KeyError(f"IndexByCoordinate: ProcessingData key {processing_key!r} not found.")
            if coordinate_key not in bundle:
                raise KeyError(
                    f"IndexByCoordinate: DataBundle {processing_key!r} has no coordinate {coordinate_key!r}."
                )

            coordinate = bundle[coordinate_key]
            edges = self._edges_for(coordinate)
            indices = assign_bin_indices(coordinate.signal, edges)
            bundle[index_key] = BaseData(
                signal=indices,
                units=ureg.dimensionless,
                uncertainties={},
                weights=np.array(1.0),
                axes=list(coordinate.axes),
                rank_of_data=coordinate.rank_of_data,
            )
            bundle[edges_key] = BaseData(
                signal=edges,
                units=coordinate.units,
                uncertainties={},
                weights=np.array(1.0),
                axes=[],
                rank_of_data=1,
            )
            output[processing_key] = bundle

        return output
