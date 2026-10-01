# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

__all__ = ["ConcatenateDatabundles"]
__version__ = "20260929.1"

from pathlib import Path

import numpy as np

from modacor import ureg
from modacor.dataclasses.basedata import BaseData
from modacor.dataclasses.databundle import DataBundle
from modacor.dataclasses.process_step import ProcessStep, ProcessStepDependencies, normalize_processing_key_values
from modacor.dataclasses.process_step_describer import ProcessStepDescriber


class ConcatenateDatabundles(ProcessStep):
    """Pool matching one-dimensional BaseData entries into a new DataBundle."""

    documentation = ProcessStepDescriber(
        calling_name="Concatenate DataBundles",
        calling_id="ConcatenateDatabundles",
        calling_module_path=Path(__file__),
        calling_version=__version__,
        required_data_keys=[],
        modifies={
            "configured data keys": ["signal", "uncertainties", "weights", "units", "axes"],
            "source_index": ["signal"],
        },
        arguments={
            "with_processing_keys": {
                "type": list,
                "required": True,
                "default": None,
                "doc": "Input DataBundle keys, in concatenation order.",
            },
            "data_keys": {
                "type": (list, str),
                "required": True,
                "default": ["signal", "Q"],
                "doc": "Matching one-dimensional BaseData entries to concatenate.",
            },
            "output_processing_key": {
                "type": str,
                "required": True,
                "default": "concatenated",
                "doc": "ProcessingData key receiving the pooled DataBundle.",
            },
            "sort_by": {
                "type": (str, type(None)),
                "default": None,
                "doc": "Optional concatenated data key used for coordinated stable sorting.",
            },
            "descending": {
                "type": bool,
                "default": False,
                "doc": "Sort descending instead of ascending when sort_by is set.",
            },
            "source_index_key": {
                "type": (str, type(None)),
                "default": "source_index",
                "doc": "Optional output key recording each point's zero-based input-bundle index.",
            },
        },
        step_keywords=["concatenate", "pool", "curves", "sort"],
        step_doc="Concatenate matching 1D BaseData entries, optionally sorting every entry together.",
        step_note=(
            "Input order is preserved when sort_by is None. Units are converted to those of the first input. "
            "Uncertainty component names must match across inputs. Sorting is stable and is not required by "
            "IndexByCoordinate."
        ),
    )

    def _data_keys(self) -> list[str]:
        configured = self.configuration.get("data_keys", ["signal", "Q"])
        keys = [configured] if isinstance(configured, str) else list(configured)
        keys = [str(key).strip() for key in keys]
        if not keys or any(not key for key in keys):
            raise ValueError("ConcatenateDatabundles data_keys must contain non-empty strings.")
        if len(set(keys)) != len(keys):
            raise ValueError("ConcatenateDatabundles data_keys must not contain duplicates.")
        return keys

    def _output_key(self) -> str:
        key = str(self.configuration.get("output_processing_key", "concatenated")).strip()
        if not key:
            raise ValueError("ConcatenateDatabundles output_processing_key must not be empty.")
        return key

    def dependency_contract(self) -> ProcessStepDependencies:
        processing_keys = normalize_processing_key_values(self.configuration.get("with_processing_keys"))
        if not processing_keys:
            return ProcessStepDependencies(processing_reads={"*"}, processing_writes={"*"})
        data_keys = self._data_keys()
        return ProcessStepDependencies(
            processing_reads={
                f"{processing_key}.{data_key}" for processing_key in processing_keys for data_key in data_keys
            },
            processing_writes={f"{self._output_key()}.*"},
        )

    @staticmethod
    def _axis_key(bundle: DataBundle, axis: BaseData | None) -> str | None:
        if axis is None:
            return None
        matches = [key for key, value in bundle.items() if value is axis]
        if len(matches) != 1:
            raise ValueError(
                "ConcatenateDatabundles can preserve axes only when each axis is a uniquely named "
                "BaseData entry in its input DataBundle."
            )
        return matches[0]

    def calculate(self) -> dict[str, DataBundle]:  # noqa: C901 - validation is intentionally explicit
        processing_keys = self._normalised_processing_keys()
        if not processing_keys:
            raise ValueError("ConcatenateDatabundles requires at least one input processing key.")
        data_keys = self._data_keys()
        output_key = self._output_key()
        source_index_key = self.configuration.get("source_index_key", "source_index")
        if source_index_key is not None:
            source_index_key = str(source_index_key).strip()
            if not source_index_key:
                raise ValueError("ConcatenateDatabundles source_index_key must be non-empty or None.")
            if source_index_key in data_keys:
                raise ValueError("ConcatenateDatabundles source_index_key conflicts with a configured data key.")

        bundles: list[DataBundle] = []
        lengths: list[int] = []
        for processing_key in processing_keys:
            if processing_key not in self.processing_data:
                raise KeyError(f"ConcatenateDatabundles input not found: {processing_key!r}.")
            bundle = self.processing_data[processing_key]
            missing = [key for key in data_keys if key not in bundle]
            if missing:
                raise KeyError(f"ConcatenateDatabundles {processing_key!r} is missing data keys {missing!r}.")
            shapes = {bundle[key].shape for key in data_keys}
            if len(shapes) != 1:
                raise ValueError(f"ConcatenateDatabundles data entries in {processing_key!r} do not share one shape.")
            shape = next(iter(shapes))
            if len(shape) != 1:
                raise ValueError("ConcatenateDatabundles currently accepts one-dimensional point series only.")
            bundles.append(bundle)
            lengths.append(shape[0])

        output = DataBundle()
        axis_key_map: dict[str, list[str | None]] = {}
        for data_key in data_keys:
            reference = bundles[0][data_key]
            reference_uncertainties = set(reference.uncertainties)
            signals: list[np.ndarray] = []
            weights: list[np.ndarray] = []
            uncertainties: dict[str, list[np.ndarray]] = {name: [] for name in reference_uncertainties}
            axis_key_map[data_key] = [self._axis_key(bundles[0], axis) for axis in reference.axes]

            for processing_key, bundle, length in zip(processing_keys, bundles, lengths, strict=True):
                current = bundle[data_key].copy(with_axes=False)
                current.signal = np.asarray(current.signal, dtype=float)
                current.to_units(reference.units)
                if set(current.uncertainties) != reference_uncertainties:
                    raise ValueError(
                        "ConcatenateDatabundles requires matching uncertainty keys for "
                        f"{data_key!r}; {processing_key!r} has {sorted(current.uncertainties)!r}, "
                        f"expected {sorted(reference_uncertainties)!r}."
                    )
                current_axis_keys = [self._axis_key(bundle, axis) for axis in bundle[data_key].axes]
                if current_axis_keys != axis_key_map[data_key]:
                    raise ValueError(f"ConcatenateDatabundles axis references differ for data key {data_key!r}.")
                signals.append(np.asarray(current.signal))
                weights.append(np.broadcast_to(np.asarray(current.weights, dtype=float), (length,)).copy())
                for name in reference_uncertainties:
                    uncertainties[name].append(
                        np.broadcast_to(np.asarray(current.uncertainties[name], dtype=float), (length,)).copy()
                    )

            output[data_key] = BaseData(
                signal=np.concatenate(signals),
                units=reference.units,
                uncertainties={name: np.concatenate(parts) for name, parts in uncertainties.items()},
                weights=np.concatenate(weights),
                axes=[],
                rank_of_data=1,
            )

        if source_index_key is not None:
            output[source_index_key] = BaseData(
                signal=np.concatenate([np.full(length, index, dtype=float) for index, length in enumerate(lengths)]),
                units=ureg.dimensionless,
                rank_of_data=1,
            )

        sort_by = self.configuration.get("sort_by")
        if sort_by is not None:
            sort_by = str(sort_by).strip()
            if sort_by not in output:
                raise KeyError(f"ConcatenateDatabundles sort_by key {sort_by!r} is not an output data key.")
            values = np.asarray(output[sort_by].signal, dtype=float)
            order = np.argsort(-values if bool(self.configuration.get("descending", False)) else values, kind="stable")
            for data_key, basedata in list(output.items()):
                output[data_key] = basedata.indexed(order, rank_of_data=1)

        for data_key, axis_keys in axis_key_map.items():
            missing_axis_keys = [key for key in axis_keys if key is not None and key not in output]
            if missing_axis_keys:
                raise ValueError(
                    f"ConcatenateDatabundles cannot preserve {data_key!r} axes because "
                    f"{missing_axis_keys!r} are not included in data_keys."
                )
            output[data_key].axes = [output[key] if key is not None else None for key in axis_keys]

        output.default_plot = bundles[0].default_plot if bundles[0].default_plot in output else None
        output.description = "Concatenated from " + ", ".join(processing_keys)
        self.processing_data[output_key] = output
        return {output_key: output}
