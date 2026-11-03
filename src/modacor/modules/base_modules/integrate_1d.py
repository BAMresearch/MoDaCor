# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

__all__ = ["Integrate1D"]
__version__ = "20261103.2"

from pathlib import Path

import numpy as np

from modacor.dataclasses.basedata import BaseData, _validate_uncertainties
from modacor.dataclasses.databundle import DataBundle
from modacor.dataclasses.process_step import ProcessStep, ProcessStepDependencies, normalize_processing_key_values
from modacor.dataclasses.process_step_describer import ProcessStepDescriber
from modacor.models.integration import IntegrationDomain, integrate_samples_1d, prepare_integration_domain


class Integrate1D(ProcessStep):
    """Integrate sampled curves over a common one-dimensional domain."""

    documentation = ProcessStepDescriber(
        calling_name="Integrate one-dimensional curves",
        calling_id="Integrate1D",
        calling_module_path=Path(__file__),
        calling_version=__version__,
        required_data_keys=["signal", "q"],
        modifies={"integral": ["signal", "uncertainties", "units"]},
        arguments={
            "with_processing_keys": {
                "type": list,
                "required": True,
                "default": None,
                "doc": "DataBundles integrated over their common valid domain.",
            },
            "signal_key": {
                "type": str,
                "default": "signal",
                "doc": "One-dimensional BaseData entry to integrate.",
            },
            "axis_key": {
                "type": str,
                "default": "q",
                "doc": "One-dimensional integration coordinate BaseData key.",
            },
            "method": {
                "type": str,
                "default": "trapezoid",
                "doc": "Quadrature rule: trapezoid or simpson.",
            },
            "mask_key": {
                "type": (str, type(None)),
                "default": None,
                "doc": "Optional mask BaseData key; nonzero values are excluded.",
            },
            "sort_axis": {
                "type": bool,
                "default": False,
                "doc": "Stably sort the common valid coordinate before integration.",
            },
            "duplicate_axis": {
                "type": str,
                "default": "error",
                "doc": "Handling of repeated coordinates: error or mean.",
            },
            "output_key": {
                "type": str,
                "default": "integral",
                "doc": "Output key when the integral remains in its input DataBundle.",
            },
            "output_processing_keys": {
                "type": (list, type(None)),
                "default": None,
                "doc": "Optional new DataBundle key per input; each result is stored as signal.",
            },
        },
        step_keywords=["integrate", "quadrature", "trapezoid", "simpson", "1D"],
        step_doc=(
            "Integrate sampled 1D BaseData while propagating coordinate units and independent uncertainty components."
        ),
        step_note=(
            "The shared coordinate may be nonuniform. It must be monotonic unless sort_axis is enabled. Repeated "
            "coordinates are rejected by default; duplicate_axis='mean' replaces each run with its arithmetic "
            "mean signal and independently propagated uncertainty of that mean. Invalid or masked samples in any "
            "input are omitted from every integral. The axis is treated as exact (fixed abscissae); axis "
            "uncertainties are not propagated."
        ),
        step_reference="DOI 10.1107/S1600576717015096",
    )

    def dependency_contract(self) -> ProcessStepDependencies:
        cfg = self.configuration
        processing_keys = normalize_processing_key_values(cfg.get("with_processing_keys"))
        if not processing_keys:
            return ProcessStepDependencies(processing_reads={"*"}, processing_writes={"*"})

        signal_key = str(cfg["signal_key"])
        axis_key = str(cfg["axis_key"])
        mask_key = cfg.get("mask_key")
        read_keys = {signal_key, axis_key}
        if mask_key is not None:
            read_keys.add(str(mask_key))
        reads = {f"{processing_key}.{basedata_key}" for processing_key in processing_keys for basedata_key in read_keys}

        output_processing_keys = cfg.get("output_processing_keys")
        if output_processing_keys is None:
            output_key = str(cfg["output_key"])
            writes = {f"{processing_key}.{output_key}" for processing_key in processing_keys}
        else:
            output_processing_keys = [str(key) for key in output_processing_keys]
            if len(output_processing_keys) != len(processing_keys):
                raise ValueError("output_processing_keys must contain one key per input bundle.")
            writes = {f"{output_processing_key}.*" for output_processing_key in output_processing_keys}

        return ProcessStepDependencies(processing_reads=reads, processing_writes=writes)

    def _collect_common_domain(
        self,
        processing_keys: list[str],
        signal_key: str,
        axis_key: str,
        mask_key: str | None,
    ) -> tuple[list[BaseData], BaseData, np.ndarray, np.ndarray]:
        signals = [self.processing_data[key][signal_key] for key in processing_keys]
        reference_axis = self.processing_data[processing_keys[0]][axis_key].copy(with_axes=False)
        axis = np.asarray(reference_axis.signal, dtype=float).squeeze()
        if axis.ndim != 1 or axis.size < 2:
            raise ValueError("Integrate1D requires a one-dimensional axis with at least two points.")

        common = np.isfinite(axis)
        for processing_key, signal in zip(processing_keys, signals, strict=True):
            values = np.asarray(signal.signal, dtype=float).squeeze()
            if values.ndim != 1 or values.shape != axis.shape:
                raise ValueError("Integrate1D requires matching one-dimensional signal and axis arrays.")
            other_axis = self.processing_data[processing_key][axis_key].copy(with_axes=False)
            other_axis.to_units(reference_axis.units)
            if not np.allclose(np.asarray(other_axis.signal).squeeze(), axis, rtol=1.0e-12, atol=1.0e-15):
                raise ValueError("Integrate1D input bundles must share the same axis.")

            common &= np.isfinite(values)
            weights = np.broadcast_to(np.asarray(signal.weights, dtype=float), values.shape)
            common &= np.isfinite(weights) & (weights > 0.0)
            bundle = self.processing_data[processing_key]
            if mask_key is not None and mask_key in bundle:
                common &= np.asarray(bundle[mask_key].signal).squeeze() == 0
            for component in signal.uncertainties.values():
                common &= np.isfinite(np.broadcast_to(component, values.shape))
        return signals, reference_axis, axis, common

    @staticmethod
    def _integrate_signal(signal: BaseData, reference_axis: BaseData, domain: IntegrationDomain) -> BaseData:
        _validate_uncertainties(signal)
        _validate_uncertainties(reference_axis)
        nominal, uncertainties = integrate_samples_1d(signal.signal, signal.uncertainties, domain)
        return BaseData(
            signal=nominal, units=signal.units * reference_axis.units, uncertainties=uncertainties, rank_of_data=0
        )

    @staticmethod
    def _normalise_output_processing_keys(
        configured_keys: list[str] | None,
        processing_keys: list[str],
    ) -> list[str] | None:
        if configured_keys is None:
            return None
        output_processing_keys = [str(key) for key in configured_keys]
        if len(output_processing_keys) != len(processing_keys):
            raise ValueError("output_processing_keys must contain one key per input bundle.")
        return output_processing_keys

    def _store_integral(
        self,
        *,
        processing_key: str,
        integral: BaseData,
        index: int,
        output_processing_keys: list[str] | None,
        output_key: str,
        method: str,
        signal_key: str,
    ) -> tuple[str, DataBundle]:
        if output_processing_keys is None:
            bundle = self.processing_data[processing_key]
            bundle[output_key] = integral
            return processing_key, bundle

        result_key = output_processing_keys[index]
        bundle = DataBundle(signal=integral)
        bundle.default_plot = "signal"
        bundle.description = f"{method} integral of {processing_key}.{signal_key}"
        self.processing_data[result_key] = bundle
        return result_key, bundle

    def calculate(self) -> dict[str, DataBundle]:
        cfg = self.configuration
        processing_keys = self._normalised_processing_keys()
        if not processing_keys:
            raise ValueError("Integrate1D requires at least one processing key.")
        signal_key = str(cfg["signal_key"])
        axis_key = str(cfg["axis_key"])
        method = str(cfg["method"]).strip().lower()
        mask_key = cfg.get("mask_key")
        duplicate_axis = str(cfg["duplicate_axis"]).strip().lower()

        signals, reference_axis, axis, common = self._collect_common_domain(
            processing_keys,
            signal_key,
            axis_key,
            mask_key,
        )
        domain = prepare_integration_domain(
            axis,
            common,
            method=method,
            sort_axis=bool(cfg["sort_axis"]),
            duplicate_axis=duplicate_axis,
        )
        output_processing_keys = self._normalise_output_processing_keys(
            cfg.get("output_processing_keys"),
            processing_keys,
        )

        output: dict[str, DataBundle] = {}
        output_key = str(cfg["output_key"])
        for index, (processing_key, signal) in enumerate(zip(processing_keys, signals, strict=True)):
            integral = self._integrate_signal(signal, reference_axis, domain)
            result_key, bundle = self._store_integral(
                processing_key=processing_key,
                integral=integral,
                index=index,
                output_processing_keys=output_processing_keys,
                output_key=output_key,
                method=method,
                signal_key=signal_key,
            )
            output[result_key] = bundle
        return output
