# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

__all__ = ["Integrate1D"]
__version__ = "20260925.1"

from pathlib import Path

import numpy as np
from scipy.integrate import simpson

from modacor.dataclasses.basedata import BaseData
from modacor.dataclasses.databundle import DataBundle
from modacor.dataclasses.process_step import ProcessStep
from modacor.dataclasses.process_step_describer import ProcessStepDescriber


class Integrate1D(ProcessStep):
    """Integrate sampled curves over a common one-dimensional domain."""

    documentation = ProcessStepDescriber(
        calling_name="Integrate one-dimensional curves",
        calling_id="Integrate1D",
        calling_module_path=Path(__file__),
        calling_version=__version__,
        required_data_keys=[],
        modifies={},
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
            "Integrate sampled 1D BaseData while propagating coordinate units and "
            "independent uncertainty components."
        ),
        step_note=(
            "The shared coordinate may be nonuniform but must be strictly monotonic. "
            "Invalid or masked samples in any input are omitted from every integral."
        ),
    )

    @staticmethod
    def _quadrature_weights(axis: np.ndarray, method: str) -> np.ndarray:
        """Return coefficients whose dot product with samples is the integral."""
        if method == "trapezoid":
            weights = np.empty(axis.size, dtype=float)
            weights[0] = 0.5 * (axis[1] - axis[0])
            weights[-1] = 0.5 * (axis[-1] - axis[-2])
            if axis.size > 2:
                weights[1:-1] = 0.5 * (axis[2:] - axis[:-2])
            return weights
        if method == "simpson":
            return np.asarray(simpson(np.eye(axis.size), x=axis, axis=1), dtype=float)
        raise ValueError("Integrate1D method must be 'trapezoid' or 'simpson'.")

    def calculate(self) -> dict[str, DataBundle]:
        cfg = self.configuration
        processing_keys = self._normalised_processing_keys()
        if not processing_keys:
            raise ValueError("Integrate1D requires at least one processing key.")
        signal_key = str(cfg.get("signal_key", "signal"))
        axis_key = str(cfg.get("axis_key", "q"))
        method = str(cfg.get("method", "trapezoid")).strip().lower()
        mask_key = cfg.get("mask_key")

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

        axis = axis[common]
        if axis.size < 2:
            raise ValueError("Integrate1D common domain contains fewer than two valid points.")
        differences = np.diff(axis)
        if not (np.all(differences > 0.0) or np.all(differences < 0.0)):
            raise ValueError("Integrate1D axis must be strictly monotonic.")
        quadrature_weights = self._quadrature_weights(axis, method)

        output_processing_keys = cfg.get("output_processing_keys")
        if output_processing_keys is not None:
            output_processing_keys = [str(key) for key in output_processing_keys]
            if len(output_processing_keys) != len(processing_keys):
                raise ValueError("output_processing_keys must contain one key per input bundle.")

        output: dict[str, DataBundle] = {}
        output_key = str(cfg.get("output_key", "integral"))
        for index, (processing_key, signal) in enumerate(zip(processing_keys, signals, strict=True)):
            bundle = self.processing_data[processing_key]
            values = np.asarray(signal.signal, dtype=float).squeeze()[common]
            uncertainties = {
                name: np.asarray(
                    np.sqrt(
                        np.sum(
                            (np.broadcast_to(component, signal.signal.shape).squeeze()[common] * quadrature_weights)
                            ** 2
                        )
                    )
                )
                for name, component in signal.uncertainties.items()
            }
            integral = BaseData(
                signal=np.asarray(np.sum(values * quadrature_weights)),
                units=signal.units * reference_axis.units,
                uncertainties=uncertainties,
                rank_of_data=0,
            )
            if output_processing_keys is None:
                bundle[output_key] = integral
                output[processing_key] = bundle
            else:
                result_key = output_processing_keys[index]
                bundle = DataBundle(signal=integral)
                bundle.default_plot = "signal"
                bundle.description = f"{method} integral of {processing_key}.{signal_key}"
                self.processing_data[result_key] = bundle
                output[result_key] = bundle
        return output
