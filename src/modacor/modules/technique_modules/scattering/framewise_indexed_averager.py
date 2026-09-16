# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

__all__ = ["FramewiseIndexedAverager"]
__version__ = "20260916.1"

from pathlib import Path
from typing import Any

import numpy as np

from modacor.dataclasses.basedata import BaseData
from modacor.dataclasses.databundle import DataBundle
from modacor.dataclasses.process_step import ProcessStep
from modacor.dataclasses.process_step_describer import ProcessStepDescriber
from modacor.modules.helpers import get_first_present, normalize_str_list
from modacor.modules.technique_modules.scattering.indexed_averager import IndexedAverager


class FramewiseIndexedAverager(ProcessStep):
    """Apply indexed detector integration independently to every leading frame.

    ``rank_of_data`` identifies the trailing detector dimensions. Any leading
    dimensions are preserved as batch dimensions, while the detector dimensions
    are replaced by one integration-bin dimension. Static geometry, pixel-index,
    and mask arrays may contain only the detector dimensions; they are reused for
    every frame without being copied to the full stack.
    """

    documentation = ProcessStepDescriber(
        calling_name="Frame-wise indexed averager",
        calling_id="FramewiseIndexedAverager",
        calling_module_path=Path(__file__),
        calling_version=__version__,
        required_data_keys=["signal", "Q", "Psi", "pixel_index"],
        modifies={
            "signal": ["signal", "uncertainties", "axes", "rank_of_data"],
            "Q": ["signal", "uncertainties", "rank_of_data"],
            "Psi": ["signal", "uncertainties", "rank_of_data"],
        },
        arguments={
            "with_processing_keys": {
                "type": (str, list, type(None)),
                "required": True,
                "default": None,
                "doc": "ProcessingData key or keys containing frame stacks.",
            },
            "averaging_direction": {
                "type": str,
                "required": True,
                "default": "azimuthal",
                "doc": "Averaging direction: 'radial' or 'azimuthal'.",
            },
            "use_signal_weights": {
                "type": bool,
                "default": True,
                "doc": "Use BaseData weights when integrating each frame.",
            },
            "use_signal_uncertainty_weights": {
                "type": bool,
                "default": False,
                "doc": "Use one signal uncertainty as an additional weight.",
            },
            "uncertainty_weight_key": {
                "type": (str, type(None)),
                "default": None,
                "doc": "Signal uncertainty name used when uncertainty weighting is enabled.",
            },
            "stats_keys": {
                "type": (list, str, type(None)),
                "default": None,
                "doc": "Outputs that receive per-bin SEM and STD statistics.",
            },
        },
        step_keywords=["frame", "batch", "azimuthal", "averaging", "binning"],
        step_doc="Azimuthally or radially integrate each detector frame without reducing the frame axes.",
        step_reference="DOI 10.1088/0953-8984/25/38/383201",
        step_note=(
            "Use this for frame-quality checks and SEC-SAXS data. The ordinary "
            "IndexedAverager remains the single-image reducer."
        ),
    )

    @staticmethod
    def _broadcast_frame(
        values: Any,
        *,
        full_shape: tuple[int, ...],
        batch_index: tuple[int, ...],
        label: str,
    ) -> np.ndarray:
        array = np.asarray(values)
        try:
            broadcast = np.broadcast_to(array, full_shape)
        except ValueError as exc:
            raise ValueError(
                f"FramewiseIndexedAverager: {label} shape {array.shape} cannot broadcast "
                f"to signal shape {full_shape}."
            ) from exc
        return np.asarray(broadcast[batch_index] if batch_index else broadcast)

    @classmethod
    def _frame_basedata(
        cls,
        source: BaseData,
        *,
        full_shape: tuple[int, ...],
        batch_index: tuple[int, ...],
        rank_of_data: int,
        label: str,
    ) -> BaseData:
        signal = cls._broadcast_frame(
            source.signal,
            full_shape=full_shape,
            batch_index=batch_index,
            label=f"{label}.signal",
        )
        uncertainties = {
            name: cls._broadcast_frame(
                values,
                full_shape=full_shape,
                batch_index=batch_index,
                label=f"{label}.uncertainties[{name!r}]",
            )
            for name, values in source.uncertainties.items()
        }
        weights_array = np.asarray(source.weights)
        if weights_array.size == 1:
            weights = np.array(weights_array, copy=True)
        else:
            weights = cls._broadcast_frame(
                weights_array,
                full_shape=full_shape,
                batch_index=batch_index,
                label=f"{label}.weights",
            )
        axes = list(source.axes[-rank_of_data:]) if source.axes else []
        return BaseData(
            signal=signal,
            units=source.units,
            uncertainties=uncertainties,
            weights=weights,
            axes=axes,
            rank_of_data=rank_of_data,
        )

    @staticmethod
    def _pad(array: np.ndarray, size: int) -> np.ndarray:
        result = np.full(size, np.nan, dtype=float)
        result[: array.size] = array
        return result

    @classmethod
    def _pad_basedata(cls, data: BaseData, size: int) -> BaseData:
        return BaseData(
            signal=cls._pad(np.asarray(data.signal), size),
            units=data.units,
            uncertainties={name: cls._pad(np.asarray(values), size) for name, values in data.uncertainties.items()},
            weights=np.ones(size, dtype=float),
            rank_of_data=1,
        )

    def _average_bundle(self, bundle: DataBundle) -> DataBundle:
        try:
            signal_source = bundle["signal"]
            q_source = bundle["Q"]
            psi_source = bundle["Psi"]
            pixel_index_source = bundle["pixel_index"]
        except KeyError as exc:
            raise KeyError("FramewiseIndexedAverager requires signal, Q, Psi, and pixel_index.") from exc

        full_shape = tuple(signal_source.shape)
        spatial_rank = int(signal_source.rank_of_data)
        if spatial_rank < 1 or spatial_rank > len(full_shape):
            raise ValueError(
                "FramewiseIndexedAverager requires rank_of_data to identify at least one trailing detector axis."
            )
        batch_shape = full_shape[:-spatial_rank]
        spatial_shape = full_shape[-spatial_rank:]

        pixel_values = np.asarray(pixel_index_source.signal, dtype=float)
        try:
            pixel_values = np.broadcast_to(pixel_values, spatial_shape)
        except ValueError as exc:
            raise ValueError(
                f"FramewiseIndexedAverager: pixel_index shape {pixel_values.shape} does not match "
                f"detector shape {spatial_shape}."
            ) from exc
        finite_indices = pixel_values[np.isfinite(pixel_values) & (pixel_values >= 0)]
        if finite_indices.size == 0:
            raise ValueError("FramewiseIndexedAverager: pixel_index contains no active bins.")
        n_bins = int(np.max(finite_indices)) + 1

        mask_source = get_first_present(bundle, "Mask", "mask")
        frame_indices = list(np.ndindex(batch_shape)) if batch_shape else [()]
        signal_frames: list[BaseData] = []
        q_frames: list[BaseData] = []
        psi_frames: list[BaseData] = []

        use_signal_weights = bool(self.configuration.get("use_signal_weights", True))
        use_uncertainty_weights = bool(self.configuration.get("use_signal_uncertainty_weights", False))
        uncertainty_weight_key = self.configuration.get("uncertainty_weight_key")
        stats_keys = normalize_str_list(self.configuration.get("stats_keys"))

        for batch_index in frame_indices:
            signal = self._frame_basedata(
                signal_source,
                full_shape=full_shape,
                batch_index=batch_index,
                rank_of_data=spatial_rank,
                label="signal",
            )
            q_data = self._frame_basedata(
                q_source,
                full_shape=full_shape,
                batch_index=batch_index,
                rank_of_data=spatial_rank,
                label="Q",
            )
            psi_data = self._frame_basedata(
                psi_source,
                full_shape=full_shape,
                batch_index=batch_index,
                rank_of_data=spatial_rank,
                label="Psi",
            )
            pixel_index = self._frame_basedata(
                pixel_index_source,
                full_shape=full_shape,
                batch_index=batch_index,
                rank_of_data=spatial_rank,
                label="pixel_index",
            )
            mask = (
                self._frame_basedata(
                    mask_source,
                    full_shape=full_shape,
                    batch_index=batch_index,
                    rank_of_data=spatial_rank,
                    label="mask",
                )
                if mask_source is not None
                else None
            )

            try:
                signal_1d, q_1d, psi_1d = IndexedAverager._compute_bin_averages(
                    signal_bd=signal,
                    q_bd=q_data,
                    psi_bd=psi_data,
                    pix_bd=pixel_index,
                    mask_bd=mask,
                    use_signal_weights=use_signal_weights,
                    use_signal_uncertainty_weights=use_uncertainty_weights,
                    uncertainty_weight_key=uncertainty_weight_key,
                    stats_keys=stats_keys,
                )
            except ValueError as exc:
                if "no valid pixels" not in str(exc):
                    raise
                signal_1d = BaseData(signal=np.full(n_bins, np.nan), units=signal.units, rank_of_data=1)
                q_1d = BaseData(signal=np.full(n_bins, np.nan), units=q_data.units, rank_of_data=1)
                psi_1d = BaseData(signal=np.full(n_bins, np.nan), units=psi_data.units, rank_of_data=1)

            signal_1d = self._pad_basedata(signal_1d, n_bins)
            q_1d = self._pad_basedata(q_1d, n_bins)
            psi_1d = self._pad_basedata(psi_1d, n_bins)
            signal_frames.append(signal_1d)
            q_frames.append(q_1d)
            psi_frames.append(psi_1d)

        output_shape = (*batch_shape, n_bins)

        def stacked(values: list[np.ndarray]) -> np.ndarray:
            return np.stack(values).reshape(output_shape)

        def stacked_uncertainties(frames: list[BaseData]) -> dict[str, np.ndarray]:
            names = set().union(*(frame.uncertainties for frame in frames))
            return {
                name: stacked([np.asarray(frame.uncertainties.get(name, np.full(n_bins, np.nan))) for frame in frames])
                for name in names
            }

        signal_out = BaseData(
            signal=stacked([frame.signal for frame in signal_frames]),
            units=signal_source.units,
            uncertainties=stacked_uncertainties(signal_frames),
            weights=np.ones(output_shape, dtype=float),
            rank_of_data=1,
        )
        q_out = BaseData(
            signal=stacked([frame.signal for frame in q_frames]),
            units=q_source.units,
            uncertainties=stacked_uncertainties(q_frames),
            weights=np.ones(output_shape, dtype=float),
            rank_of_data=1,
        )
        psi_out = BaseData(
            signal=stacked([frame.signal for frame in psi_frames]),
            units=psi_source.units,
            uncertainties=stacked_uncertainties(psi_frames),
            weights=np.ones(output_shape, dtype=float),
            rank_of_data=1,
        )
        direction = str(self.configuration.get("averaging_direction", "azimuthal")).lower()
        if direction not in {"azimuthal", "radial"}:
            raise ValueError("FramewiseIndexedAverager averaging_direction must be 'azimuthal' or 'radial'.")
        # A frame-wise mask can change the mean Q/Psi coordinate within a bin,
        # so no single static axis is guaranteed to describe every frame.
        # Keep the complete coordinates in the separate frame-wise Q/Psi
        # outputs and explicitly leave the signal data axis unspecified.
        signal_out.axes = [None]
        return DataBundle(signal=signal_out, Q=q_out, Psi=psi_out)

    def calculate(self) -> dict[str, DataBundle]:
        output: dict[str, DataBundle] = {}
        for key in self._normalised_processing_keys():
            if key not in self.processing_data:
                raise KeyError(f"FramewiseIndexedAverager DataBundle not found: {key!r}")
            averaged = self._average_bundle(self.processing_data[key])
            self.processing_data[key] = averaged
            output[key] = averaged
        return output
