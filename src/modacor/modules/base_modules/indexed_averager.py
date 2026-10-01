# SPDX-License-Identifier: BSD-3-Clause
"""Weighted averaging of values grouped by a precomputed index map."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from attrs import define

from modacor import ureg
from modacor.dataclasses.basedata import BaseData
from modacor.dataclasses.databundle import DataBundle
from modacor.dataclasses.process_step import ProcessStep, ProcessStepDependencies, normalize_processing_key_values
from modacor.dataclasses.process_step_describer import ProcessStepDescriber
from modacor.modules.helpers import finalize_weighted_scatter, normalize_str_list

__all__ = ["IndexedAverager"]
__version__ = "20261001.1"

_DIAGNOSTIC_KEYS = (
    "bin_count",
    "positive_weight_count",
    "sum_weights",
    "effective_sample_size",
)


@define(frozen=True, slots=True)
class _ReductionInputs:
    value_key: str
    axis_key: str | None
    bin_id_key: str
    value: BaseData
    axis: BaseData | None
    shape: tuple[int, ...]
    values: np.ndarray
    axis_values: np.ndarray | None
    indices: np.ndarray
    valid: np.ndarray


@define(frozen=True, slots=True)
class _GroupedObservations:
    bin_ids: np.ndarray
    selection: np.ndarray
    dense_indices: np.ndarray
    weights: np.ndarray
    values: np.ndarray
    axis_values: np.ndarray | None
    sum_weights: np.ndarray
    sum_squared_weights: np.ndarray


def _broadcast(array: np.ndarray, shape: tuple[int, ...], description: str) -> np.ndarray:
    try:
        return np.broadcast_to(np.asarray(array), shape).ravel()
    except ValueError as exc:
        raise ValueError(f"IndexedAverager: {description} cannot broadcast to value shape {shape}.") from exc


def _propagate_uncertainties(
    data: BaseData,
    *,
    shape: tuple[int, ...],
    selection: np.ndarray,
    dense_indices: np.ndarray,
    weights: np.ndarray,
    sum_weights: np.ndarray,
) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for name, uncertainty in data.uncertainties.items():
        selected = _broadcast(uncertainty, shape, f"uncertainty {name!r}")[selection]
        variance_sum = np.bincount(
            dense_indices,
            weights=(weights**2) * (selected**2),
            minlength=sum_weights.size,
        )
        result[name] = np.sqrt(variance_sum) / sum_weights
    return result


def _scatter_statistics(
    values: np.ndarray,
    means: np.ndarray,
    dense_indices: np.ndarray,
    weights: np.ndarray,
    sum_weights: np.ndarray,
    sum_squared_weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    deviations = values - means[dense_indices]
    squared_deviation_sum = np.bincount(
        dense_indices,
        weights=weights * deviations**2,
        minlength=sum_weights.size,
    )
    estimates = finalize_weighted_scatter(
        sum_w=sum_weights,
        sum_w2=sum_squared_weights,
        sum_w_squared_deviations=squared_deviation_sum,
        ddof=0,
    )
    enough_observations = estimates.effective_sample_size > 1.0
    sem = np.where(enough_observations, estimates.standard_error_mean, np.nan)
    std = np.where(enough_observations, estimates.standard_deviation, np.nan)
    return sem, std


class IndexedAverager(ProcessStep):
    """Average one value and an optional measured axis by precomputed indices.

    Bin membership comes exclusively from ``index_key``. ``axis_key`` is an
    optional co-reduced value used as the output axis; it never changes a
    point's bin. Only bins with positive total weight are emitted.
    """

    documentation = ProcessStepDescriber(
        calling_name="Indexed Averager",
        calling_id="IndexedAverager",
        calling_module_path=Path(__file__),
        calling_version=__version__,
        required_data_keys=[],
        arguments={
            "with_processing_keys": {
                "type": (str, list, type(None)),
                "default": None,
                "doc": "ProcessingData key(s) to reduce.",
            },
            "output_processing_key": {
                "type": (str, type(None)),
                "default": None,
                "doc": "Optional distinct output bundle; requires exactly one input bundle.",
            },
            "value_key": {
                "type": str,
                "default": "signal",
                "doc": "BaseData value to average.",
            },
            "index_key": {
                "type": str,
                "default": "bin_index",
                "doc": "BaseData containing precomputed integer group indices.",
            },
            "axis_key": {
                "type": (str, type(None)),
                "default": "Q",
                "doc": "Optional BaseData to average over the same points and attach as the output axis.",
            },
            "mask_key": {
                "type": (str, type(None)),
                "default": "Mask",
                "doc": "Optional mask BaseData; true values are excluded when the key is present.",
            },
            "bin_id_key": {
                "type": str,
                "default": "bin_id",
                "doc": "Output BaseData key recording the original populated bin IDs.",
            },
            "use_value_weights": {
                "type": bool,
                "default": True,
                "doc": "Use the value BaseData weights for both the value and output axis.",
            },
            "use_value_uncertainty_weights": {
                "type": bool,
                "default": False,
                "doc": "Also weight by inverse variance from one named value uncertainty.",
            },
            "uncertainty_weight_key": {
                "type": (str, type(None)),
                "default": None,
                "doc": "Value uncertainty component used for inverse-variance weighting.",
            },
            "stats_keys": {
                "type": (list, str, type(None)),
                "default": None,
                "doc": "Value and/or axis keys receiving scatter-derived SEM and STD; None selects both.",
            },
        },
        modifies={
            "configured value": ["signal", "uncertainties", "axes"],
            "configured axis": ["signal", "uncertainties"],
            "bin_id": ["signal"],
            "bin_count": ["signal"],
            "positive_weight_count": ["signal"],
            "sum_weights": ["signal"],
            "effective_sample_size": ["signal"],
        },
        step_keywords=["indexed", "grouped", "weighted average", "reduction"],
        step_doc="Compute weighted group means from a precomputed integer index map.",
        step_note=(
            "The index map alone determines membership. The optional axis is averaged over the same valid, "
            "unmasked observations and weights; bin edges are neither required nor read."
        ),
    )

    def _configuration_keys(self) -> tuple[str, str, str | None, str | None, str]:
        values: dict[str, str | None] = {}
        for name in ("value_key", "index_key", "axis_key", "mask_key", "bin_id_key"):
            configured = self.configuration.get(name)
            if configured is None and name in {"axis_key", "mask_key"}:
                values[name] = None
                continue
            normalized = str(configured).strip()
            if not normalized:
                raise ValueError(f"IndexedAverager: {name} must be a non-empty string or None.")
            values[name] = normalized

        required_distinct = [values["value_key"], values["index_key"], values["bin_id_key"]]
        if values["axis_key"] is not None:
            required_distinct.append(values["axis_key"])
        if len(required_distinct) != len(set(required_distinct)):
            raise ValueError("IndexedAverager: value, index, axis, and bin-ID keys must be distinct.")
        return (
            values["value_key"],
            values["index_key"],
            values["axis_key"],
            values["mask_key"],
            values["bin_id_key"],
        )

    def _output_key(self, processing_keys: list[str]) -> str | None:
        configured = self.configuration.get("output_processing_key")
        if configured is None:
            return None
        output_key = str(configured).strip()
        if not output_key:
            return None
        if len(processing_keys) != 1:
            raise ValueError("IndexedAverager: output_processing_key requires exactly one input bundle.")
        return output_key

    def prepare_execution(self) -> None:
        processing_keys = normalize_processing_key_values(self.configuration.get("with_processing_keys"))
        if not processing_keys and self.processing_data is not None:
            processing_keys = self._normalised_processing_keys()
        self._configuration_keys()
        self._output_key(processing_keys)
        if self.configuration.get("use_value_uncertainty_weights") and not self.configuration.get(
            "uncertainty_weight_key"
        ):
            raise ValueError(
                "IndexedAverager: uncertainty_weight_key is required when " "use_value_uncertainty_weights is true."
            )

    def dependency_contract(self) -> ProcessStepDependencies:
        processing_keys = normalize_processing_key_values(self.configuration.get("with_processing_keys"))
        if not processing_keys:
            return ProcessStepDependencies(processing_reads={"*"}, processing_writes={"*"})

        value_key, index_key, axis_key, mask_key, bin_id_key = self._configuration_keys()
        input_keys = {value_key, index_key}
        if axis_key is not None:
            input_keys.add(axis_key)
        if mask_key is not None:
            input_keys.add(mask_key)
        reads = {
            f"{processing_key}.{basedata_key}" for processing_key in processing_keys for basedata_key in input_keys
        }
        output_key = self._output_key(processing_keys)
        if output_key is not None and output_key not in processing_keys:
            writes = {f"{output_key}.*"}
        else:
            output_keys = {value_key, bin_id_key, *_DIAGNOSTIC_KEYS}
            if axis_key is not None:
                output_keys.add(axis_key)
            writes = {
                f"{processing_key}.{basedata_key}" for processing_key in processing_keys for basedata_key in output_keys
            }
        return ProcessStepDependencies(processing_reads=reads, processing_writes=writes)

    def _prepare_inputs(self, bundle: DataBundle) -> _ReductionInputs:
        value_key, index_key, axis_key, mask_key, bin_id_key = self._configuration_keys()
        missing = [key for key in (value_key, index_key, axis_key) if key is not None and key not in bundle]
        if missing:
            raise KeyError(f"IndexedAverager: DataBundle is missing required keys {missing!r}.")

        value = bundle[value_key]
        axis = None if axis_key is None else bundle[axis_key]
        shape = value.shape
        value_values = np.asarray(value.signal, dtype=float).ravel()
        raw_indices = _broadcast(bundle[index_key].signal, shape, f"index {index_key!r}").astype(float)

        finite_indices = np.isfinite(raw_indices)
        nonnegative_indices = finite_indices & (raw_indices >= 0.0)
        if np.any(nonnegative_indices & (raw_indices != np.floor(raw_indices))):
            raise ValueError("IndexedAverager: non-negative index values must be integers.")
        indices = np.full(raw_indices.shape, -1, dtype=np.int64)
        indices[finite_indices] = raw_indices[finite_indices].astype(np.int64)

        valid = nonnegative_indices & np.isfinite(value_values)
        axis_values = None
        if axis is not None:
            axis_values = _broadcast(axis.signal, shape, f"axis {axis_key!r}").astype(float)
            valid &= np.isfinite(axis_values)

        if mask_key is not None and mask_key in bundle:
            mask = _broadcast(bundle[mask_key].signal, shape, f"mask {mask_key!r}").astype(bool)
            valid &= ~mask

        return _ReductionInputs(
            value_key=value_key,
            axis_key=axis_key,
            bin_id_key=bin_id_key,
            value=value,
            axis=axis,
            shape=shape,
            values=value_values,
            axis_values=axis_values,
            indices=indices,
            valid=valid,
        )

    def _resolve_weights(self, inputs: _ReductionInputs) -> tuple[np.ndarray, np.ndarray]:
        valid = inputs.valid.copy()
        value = inputs.value
        shape = inputs.shape
        value_key = inputs.value_key

        weights = np.ones(inputs.values.shape, dtype=float)
        if self.configuration.get("use_value_weights", True):
            weights *= _broadcast(value.weights, shape, f"weights for {value_key!r}").astype(float)

        if self.configuration.get("use_value_uncertainty_weights", False):
            uncertainty_key = self.configuration.get("uncertainty_weight_key")
            if not uncertainty_key:
                raise ValueError("IndexedAverager: uncertainty_weight_key is required for uncertainty weighting.")
            if uncertainty_key not in value.uncertainties:
                raise KeyError(f"IndexedAverager: uncertainty {uncertainty_key!r} is not present on {value_key!r}.")
            sigma = _broadcast(
                value.uncertainties[uncertainty_key],
                shape,
                f"uncertainty {uncertainty_key!r}",
            ).astype(float)
            usable_sigma = np.isfinite(sigma) & (sigma > 0.0)
            valid &= usable_sigma
            weights[usable_sigma] *= 1.0 / sigma[usable_sigma] ** 2

        if np.any(valid & np.isfinite(weights) & (weights < 0.0)):
            raise ValueError("IndexedAverager: weights must not be negative.")
        valid &= np.isfinite(weights)
        return weights, valid

    @staticmethod
    def _group_observations(
        inputs: _ReductionInputs,
        weights: np.ndarray,
        valid: np.ndarray,
    ) -> _GroupedObservations:
        contributing = valid & (weights > 0.0)
        if not np.any(contributing):
            raise ValueError("IndexedAverager: no valid observations with positive weight remain.")

        populated_bin_ids = np.unique(inputs.indices[contributing])
        selection = valid & np.isin(inputs.indices, populated_bin_ids)
        selected_indices = inputs.indices[selection]
        dense_indices = np.searchsorted(populated_bin_ids, selected_indices)
        selected_weights = weights[selection]
        selected_values = inputs.values[selection]
        n_bins = populated_bin_ids.size

        sum_weights = np.bincount(dense_indices, weights=selected_weights, minlength=n_bins)
        sum_squared_weights = np.bincount(dense_indices, weights=selected_weights**2, minlength=n_bins)
        return _GroupedObservations(
            bin_ids=populated_bin_ids,
            selection=selection,
            dense_indices=dense_indices,
            weights=selected_weights,
            values=selected_values,
            axis_values=None if inputs.axis_values is None else inputs.axis_values[selection],
            sum_weights=sum_weights,
            sum_squared_weights=sum_squared_weights,
        )

    def _stats_keys(self, value_key: str, axis_key: str | None) -> set[str]:
        available = {value_key}
        if axis_key is not None:
            available.add(axis_key)
        configured = normalize_str_list(self.configuration.get("stats_keys"))
        selected = available if configured is None else set(configured)
        unknown = selected - available
        if unknown:
            raise ValueError(
                "IndexedAverager: stats_keys contains keys that are not the value or axis: " f"{sorted(unknown)!r}."
            )
        return selected

    def _reduce_bundle(self, bundle: DataBundle) -> DataBundle:
        inputs = self._prepare_inputs(bundle)
        weights, valid = self._resolve_weights(inputs)
        grouped = self._group_observations(inputs, weights, valid)
        n_bins = grouped.bin_ids.size
        mean_values = (
            np.bincount(
                grouped.dense_indices,
                weights=grouped.weights * grouped.values,
                minlength=n_bins,
            )
            / grouped.sum_weights
        )

        mean_axis = None
        if grouped.axis_values is not None:
            mean_axis = (
                np.bincount(
                    grouped.dense_indices,
                    weights=grouped.weights * grouped.axis_values,
                    minlength=n_bins,
                )
                / grouped.sum_weights
            )

        value_uncertainties = _propagate_uncertainties(
            inputs.value,
            shape=inputs.shape,
            selection=grouped.selection,
            dense_indices=grouped.dense_indices,
            weights=grouped.weights,
            sum_weights=grouped.sum_weights,
        )
        axis_uncertainties = (
            {}
            if inputs.axis is None
            else _propagate_uncertainties(
                inputs.axis,
                shape=inputs.shape,
                selection=grouped.selection,
                dense_indices=grouped.dense_indices,
                weights=grouped.weights,
                sum_weights=grouped.sum_weights,
            )
        )

        stats_keys = self._stats_keys(inputs.value_key, inputs.axis_key)
        if inputs.value_key in stats_keys:
            value_uncertainties["SEM"], value_uncertainties["STD"] = _scatter_statistics(
                grouped.values,
                mean_values,
                grouped.dense_indices,
                grouped.weights,
                grouped.sum_weights,
                grouped.sum_squared_weights,
            )
        if inputs.axis_key is not None and inputs.axis_key in stats_keys:
            axis_uncertainties["SEM"], axis_uncertainties["STD"] = _scatter_statistics(
                grouped.axis_values,
                mean_axis,
                grouped.dense_indices,
                grouped.weights,
                grouped.sum_weights,
                grouped.sum_squared_weights,
            )

        bin_ids = BaseData(grouped.bin_ids, ureg.dimensionless, rank_of_data=1)
        axis_output = None
        if inputs.axis is not None:
            axis_output = BaseData(
                signal=mean_axis,
                units=inputs.axis.units,
                uncertainties=axis_uncertainties,
                weights=np.ones(n_bins),
                axes=[],
                rank_of_data=1,
            )
        output_axis = axis_output if axis_output is not None else bin_ids
        value_output = BaseData(
            signal=mean_values,
            units=inputs.value.units,
            uncertainties=value_uncertainties,
            weights=np.ones(n_bins),
            axes=[output_axis],
            rank_of_data=1,
        )

        bin_count = np.bincount(grouped.dense_indices, minlength=n_bins).astype(float)
        positive_weight_count = np.bincount(
            grouped.dense_indices[grouped.weights > 0.0],
            minlength=n_bins,
        ).astype(float)
        effective_sample_size = grouped.sum_weights**2 / grouped.sum_squared_weights
        diagnostics = {
            "bin_count": bin_count,
            "positive_weight_count": positive_weight_count,
            "sum_weights": grouped.sum_weights,
            "effective_sample_size": effective_sample_size,
        }

        result = DataBundle({inputs.value_key: value_output, inputs.bin_id_key: bin_ids})
        if inputs.axis_key is not None:
            result[inputs.axis_key] = axis_output
        for name, values in diagnostics.items():
            result[name] = BaseData(
                signal=values,
                units=ureg.dimensionless,
                axes=[output_axis],
                rank_of_data=1,
            )
        return result

    def calculate(self) -> dict[str, DataBundle]:
        processing_keys = self._normalised_processing_keys()
        output_processing_key = self._output_key(processing_keys)
        output: dict[str, DataBundle] = {}
        for processing_key in processing_keys:
            if processing_key not in self.processing_data:
                raise KeyError(f"IndexedAverager: ProcessingData key {processing_key!r} not found.")
            reduced = self._reduce_bundle(self.processing_data[processing_key])
            destination = output_processing_key or processing_key
            if destination == processing_key:
                self.processing_data[processing_key].update(reduced)
                result = self.processing_data[processing_key]
            else:
                self.processing_data[destination] = reduced
                result = reduced
            output[destination] = result
        return output
