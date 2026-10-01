from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from modacor import ureg
from modacor.dataclasses.basedata import BaseData
from modacor.dataclasses.databundle import DataBundle
from modacor.dataclasses.process_step import ProcessStepDependencies
from modacor.dataclasses.processing_data import ProcessingData
from modacor.io.io_sources import IoSources
from modacor.modules import ConcatenateDatabundles, IndexedAverager


def _bundle(signal, q, q_units, weights) -> DataBundle:
    q_bd = BaseData(
        np.asarray(q, dtype=float),
        q_units,
        uncertainties={"dq": np.full(len(q), 0.1)},
        rank_of_data=1,
    )
    signal_bd = BaseData(
        np.asarray(signal, dtype=float),
        ureg.count,
        uncertainties={"counting": np.full(len(q), 0.5)},
        weights=np.asarray(weights, dtype=float),
        axes=[q_bd],
        rank_of_data=1,
    )
    bundle = DataBundle(signal=signal_bd, Q=q_bd)
    bundle.default_plot = "signal"
    return bundle


def test_concatenate_preserves_order_units_metadata_and_source_indices():
    processing_data = ProcessingData()
    processing_data["a"] = _bundle([10, 20], [1, 2], ureg.Unit("1/nm"), [1, 2])
    processing_data["b"] = _bundle([30, 40], [0.3, 0.4], ureg.Unit("1/angstrom"), [3, 4])
    step = ConcatenateDatabundles(io_sources=IoSources())
    step.modify_config_by_dict(
        {
            "with_processing_keys": ["a", "b"],
            "data_keys": ["signal", "Q"],
            "output_processing_key": "pooled",
            "source_position_key": "source_position",
        }
    )

    step.execute(processing_data)
    output = processing_data["pooled"]

    assert_allclose(output["signal"].signal, [10, 20, 30, 40])
    assert_allclose(output["signal"].weights, [1, 2, 3, 4])
    assert_allclose(output["signal"].uncertainties["counting"], [0.5] * 4)
    assert_allclose(output["Q"].signal, [1, 2, 3, 4])
    assert_allclose(output["Q"].uncertainties["dq"], [0.1, 0.1, 1.0, 1.0])
    assert_allclose(output["source_index"].signal, [0, 0, 1, 1])
    assert_allclose(output["source_position"].signal, [0, 1, 0, 1])
    assert output["signal"].axes == [output["Q"]]
    assert output.default_plot == "signal"


def test_concatenate_sort_by_reorders_every_entry_stably():
    processing_data = ProcessingData()
    processing_data["a"] = _bundle([30, 10], [3, 1], ureg.Unit("1/nm"), [3, 1])
    processing_data["b"] = _bundle([20, 11], [2, 1], ureg.Unit("1/nm"), [2, 1.1])
    step = ConcatenateDatabundles(io_sources=IoSources())
    step.processing_data = processing_data
    step.modify_config_by_dict(
        {
            "with_processing_keys": ["a", "b"],
            "data_keys": ["signal", "Q"],
            "output_processing_key": "pooled",
            "sort_by": "Q",
        }
    )

    output = step.calculate()["pooled"]

    assert_allclose(output["Q"].signal, [1, 1, 2, 3])
    assert_allclose(output["signal"].signal, [10, 11, 20, 30])
    assert_allclose(output["signal"].weights, [1, 1.1, 2, 3])
    assert_allclose(output["source_index"].signal, [0, 1, 1, 0])


def test_concatenate_rejects_mismatched_uncertainty_components():
    processing_data = ProcessingData()
    processing_data["a"] = _bundle([1], [1], ureg.Unit("1/nm"), [1])
    processing_data["b"] = _bundle([2], [2], ureg.Unit("1/nm"), [1])
    processing_data["b"]["signal"].uncertainties = {}
    step = ConcatenateDatabundles(io_sources=IoSources())
    step.processing_data = processing_data
    step.modify_config_by_dict({"with_processing_keys": ["a", "b"], "output_processing_key": "pooled"})

    with pytest.raises(ValueError, match="matching uncertainty keys"):
        step.calculate()


def test_concatenate_rejects_conflicting_generated_keys():
    processing_data = ProcessingData()
    processing_data["a"] = _bundle([1], [1], ureg.Unit("1/nm"), [1])
    step = ConcatenateDatabundles(io_sources=IoSources())
    step.processing_data = processing_data
    step.modify_config_by_dict(
        {
            "with_processing_keys": ["a"],
            "output_processing_key": "pooled",
            "source_index_key": "origin",
            "source_position_key": "origin",
        }
    )

    with pytest.raises(ValueError, match="provenance keys must be distinct"):
        step.calculate()


def test_source_position_can_group_aligned_inputs_for_indexed_averaging():
    processing_data = ProcessingData()
    processing_data["front"] = _bundle([10, 20], [1, 2], ureg.Unit("1/nm"), [1, 1])
    processing_data["rear"] = _bundle([14, 18], [1, 2], ureg.Unit("1/nm"), [1, 1])

    concatenate = ConcatenateDatabundles(io_sources=IoSources())
    concatenate.modify_config_by_dict(
        {
            "with_processing_keys": ["front", "rear"],
            "data_keys": ["signal", "Q"],
            "output_processing_key": "pooled",
            "source_position_key": "point_index",
            "alignment_key": "Q",
        }
    )
    concatenate.execute(processing_data)

    average = IndexedAverager(io_sources=IoSources())
    average.modify_config_by_dict(
        {
            "with_processing_keys": ["pooled"],
            "output_processing_key": "merged",
            "value_key": "signal",
            "index_key": "point_index",
            "axis_key": "Q",
            "mask_key": None,
            "use_value_weights": True,
            "use_value_uncertainty_weights": True,
            "uncertainty_weight_key": "counting",
            "stats_keys": [],
        }
    )
    average.execute(processing_data)

    assert_allclose(processing_data["merged"]["signal"].signal, [12, 19])
    assert_allclose(processing_data["merged"]["Q"].signal, [1, 2])
    assert_allclose(processing_data["merged"]["bin_count"].signal, [2, 2])


def test_concatenate_alignment_check_rejects_mismatched_coordinates():
    processing_data = ProcessingData()
    processing_data["front"] = _bundle([10, 20], [1, 2], ureg.Unit("1/nm"), [1, 1])
    processing_data["rear"] = _bundle([14, 18], [1, 2.1], ureg.Unit("1/nm"), [1, 1])
    step = ConcatenateDatabundles(io_sources=IoSources())
    step.processing_data = processing_data
    step.modify_config_by_dict(
        {
            "with_processing_keys": ["front", "rear"],
            "data_keys": ["signal", "Q"],
            "output_processing_key": "pooled",
            "source_position_key": "point_index",
            "alignment_key": "Q",
        }
    )

    with pytest.raises(ValueError, match="does not match pointwise"):
        step.calculate()


def test_concatenate_dependency_contract_is_exact():
    step = ConcatenateDatabundles(io_sources=IoSources())
    step.modify_config_by_dict(
        {
            "with_processing_keys": ["front", "rear"],
            "data_keys": ["signal", "Q", "Mask"],
            "output_processing_key": "pooled",
        }
    )

    assert step.dependency_contract() == ProcessStepDependencies(
        processing_reads={
            "front.signal",
            "front.Q",
            "front.Mask",
            "rear.signal",
            "rear.Q",
            "rear.Mask",
        },
        processing_writes={"pooled.*"},
    )
