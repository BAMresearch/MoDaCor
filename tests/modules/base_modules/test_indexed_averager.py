from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from modacor import ureg
from modacor.dataclasses.basedata import BaseData
from modacor.dataclasses.databundle import DataBundle
from modacor.dataclasses.process_step import ProcessStepDependencies
from modacor.dataclasses.processing_data import ProcessingData
from modacor.io.io_sources import IoSources
from modacor.modules import IndexedAverager


def _bundle(
    *,
    signal=(1.0, 2.0, 3.0, 4.0),
    q=(10.0, 20.0, 30.0, 40.0),
    indices=(0, 0, 1, 1),
    weights=1.0,
) -> DataBundle:
    return DataBundle(
        {
            "signal": BaseData(
                np.asarray(signal, dtype=float),
                ureg.count,
                weights=np.asarray(weights, dtype=float),
                rank_of_data=1,
            ),
            "Q": BaseData(np.asarray(q, dtype=float), ureg.Unit("1/nm"), rank_of_data=1),
            "bin_index": BaseData(np.asarray(indices), ureg.dimensionless, rank_of_data=1),
        }
    )


def _run(bundle: DataBundle, **configuration) -> tuple[ProcessingData, DataBundle]:
    processing_data = ProcessingData()
    processing_data["curve"] = bundle
    step = IndexedAverager(io_sources=IoSources())
    step.modify_config_by_dict({"with_processing_keys": ["curve"], **configuration})
    step.execute(processing_data)
    destination = configuration.get("output_processing_key") or "curve"
    return processing_data, processing_data[destination]


def test_indexed_averager_uses_indices_only_for_membership_and_measures_axis() -> None:
    bundle = _bundle(weights=[1.0, 3.0, 1.0, 1.0])

    _, result = _run(bundle, stats_keys=["signal", "Q"])

    assert_allclose(result["signal"].signal, [1.75, 3.5])
    assert_allclose(result["Q"].signal, [17.5, 35.0])
    assert result["signal"].axes == [result["Q"]]
    assert_array_equal(result["bin_id"].signal, [0, 1])
    assert_allclose(result["sum_weights"].signal, [4.0, 2.0])


def test_indexed_averager_applies_mask_after_indexing() -> None:
    bundle = _bundle()
    bundle["Mask"] = BaseData(np.array([False, True, False, False]), ureg.dimensionless, rank_of_data=1)

    _, result = _run(bundle)

    assert_allclose(result["signal"].signal, [1.0, 3.5])
    assert_allclose(result["Q"].signal, [10.0, 35.0])
    assert_allclose(result["bin_count"].signal, [1.0, 2.0])


def test_indexed_averager_omits_empty_and_zero_weight_bins_but_retains_ids() -> None:
    bundle = _bundle(
        signal=[1.0, 2.0, 3.0, 4.0],
        q=[10.0, 20.0, 30.0, 40.0],
        indices=[2, 2, 5, 7],
        weights=[1.0, 1.0, 2.0, 0.0],
    )

    _, result = _run(bundle)

    assert_array_equal(result["bin_id"].signal, [2, 5])
    assert_allclose(result["signal"].signal, [1.5, 3.0])
    assert_allclose(result["Q"].signal, [15.0, 30.0])


def test_indexed_averager_propagates_named_uncertainties_and_scatter() -> None:
    bundle = _bundle()
    bundle["signal"].uncertainties = {"counting": np.array([0.1, 0.1, 0.2, 0.2])}
    bundle["Q"].uncertainties = {"calibration": np.array([0.5, 0.5, 1.0, 1.0])}

    _, result = _run(bundle, stats_keys=["signal", "Q"])

    assert_allclose(result["signal"].uncertainties["counting"], [0.1 / np.sqrt(2), 0.2 / np.sqrt(2)])
    assert_allclose(result["Q"].uncertainties["calibration"], [0.5 / np.sqrt(2), 1.0 / np.sqrt(2)])
    assert_allclose(result["signal"].uncertainties["STD"], [0.5, 0.5])
    assert_allclose(result["signal"].uncertainties["SEM"], [np.sqrt(0.125), np.sqrt(0.125)])
    assert_allclose(result["Q"].uncertainties["STD"], [5.0, 5.0])


def test_indexed_averager_uses_user_selected_uncertainty_for_weights() -> None:
    bundle = _bundle()
    bundle["signal"].uncertainties = {"fit": np.array([1.0, 2.0, 1.0, 1.0])}

    _, result = _run(
        bundle,
        use_value_weights=False,
        use_value_uncertainty_weights=True,
        uncertainty_weight_key="fit",
    )

    assert_allclose(result["signal"].signal, [1.2, 3.5])
    assert_allclose(result["Q"].signal, [12.0, 35.0])


def test_indexed_averager_broadcasts_static_index_axis_and_mask() -> None:
    bundle = DataBundle(
        {
            "intensity": BaseData(
                np.array([[1.0, 2.0], [3.0, 4.0]]),
                ureg.count,
                rank_of_data=1,
            ),
            "angle": BaseData(np.array([10.0, 20.0]), ureg.degree, rank_of_data=1),
            "groups": BaseData(np.array([0, 1]), ureg.dimensionless, rank_of_data=1),
            "excluded": BaseData(np.array([False, True]), ureg.dimensionless, rank_of_data=1),
        }
    )

    _, result = _run(
        bundle,
        value_key="intensity",
        index_key="groups",
        axis_key="angle",
        mask_key="excluded",
        bin_id_key="group_id",
    )

    assert_array_equal(result["group_id"].signal, [0])
    assert_allclose(result["intensity"].signal, [2.0])
    assert_allclose(result["angle"].signal, [10.0])


def test_indexed_averager_can_reduce_without_a_measured_axis() -> None:
    bundle = _bundle()

    _, result = _run(bundle, axis_key=None)

    assert "Q" in result  # retained by in-place operation but not read or replaced
    assert result["signal"].axes == [result["bin_id"]]
    assert_allclose(result["Q"].signal, [10.0, 20.0, 30.0, 40.0])


def test_indexed_averager_distinct_output_contains_only_reduced_data() -> None:
    bundle = _bundle()
    source_signal = bundle["signal"]

    processing_data, result = _run(bundle, output_processing_key="averaged")

    assert processing_data["curve"]["signal"] is source_signal
    assert set(result) == {
        "signal",
        "Q",
        "bin_id",
        "bin_count",
        "positive_weight_count",
        "sum_weights",
        "effective_sample_size",
    }


def test_indexed_averager_dependency_contract_has_no_edge_dependency() -> None:
    step = IndexedAverager(io_sources=IoSources())
    step.modify_config_by_dict(
        {
            "with_processing_keys": ["sample"],
            "value_key": "intensity",
            "index_key": "groups",
            "axis_key": "angle",
            "mask_key": "excluded",
            "bin_id_key": "group_id",
        }
    )

    assert step.dependency_contract() == ProcessStepDependencies(
        processing_reads={"sample.intensity", "sample.groups", "sample.angle", "sample.excluded"},
        processing_writes={
            "sample.intensity",
            "sample.angle",
            "sample.group_id",
            "sample.bin_count",
            "sample.positive_weight_count",
            "sample.sum_weights",
            "sample.effective_sample_size",
        },
    )


def test_indexed_averager_rejects_non_integer_group_indices() -> None:
    bundle = _bundle(indices=[0.0, 0.5, 1.0, 1.0])

    with pytest.raises(ValueError, match="must be integers"):
        _run(bundle)
