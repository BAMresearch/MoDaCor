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
from modacor.modules import IndexByCoordinate


def _processing_data(values, *, units="1/nm") -> ProcessingData:
    data = ProcessingData()
    data["curve"] = DataBundle(
        {
            "position": BaseData(
                signal=np.asarray(values, dtype=float),
                units=ureg.Unit(units),
                rank_of_data=1,
            )
        }
    )
    return data


def test_index_by_coordinate_generates_edges_and_includes_final_right_edge() -> None:
    processing_data = _processing_data([0.0, 1.0, 2.0, 3.0, np.nan])
    step = IndexByCoordinate(io_sources=IoSources())
    step.modify_config_by_dict(
        {
            "with_processing_keys": ["curve"],
            "coordinate_key": "position",
            "bin_min": 0.0,
            "bin_max": 3.0,
            "n_bins": 3,
            "spacing": "linear",
        }
    )

    step.execute(processing_data)

    assert_array_equal(processing_data["curve"]["bin_index"].signal, [0, 1, 2, 2, -1])
    assert_allclose(processing_data["curve"]["bin_edges"].signal, [0.0, 1.0, 2.0, 3.0])
    assert processing_data["curve"]["bin_edges"].units == ureg.Unit("1/nm")


def test_index_by_coordinate_converts_configured_units() -> None:
    processing_data = _processing_data([0.0, 0.5, 1.0], units="1/angstrom")
    step = IndexByCoordinate(io_sources=IoSources())
    step.modify_config_by_dict(
        {
            "with_processing_keys": ["curve"],
            "coordinate_key": "position",
            "bin_edges": [0.0, 5.0, 10.0],
            "bin_units": "1/nm",
        }
    )

    step.execute(processing_data)

    assert_allclose(processing_data["curve"]["bin_edges"].signal, [0.0, 0.5, 1.0])
    assert_array_equal(processing_data["curve"]["bin_index"].signal, [0, 1, 1])


def test_index_by_coordinate_uses_coordinate_metadata_for_index_map() -> None:
    axis = BaseData(np.arange(3), ureg.dimensionless, rank_of_data=1)
    coordinate = BaseData(
        np.arange(3),
        ureg.degree,
        axes=[axis],
        rank_of_data=1,
    )
    processing_data = ProcessingData()
    processing_data["curve"] = DataBundle({"angle": coordinate})
    step = IndexByCoordinate(io_sources=IoSources())
    step.modify_config_by_dict(
        {
            "with_processing_keys": ["curve"],
            "coordinate_key": "angle",
            "bin_edges": [0.0, 1.0, 2.0],
        }
    )

    step.execute(processing_data)

    index = processing_data["curve"]["bin_index"]
    assert index.axes == [axis]
    assert index.rank_of_data == 1
    assert index.units == ureg.dimensionless


def test_index_by_coordinate_rejects_mixed_explicit_and_generated_edges() -> None:
    processing_data = _processing_data([0.0, 1.0])
    step = IndexByCoordinate(io_sources=IoSources())
    step.processing_data = processing_data
    step.modify_config_by_dict(
        {
            "with_processing_keys": ["curve"],
            "coordinate_key": "position",
            "bin_edges": [0.0, 1.0],
            "n_bins": 1,
        }
    )

    with pytest.raises(ValueError, match="mutually exclusive"):
        step.calculate()


def test_index_by_coordinate_dependency_contract_uses_configured_keys() -> None:
    step = IndexByCoordinate(io_sources=IoSources())
    step.modify_config_by_dict(
        {
            "with_processing_keys": ["sample", "background"],
            "coordinate_key": "angle",
            "index_key": "groups",
            "edges_key": "boundaries",
        }
    )

    assert step.dependency_contract() == ProcessStepDependencies(
        processing_reads={"sample.angle", "background.angle"},
        processing_writes={
            "sample.groups",
            "sample.boundaries",
            "background.groups",
            "background.boundaries",
        },
    )


def test_index_by_coordinate_does_not_read_or_apply_a_mask() -> None:
    processing_data = _processing_data([0.25, 0.75])
    processing_data["curve"]["Mask"] = BaseData(
        np.array([True, False]),
        ureg.dimensionless,
        rank_of_data=1,
    )
    step = IndexByCoordinate(io_sources=IoSources())
    step.modify_config_by_dict(
        {
            "with_processing_keys": ["curve"],
            "coordinate_key": "position",
            "bin_edges": [0.0, 0.5, 1.0],
        }
    )

    step.execute(processing_data)

    assert_array_equal(processing_data["curve"]["bin_index"].signal, [0, 1])
