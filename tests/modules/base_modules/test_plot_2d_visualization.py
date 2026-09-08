# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import numpy as np
import pytest

from modacor import ureg
from modacor.dataclasses.basedata import BaseData
from modacor.dataclasses.databundle import DataBundle
from modacor.dataclasses.processing_data import ProcessingData
from modacor.io.buffer.runtime_buffer_store import RuntimeBufferStore
from modacor.io.io_sinks import IoSinks
from modacor.io.visualization import PlotlyJSONSink
from modacor.modules.base_modules.plot_2d_visualization import Plot2DVisualization


def _sinks(store: RuntimeBufferStore) -> IoSinks:
    sinks = IoSinks()
    sinks.register_sink(
        PlotlyJSONSink(
            sink_reference="plots",
            resource_location="buffer://session",
            buffer_store=store,
            session_id="s1",
        )
    )
    return sinks


def test_plot_2d_visualization_publishes_first_frame_for_higher_rank_data():
    processing = ProcessingData()
    sample = DataBundle()
    signal = np.arange(24, dtype=float).reshape(2, 3, 4)
    signal[0, 1, 2] = np.nan
    sample["image"] = BaseData(signal=signal, units=ureg.Unit("count"), rank_of_data=3)
    processing["sample"] = sample
    store = RuntimeBufferStore()

    step = Plot2DVisualization(processing_data=processing, io_sinks=_sinks(store), step_id="plot2d")
    step.modify_config_by_dict(
        {
            "target": "plots::detector",
            "data_path": "/sample/image",
            "title": "Detector image",
        }
    )

    assert step.calculate() == {}

    payload = store.get_metadata("s1", "sink", "plots", "detector")
    assert payload["schema_version"] == "modacor.plotly_2d.v1"
    assert payload["metadata"]["input_shape"] == [2, 3, 4]
    assert payload["metadata"]["display_shape"] == [3, 4]
    assert payload["metadata"]["frame_index"] == [0]
    assert payload["metadata"]["finite_pixels"] == 11
    assert payload["metadata"]["positive_pixels"] == 10
    assert payload["metadata"]["color_scale"]["scale"] == "log10"
    assert payload["metadata"]["color_scale"]["source_zmin"] == 1.0
    assert payload["metadata"]["color_scale"]["source_zmax"] == pytest.approx(10.91)
    assert payload["data"][0]["colorscale"] == "Plasma"
    assert payload["metadata"]["colormap"] == "Plasma"
    assert payload["data"][0]["z"][0] == [None, 0.0, pytest.approx(np.log10(2.0)), pytest.approx(np.log10(3.0))]
    assert payload["data"][0]["z"][1][2] is None
    assert payload["data"][0]["zmin"] == 0.0
    assert payload["data"][0]["zmax"] == pytest.approx(np.log10(10.91))
    assert payload["layout"]["yaxis"]["autorange"] == "reversed"
    assert payload["layout"]["uirevision"] == "plots::detector"


def test_plot_2d_visualization_accepts_colormap_configuration():
    processing = ProcessingData()
    sample = DataBundle()
    sample["image"] = BaseData(signal=np.array([[1.0, 2.0], [3.0, 4.0]]), units=ureg.Unit("count"), rank_of_data=2)
    processing["sample"] = sample
    store = RuntimeBufferStore()

    step = Plot2DVisualization(processing_data=processing, io_sinks=_sinks(store), step_id="plot2d")
    step.modify_config_by_dict(
        {
            "target": "plots::detector",
            "data_path": "/sample/image",
            "colormap": "Cividis",
        }
    )

    step.calculate()

    payload = store.get_metadata("s1", "sink", "plots", "detector")
    assert payload["data"][0]["colorscale"] == "Cividis"
    assert payload["metadata"]["colormap"] == "Cividis"
