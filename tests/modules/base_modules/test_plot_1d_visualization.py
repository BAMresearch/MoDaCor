# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import numpy as np

from modacor import ureg
from modacor.dataclasses.basedata import BaseData
from modacor.dataclasses.databundle import DataBundle
from modacor.dataclasses.processing_data import ProcessingData
from modacor.io.buffer.runtime_buffer_store import RuntimeBufferStore
from modacor.io.io_sinks import IoSinks
from modacor.io.visualization import PlotlyJSONSink
from modacor.modules.base_modules.plot_1d_visualization import Plot1DVisualization


def _processing_data() -> ProcessingData:
    processing = ProcessingData()
    sample = DataBundle()
    q = BaseData(signal=np.array([0.1, 0.2, 0.3]), units=ureg.Unit("1/nm"))
    q.uncertainties["SEM"] = np.array([0.01, 0.02, 0.03])
    signal = BaseData(signal=np.array([10.0, np.nan, 30.0]), units=ureg.Unit("count"))
    signal.uncertainties["Poisson"] = np.array([1.0, 2.0, 3.0])
    sample["Q"] = q
    sample["signal"] = signal
    processing["sample"] = sample
    return processing


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


def test_plot_1d_visualization_publishes_plotly_payload_with_uncertainties():
    store = RuntimeBufferStore()
    step = Plot1DVisualization(processing_data=_processing_data(), io_sinks=_sinks(store), step_id="plot")
    step.modify_config_by_dict(
        {
            "target": "plots::corrected",
            "x_path": "/sample/Q/signal",
            "y_path": "/sample/signal/signal",
            "xerr_uncertainty_names": ["uncertainty_combined", "SEM"],
            "yerr_uncertainty_names": ["poisson"],
            "title": "I22 corrected curve",
        }
    )

    assert step.calculate() == {}

    payload = store.get_metadata("s1", "sink", "plots", "corrected")
    trace = payload["data"][0]
    assert trace["x"] == [0.1, 0.3]
    assert trace["y"] == [10.0, 30.0]
    assert trace["error_x"]["array"] == [0.01, 0.03]
    assert trace["error_y"]["array"] == [1.0, 3.0]
    assert payload["metadata"]["xerr_name"] == "SEM"
    assert payload["metadata"]["yerr_name"] == "Poisson"
    assert payload["metadata"]["valid_points"] == 2
    assert payload["metadata"]["total_points"] == 3
    assert payload["layout"]["xaxis"]["type"] == "log"
    assert payload["layout"]["yaxis"]["type"] == "log"


def test_plot_1d_visualization_direct_error_path_takes_precedence():
    processing = _processing_data()
    processing["sample"]["signal"].uncertainties["SEM"] = np.array([4.0, 5.0, 6.0])
    store = RuntimeBufferStore()
    step = Plot1DVisualization(processing_data=processing, io_sinks=_sinks(store), step_id="plot")
    step.modify_config_by_dict(
        {
            "target": "plots::direct",
            "x_path": "/sample/Q/signal",
            "y_path": "/sample/signal/signal",
            "yerr_path": "/sample/signal/uncertainties/SEM",
            "yerr_uncertainty_names": ["Poisson"],
        }
    )

    step.calculate()

    payload = store.get_metadata("s1", "sink", "plots", "direct")
    assert payload["data"][0]["error_y"]["array"] == [4.0, 6.0]
    assert payload["metadata"]["yerr_name"] == "/sample/signal/uncertainties/SEM"
    assert payload["metadata"]["yerr_path"] == "/sample/signal/uncertainties/SEM"
