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


def _data_traces(payload):
    return [trace for trace in payload["data"] if trace.get("meta", {}).get("modacor_role") != "legend"]


def _legend_traces(payload):
    return [trace for trace in payload["data"] if trace.get("meta", {}).get("modacor_role") == "legend"]


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
    trace = _data_traces(payload)[0]
    legend_trace = _legend_traces(payload)[0]
    assert trace["x"] == [0.1, 0.3]
    assert trace["y"] == [10.0, 30.0]
    assert trace["error_x"]["array"] == [0.01, 0.03]
    assert trace["error_y"]["array"] == [1.0, 3.0]
    assert trace["error_x"]["thickness"] == 2.0
    assert trace["error_y"]["width"] == 3.0
    assert trace["error_y"]["color"] == "rgba(31, 119, 180, 0.65)"
    assert trace["showlegend"] is False
    assert legend_trace["showlegend"] is True
    assert legend_trace["marker"]["size"] == 9.0
    assert legend_trace["legendgroup"] == trace["legendgroup"]
    assert payload["metadata"]["xerr_name"] == "SEM"
    assert payload["metadata"]["yerr_name"] == "Poisson"
    assert payload["metadata"]["valid_points"] == 2
    assert payload["metadata"]["total_points"] == 3
    assert payload["layout"]["xaxis"]["type"] == "log"
    assert payload["layout"]["yaxis"]["type"] == "log"
    assert payload["layout"]["uirevision"] == "plots::corrected"


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
    assert _data_traces(payload)[0]["error_y"]["array"] == [4.0, 6.0]
    assert payload["metadata"]["yerr_name"] == "/sample/signal/uncertainties/SEM"
    assert payload["metadata"]["yerr_path"] == "/sample/signal/uncertainties/SEM"


def test_plot_1d_visualization_publishes_multiple_y_uncertainty_traces():
    processing = _processing_data()
    processing["sample"]["signal"].uncertainties["SEM"] = np.array([4.0, 5.0, 6.0])
    processing["sample"]["signal"].uncertainties["combined"] = np.array([7.0, 8.0, 9.0])
    store = RuntimeBufferStore()
    step = Plot1DVisualization(processing_data=processing, io_sinks=_sinks(store), step_id="plot")
    step.modify_config_by_dict(
        {
            "target": "plots::multi",
            "x_path": "/sample/Q/signal",
            "y_path": "/sample/signal/signal",
            "yerr_uncertainty_names": ["combined", "SEM", "Poisson"],
        }
    )

    step.calculate()

    payload = store.get_metadata("s1", "sink", "plots", "multi")
    traces = _data_traces(payload)
    legend_traces = _legend_traces(payload)
    assert [trace["error_y"]["array"] for trace in traces] == [[7.0, 9.0], [4.0, 6.0], [1.0, 3.0]]
    assert [trace["marker"]["size"] for trace in traces] == [5, 0, 0]
    assert len({trace["error_y"]["color"] for trace in traces}) == 3
    assert all(not trace["showlegend"] for trace in traces)
    assert [trace["marker"]["size"] for trace in legend_traces] == [9.0, 9.0, 9.0]
    assert all(trace["showlegend"] for trace in legend_traces)
    assert payload["layout"]["showlegend"] is True
    assert payload["layout"]["legend"]["groupclick"] == "togglegroup"
    assert payload["metadata"]["yerr_name"] == "combined"
    assert payload["metadata"]["yerr_names"] == ["combined", "SEM", "Poisson"]


def test_plot_1d_visualization_publishes_multiple_x_and_y_uncertainty_traces():
    processing = _processing_data()
    processing["sample"]["Q"].uncertainties["STD"] = np.array([0.04, 0.05, 0.06])
    processing["sample"]["signal"].uncertainties["SEM"] = np.array([4.0, 5.0, 6.0])
    store = RuntimeBufferStore()
    step = Plot1DVisualization(processing_data=processing, io_sinks=_sinks(store), step_id="plot")
    step.modify_config_by_dict(
        {
            "target": "plots::multi_xy",
            "x_path": "/sample/Q/signal",
            "y_path": "/sample/signal/signal",
            "xerr_uncertainty_names": ["SEM", "STD"],
            "yerr_uncertainty_names": ["Poisson", "SEM"],
        }
    )

    step.calculate()

    payload = store.get_metadata("s1", "sink", "plots", "multi_xy")
    traces = _data_traces(payload)
    legend_traces = _legend_traces(payload)
    assert len(traces) == 2
    assert len(legend_traces) == 2
    assert [trace["marker"]["size"] for trace in traces] == [5, 0]
    assert [trace["marker"]["size"] for trace in legend_traces] == [9.0, 9.0]
    assert traces[0]["error_x"]["array"] == [0.01, 0.03]
    assert traces[0]["error_y"]["array"] == [1.0, 3.0]
    assert traces[1]["error_x"]["array"] == [0.04, 0.06]
    assert traces[1]["error_y"]["array"] == [4.0, 6.0]
    assert traces[0]["name"].endswith("Q +/- SEM; I +/- Poisson")
    assert traces[1]["name"].endswith("Q +/- STD; I +/- SEM")
    assert payload["metadata"]["xerr_name"] == "SEM"
    assert payload["metadata"]["xerr_names"] == ["SEM", "STD"]
    assert payload["metadata"]["yerr_names"] == ["Poisson", "SEM"]


def test_plot_1d_visualization_converts_x_units_and_uncertainties_for_display():
    processing = _processing_data()
    processing["sample"]["Q"].units = ureg.Unit("1/angstrom")
    store = RuntimeBufferStore()
    step = Plot1DVisualization(processing_data=processing, io_sinks=_sinks(store), step_id="plot")
    step.modify_config_by_dict(
        {
            "target": "plots::converted_q",
            "x_path": "/sample/Q/signal",
            "y_path": "/sample/signal/signal",
            "xerr_uncertainty_names": ["SEM"],
            "x_units": "1/nm",
        }
    )

    step.calculate()

    payload = store.get_metadata("s1", "sink", "plots", "converted_q")
    trace = _data_traces(payload)[0]
    np.testing.assert_allclose(trace["x"], [1.0, 3.0])
    np.testing.assert_allclose(trace["error_x"]["array"], [0.1, 0.3])
    assert payload["layout"]["xaxis"]["title"]["text"] == "Q (1/nm)"
    assert payload["metadata"]["x_units"] == "1/nm"
