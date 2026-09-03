# SPDX-License-Identifier: BSD-3-Clause
# /usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

__coding__ = "utf-8"
__authors__ = ["Brian R. Pauw"]
__copyright__ = "Copyright 2026, The MoDaCor team"
__date__ = "03/09/2026"
__status__ = "Development"

__all__ = ["Plot1DVisualization"]
__version__ = "20260903.1"

from pathlib import Path
from typing import Any

import numpy as np

from modacor.dataclasses.basedata import BaseData
from modacor.dataclasses.databundle import DataBundle
from modacor.dataclasses.process_step import ProcessStep, ProcessStepDependencies
from modacor.dataclasses.process_step_describer import ProcessStepDescriber
from modacor.io.processing_path import infer_units_for_path, parse_processing_path, resolve_processing_path


def _str_or_none(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _normalise_names(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    return [str(item).strip() for item in value if str(item).strip()]


def _processing_pattern(path: str | None) -> str | None:
    if not path:
        return None
    try:
        parsed = parse_processing_path(path)
    except (TypeError, ValueError):
        return "*"
    return f"{parsed.databundle_key}.{parsed.basedata_name}"


def _array_for_path(processing_data: Any, path: str, *, label: str) -> np.ndarray:
    arr = np.asarray(resolve_processing_path(processing_data, path), dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"Plot1DVisualization expects 1D {label} data, got shape {arr.shape} from {path!r}.")
    return arr


def _basedata_for_path(processing_data: Any, path: str) -> BaseData:
    parsed = parse_processing_path(path)
    basedata = processing_data[parsed.databundle_key][parsed.basedata_name]
    if not isinstance(basedata, BaseData):
        raise TypeError(f"Path {path!r} does not resolve to a BaseData root.")
    return basedata


class Plot1DVisualization(ProcessStep):
    """
    Publish a Plotly-compatible 1D plot payload through an IoSink.

    Direct error paths take precedence over uncertainty-name fallbacks.
    """

    documentation = ProcessStepDescriber(
        calling_name="Plot 1D Visualization",
        calling_id="Plot1DVisualization",
        calling_module_path=Path(__file__),
        calling_version=__version__,
        required_data_keys=[],
        modifies={},
        arguments={
            "target": {
                "type": str,
                "required": True,
                "default": "",
                "doc": "Sink target in the form 'plot_sink::plot_id'.",
            },
            "x_path": {
                "type": str,
                "required": True,
                "default": "",
                "doc": "ProcessingData path for the x array.",
            },
            "y_path": {
                "type": str,
                "required": True,
                "default": "",
                "doc": "ProcessingData path for the y array.",
            },
            "xerr_path": {
                "type": (str, type(None)),
                "default": None,
                "doc": "Optional ProcessingData path for x error bars.",
            },
            "yerr_path": {
                "type": (str, type(None)),
                "default": None,
                "doc": "Optional ProcessingData path for y error bars.",
            },
            "xerr_uncertainty_names": {
                "type": (list, str, type(None)),
                "default": None,
                "doc": "Optional uncertainty-name fallbacks below the x BaseData.",
            },
            "yerr_uncertainty_names": {
                "type": (list, str, type(None)),
                "default": None,
                "doc": "Optional uncertainty-name fallbacks below the y BaseData.",
            },
            "title": {
                "type": (str, type(None)),
                "default": None,
                "doc": "Optional plot title.",
            },
            "auto_log_x": {
                "type": bool,
                "default": True,
                "doc": "Use a logarithmic x axis when all valid x values are positive.",
            },
            "auto_log_y": {
                "type": bool,
                "default": True,
                "doc": "Use a logarithmic y axis when all valid y values are positive.",
            },
        },
        step_keywords=["plot", "visualization", "plotly", "1d"],
        step_doc="Publish a Plotly-compatible 1D data plot payload.",
        step_reference="",
        step_note="This step performs a visualization side-effect and returns an empty output dict.",
    )

    def dependency_contract(self) -> ProcessStepDependencies:
        cfg = self.configuration or {}
        read_patterns = {
            pattern
            for pattern in (
                _processing_pattern(_str_or_none(cfg.get("x_path"))),
                _processing_pattern(_str_or_none(cfg.get("y_path"))),
                _processing_pattern(_str_or_none(cfg.get("xerr_path"))),
                _processing_pattern(_str_or_none(cfg.get("yerr_path"))),
            )
            if pattern
        }
        if not read_patterns:
            read_patterns.add("*")
        return ProcessStepDependencies(processing_reads=read_patterns, processing_writes=())

    def _resolve_error(
        self,
        *,
        data_path: str,
        error_path: str | None,
        names: list[str],
        label: str,
    ) -> tuple[np.ndarray | None, str | None, str | None]:
        if error_path:
            return _array_for_path(self.processing_data, error_path, label=label), error_path, error_path

        if not names:
            return None, None, None

        basedata = _basedata_for_path(self.processing_data, data_path)
        for preferred_name in names:
            if preferred_name in basedata.uncertainties:
                return np.asarray(basedata.uncertainties[preferred_name], dtype=float), preferred_name, None
            for actual_name in basedata.uncertainties:
                if str(actual_name).lower() == preferred_name.lower():
                    return np.asarray(basedata.uncertainties[actual_name], dtype=float), str(actual_name), None
        return None, None, None

    def calculate(self) -> dict[str, DataBundle]:
        cfg = self.configuration or {}
        target = _str_or_none(cfg.get("target"))
        x_path = _str_or_none(cfg.get("x_path"))
        y_path = _str_or_none(cfg.get("y_path"))
        if not target:
            raise ValueError("Plot1DVisualization requires a non-empty 'target'.")
        if not x_path or not y_path:
            raise ValueError("Plot1DVisualization requires non-empty 'x_path' and 'y_path'.")
        if self.io_sinks is None:
            raise RuntimeError("Plot1DVisualization requires io_sinks.")

        x = _array_for_path(self.processing_data, x_path, label="x")
        y = _array_for_path(self.processing_data, y_path, label="y")
        if x.shape != y.shape:
            raise ValueError(
                f"Plot1DVisualization requires x and y arrays with identical shape, got {x.shape} and {y.shape}."
            )

        xerr, xerr_name, xerr_source = self._resolve_error(
            data_path=x_path,
            error_path=_str_or_none(cfg.get("xerr_path")),
            names=_normalise_names(cfg.get("xerr_uncertainty_names")),
            label="xerr",
        )
        yerr, yerr_name, yerr_source = self._resolve_error(
            data_path=y_path,
            error_path=_str_or_none(cfg.get("yerr_path")),
            names=_normalise_names(cfg.get("yerr_uncertainty_names")),
            label="yerr",
        )

        valid = np.isfinite(x) & np.isfinite(y)
        if xerr is not None:
            if xerr.shape != x.shape:
                raise ValueError(f"xerr shape {xerr.shape} does not match x shape {x.shape}.")
            valid &= np.isfinite(xerr)
        if yerr is not None:
            if yerr.shape != y.shape:
                raise ValueError(f"yerr shape {yerr.shape} does not match y shape {y.shape}.")
            valid &= np.isfinite(yerr)

        x_valid = x[valid]
        y_valid = y[valid]
        x_units = infer_units_for_path(self.processing_data, x_path)
        y_units = infer_units_for_path(self.processing_data, y_path)
        title = _str_or_none(cfg.get("title")) or f"{y_path} vs {x_path}"

        trace: dict[str, Any] = {
            "type": "scatter",
            "mode": "markers",
            "name": title,
            "x": x_valid.tolist(),
            "y": y_valid.tolist(),
            "marker": {"size": 5, "opacity": 0.8},
        }
        if xerr is not None:
            trace["error_x"] = {"type": "data", "array": xerr[valid].tolist(), "visible": True, "thickness": 0.7}
        if yerr is not None:
            trace["error_y"] = {"type": "data", "array": yerr[valid].tolist(), "visible": True, "thickness": 0.7}

        layout: dict[str, Any] = {
            "title": {"text": title},
            "xaxis": {"title": {"text": f"Q ({x_units})" if x_units else "Q"}, "showgrid": True},
            "yaxis": {"title": {"text": f"Signal ({y_units})" if y_units else "Signal"}, "showgrid": True},
            "margin": {"l": 76, "r": 30, "t": 56, "b": 64},
            "template": "plotly_white",
        }
        if bool(cfg.get("auto_log_x", True)) and x_valid.size and np.all(x_valid > 0):
            layout["xaxis"]["type"] = "log"
        if bool(cfg.get("auto_log_y", True)) and y_valid.size and np.all(y_valid > 0):
            layout["yaxis"]["type"] = "log"

        payload = {
            "schema_version": "modacor.plotly_1d.v1",
            "data": [trace],
            "layout": layout,
            "metadata": {
                "x_path": x_path,
                "y_path": y_path,
                "xerr_path": xerr_source,
                "yerr_path": yerr_source,
                "xerr_name": xerr_name,
                "yerr_name": yerr_name,
                "x_units": x_units,
                "y_units": y_units,
                "valid_points": int(valid.sum()),
                "total_points": int(valid.size),
            },
        }
        self.io_sinks.write_data(target, payload)
        return {}
