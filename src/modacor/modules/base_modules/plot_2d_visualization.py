# SPDX-License-Identifier: BSD-3-Clause
# /usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

__coding__ = "utf-8"
__authors__ = ["Brian R. Pauw"]
__copyright__ = "Copyright 2026, The MoDaCor team"
__date__ = "03/09/2026"
__status__ = "Development"

__all__ = ["Plot2DVisualization"]
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


def _processing_pattern(path: str | None) -> str | None:
    if not path:
        return None
    try:
        parsed = parse_processing_path(path)
    except (TypeError, ValueError):
        return "*"
    return f"{parsed.databundle_key}.{parsed.basedata_name}"


def _resolve_array(processing_data: Any, path: str) -> np.ndarray:
    value = resolve_processing_path(processing_data, path)
    if isinstance(value, BaseData):
        value = value.signal
    return np.asarray(value, dtype=float)


def _units_for_path(processing_data: Any, path: str) -> str:
    parsed = parse_processing_path(path)
    if not parsed.subpath:
        basedata = processing_data[parsed.databundle_key][parsed.basedata_name]
        if isinstance(basedata, BaseData):
            return str(basedata.units)
    return infer_units_for_path(processing_data, path)


def _first_2d_frame(array: np.ndarray) -> tuple[np.ndarray, list[int]]:
    if array.ndim < 2:
        raise ValueError(f"Plot2DVisualization expects at least 2D data, got shape {array.shape}.")
    if array.ndim == 2:
        return array, []
    frame_index = [0] * (array.ndim - 2)
    return array[tuple(frame_index + [slice(None), slice(None)])], frame_index


def _plotly_z_values(array: np.ndarray) -> list[list[float | None]]:
    return [[float(value) if np.isfinite(value) else None for value in row] for row in array]


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _configured_colormap(cfg: dict[str, Any]) -> str:
    colormap = str(cfg.get("colormap") or "Plasma")
    colorscale = _str_or_none(cfg.get("colorscale"))
    if colorscale is not None and colormap == "Plasma":
        return colorscale
    return colormap


def _scaled_frame(
    frame: np.ndarray,
    *,
    scale: str,
    zmin: float | None,
    zmax: float | None,
    percentile: float,
) -> tuple[np.ndarray, float | None, float | None, dict[str, Any]]:
    finite = np.isfinite(frame)
    if scale == "linear":
        finite_values = frame[finite]
        if finite_values.size == 0:
            return frame, zmin, zmax, {"scale": "linear", "reason": "no_finite_pixels"}
        lower = float(np.nanmin(finite_values)) if zmin is None else zmin
        upper = float(np.nanpercentile(finite_values, percentile)) if zmax is None else zmax
        if upper <= lower:
            upper = float(np.nanmax(finite_values))
        return frame, lower, upper, {"scale": "linear", "percentile": percentile}

    if scale != "log10":
        raise ValueError("Plot2DVisualization scale must be one of: log10, linear.")

    positive = finite & (frame > 0)
    positive_values = frame[positive]
    if positive_values.size == 0:
        return frame, zmin, zmax, {"scale": "linear", "requested_scale": "log10", "reason": "no_positive_pixels"}

    lower = float(np.nanmin(positive_values)) if zmin is None else zmin
    upper = float(np.nanpercentile(positive_values, percentile)) if zmax is None else zmax
    if lower <= 0:
        raise ValueError("Plot2DVisualization log10 scale requires zmin to be positive when provided.")
    if upper <= lower:
        upper = float(np.nanmax(positive_values))

    scaled = np.full(frame.shape, np.nan, dtype=float)
    scaled[positive] = np.log10(frame[positive])
    return (
        scaled,
        float(np.log10(lower)),
        float(np.log10(upper)) if upper > 0 else None,
        {
            "scale": "log10",
            "source_zmin": lower,
            "source_zmax": upper,
            "percentile": percentile,
        },
    )


class Plot2DVisualization(ProcessStep):
    """Publish a Plotly-compatible 2D heatmap payload through an IoSink."""

    documentation = ProcessStepDescriber(
        calling_name="Plot 2D Visualization",
        calling_id="Plot2DVisualization",
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
            "data_path": {
                "type": str,
                "required": True,
                "default": "",
                "doc": "ProcessingData path for the 2D array or BaseData root.",
            },
            "title": {
                "type": (str, type(None)),
                "default": None,
                "doc": "Optional plot title.",
            },
            "colorscale": {
                "type": (str, type(None)),
                "default": None,
                "doc": "Backward-compatible alias for colormap.",
            },
            "colormap": {
                "type": str,
                "default": "Plasma",
                "doc": "Plotly colormap/colorscale name.",
            },
            "zmin": {
                "type": (int, float, type(None)),
                "default": None,
                "doc": "Optional lower color scale bound.",
            },
            "zmax": {
                "type": (int, float, type(None)),
                "default": None,
                "doc": "Optional upper color scale bound.",
            },
            "scale": {
                "type": str,
                "default": "log10",
                "doc": "Color scaling: 'log10' or 'linear'.",
            },
            "auto_zmax_percentile": {
                "type": (int, float),
                "default": 99.0,
                "doc": "Percentile used for automatic zmax on the displayed frame.",
            },
            "transpose": {
                "type": bool,
                "default": False,
                "doc": "Transpose the displayed 2D frame.",
            },
            "reverse_y": {
                "type": bool,
                "default": True,
                "doc": "Reverse the y axis for detector-image style display.",
            },
        },
        step_keywords=["plot", "visualization", "plotly", "2d", "image"],
        step_doc="Publish a Plotly-compatible 2D heatmap payload.",
        step_reference="",
        step_note="Higher-dimensional data is sliced to the first frame over leading dimensions.",
    )

    def dependency_contract(self) -> ProcessStepDependencies:
        pattern = _processing_pattern(_str_or_none((self.configuration or {}).get("data_path")))
        return ProcessStepDependencies(processing_reads={pattern or "*"}, processing_writes=())

    def calculate(self) -> dict[str, DataBundle]:
        cfg = self.configuration or {}
        target = _str_or_none(cfg.get("target"))
        data_path = _str_or_none(cfg.get("data_path"))
        if not target:
            raise ValueError("Plot2DVisualization requires a non-empty 'target'.")
        if not data_path:
            raise ValueError("Plot2DVisualization requires a non-empty 'data_path'.")
        if self.io_sinks is None:
            raise RuntimeError("Plot2DVisualization requires io_sinks.")

        raw = _resolve_array(self.processing_data, data_path)
        frame, frame_index = _first_2d_frame(raw)
        if bool(cfg.get("transpose", False)):
            frame = frame.T

        title = _str_or_none(cfg.get("title")) or data_path
        units = _units_for_path(self.processing_data, data_path)
        finite = np.isfinite(frame)
        percentile = float(cfg.get("auto_zmax_percentile", 99.0))
        if not 0 < percentile <= 100:
            raise ValueError("Plot2DVisualization auto_zmax_percentile must be in the range (0, 100].")
        scale = str(cfg.get("scale") or "log10").strip().lower()
        scaled_frame, zmin, zmax, scale_metadata = _scaled_frame(
            frame,
            scale=scale,
            zmin=_float_or_none(cfg.get("zmin")),
            zmax=_float_or_none(cfg.get("zmax")),
            percentile=percentile,
        )

        trace: dict[str, Any] = {
            "type": "heatmap",
            "z": _plotly_z_values(scaled_frame),
            "colorscale": _configured_colormap(cfg),
            "colorbar": {
                "title": {"text": f"log10({units})" if scale_metadata["scale"] == "log10" and units else units}
            },
        }
        if zmin is not None:
            trace["zmin"] = zmin
        if zmax is not None:
            trace["zmax"] = zmax

        layout: dict[str, Any] = {
            "title": {"text": title},
            "xaxis": {"title": {"text": "x pixel"}, "showgrid": False},
            "yaxis": {"title": {"text": "y pixel"}, "showgrid": False, "scaleanchor": "x"},
            "margin": {"l": 70, "r": 70, "t": 56, "b": 58},
            "template": "plotly_white",
        }
        if bool(cfg.get("reverse_y", True)):
            layout["yaxis"]["autorange"] = "reversed"

        payload = {
            "schema_version": "modacor.plotly_2d.v1",
            "data": [trace],
            "layout": layout,
            "metadata": {
                "data_path": data_path,
                "units": units,
                "input_shape": list(raw.shape),
                "display_shape": list(frame.shape),
                "frame_index": frame_index,
                "finite_pixels": int(finite.sum()),
                "positive_pixels": int((finite & (frame > 0)).sum()),
                "total_pixels": int(finite.size),
                "color_scale": scale_metadata,
                "colormap": _configured_colormap(cfg),
            },
        }
        self.io_sinks.write_data(target, payload)
        return {}
