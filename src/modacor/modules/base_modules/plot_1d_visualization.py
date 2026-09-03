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
from attrs import define
from pint.errors import PintError

from modacor import ureg
from modacor.dataclasses.basedata import BaseData
from modacor.dataclasses.databundle import DataBundle
from modacor.dataclasses.process_step import ProcessStep, ProcessStepDependencies
from modacor.dataclasses.process_step_describer import ProcessStepDescriber
from modacor.io.processing_path import infer_units_for_path, parse_processing_path, resolve_processing_path

ERROR_BAR_COLORS = [
    "#1f77b4",
    "#d62728",
    "#2ca02c",
    "#9467bd",
    "#ff7f0e",
    "#17becf",
    "#8c564b",
    "#7f7f7f",
]
DEFAULT_ERROR_BAR_OPACITY = 0.65
DEFAULT_ERROR_BAR_THICKNESS = 2.0
DEFAULT_ERROR_BAR_WIDTH = 3


@define(frozen=True, slots=True)
class ErrorCandidate:
    values: np.ndarray
    name: str
    source: str | None


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


def _rgba(hex_color: str, opacity: float) -> str:
    color = hex_color.lstrip("#")
    if len(color) != 6:
        return hex_color
    red = int(color[0:2], 16)
    green = int(color[2:4], 16)
    blue = int(color[4:6], 16)
    return f"rgba({red}, {green}, {blue}, {opacity:.3g})"


def _convert_values(values: np.ndarray, *, from_units: str, to_units: str | None, label: str) -> tuple[np.ndarray, str]:
    if not to_units or not from_units or to_units == from_units:
        return values, from_units
    try:
        converted = ureg.Quantity(values, from_units).to(to_units).magnitude
    except (PintError, TypeError, ValueError) as exc:
        raise ValueError(f"Could not convert {label} from {from_units!r} to {to_units!r}.") from exc
    return np.asarray(converted, dtype=float), to_units


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
            "x_units": {
                "type": (str, type(None)),
                "default": None,
                "doc": "Optional display units for x values and x error bars.",
            },
            "y_units": {
                "type": (str, type(None)),
                "default": None,
                "doc": "Optional display units for y values and y error bars.",
            },
            "error_bar_thickness": {
                "type": (int, float),
                "default": DEFAULT_ERROR_BAR_THICKNESS,
                "doc": "Plotly error bar line thickness.",
            },
            "error_bar_opacity": {
                "type": (int, float),
                "default": DEFAULT_ERROR_BAR_OPACITY,
                "doc": "Error bar opacity encoded in the error bar colour.",
            },
            "error_bar_width": {
                "type": (int, float),
                "default": DEFAULT_ERROR_BAR_WIDTH,
                "doc": "Plotly error bar cap width.",
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

    def _resolve_axis_errors(
        self,
        *,
        data_path: str,
        error_path: str | None,
        names: list[str],
        label: str,
    ) -> list[ErrorCandidate]:
        if error_path:
            return [
                ErrorCandidate(
                    values=_array_for_path(self.processing_data, error_path, label=label),
                    name=error_path,
                    source=error_path,
                )
            ]

        if not names:
            return []

        basedata = _basedata_for_path(self.processing_data, data_path)
        candidates: list[ErrorCandidate] = []
        seen: set[str] = set()
        for preferred_name in names:
            match_name: str | None = None
            if preferred_name in basedata.uncertainties:
                match_name = preferred_name
            else:
                for actual_name in basedata.uncertainties:
                    if str(actual_name).lower() == preferred_name.lower():
                        match_name = str(actual_name)
                        break
            if match_name is None or match_name in seen:
                continue
            seen.add(match_name)
            candidates.append(
                ErrorCandidate(
                    values=np.asarray(basedata.uncertainties[match_name], dtype=float),
                    name=match_name,
                    source=None,
                )
            )
        return candidates

    @staticmethod
    def _trace_name(title: str, xerr: ErrorCandidate | None, yerr: ErrorCandidate | None) -> str:
        parts = []
        if xerr is not None:
            parts.append(f"Q +/- {xerr.name}")
        if yerr is not None:
            parts.append(f"I +/- {yerr.name}")
        return title if not parts else f"{title}: {'; '.join(parts)}"

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

        x_source_units = infer_units_for_path(self.processing_data, x_path)
        y_source_units = infer_units_for_path(self.processing_data, y_path)
        x, x_units = _convert_values(x, from_units=x_source_units, to_units=_str_or_none(cfg.get("x_units")), label="x")
        y, y_units = _convert_values(y, from_units=y_source_units, to_units=_str_or_none(cfg.get("y_units")), label="y")

        xerr_candidates = self._resolve_axis_errors(
            data_path=x_path,
            error_path=_str_or_none(cfg.get("xerr_path")),
            names=_normalise_names(cfg.get("xerr_uncertainty_names")),
            label="xerr",
        )
        yerr_candidates = self._resolve_axis_errors(
            data_path=y_path,
            error_path=_str_or_none(cfg.get("yerr_path")),
            names=_normalise_names(cfg.get("yerr_uncertainty_names")),
            label="yerr",
        )

        base_valid = np.isfinite(x) & np.isfinite(y)
        for candidate in xerr_candidates:
            if candidate.values.shape != x.shape:
                raise ValueError(f"xerr shape {candidate.values.shape} does not match x shape {x.shape}.")
        for candidate in yerr_candidates:
            if candidate.values.shape != y.shape:
                raise ValueError(f"yerr shape {candidate.values.shape} does not match y shape {y.shape}.")

        xerr_candidates = [
            ErrorCandidate(
                values=_convert_values(candidate.values, from_units=x_source_units, to_units=x_units, label="xerr")[0],
                name=candidate.name,
                source=candidate.source,
            )
            for candidate in xerr_candidates
        ]
        yerr_candidates = [
            ErrorCandidate(
                values=_convert_values(candidate.values, from_units=y_source_units, to_units=y_units, label="yerr")[0],
                name=candidate.name,
                source=candidate.source,
            )
            for candidate in yerr_candidates
        ]
        title = _str_or_none(cfg.get("title")) or f"{y_path} vs {x_path}"
        error_bar_opacity = float(cfg.get("error_bar_opacity", DEFAULT_ERROR_BAR_OPACITY))
        error_bar_thickness = float(cfg.get("error_bar_thickness", DEFAULT_ERROR_BAR_THICKNESS))
        error_bar_width = float(cfg.get("error_bar_width", DEFAULT_ERROR_BAR_WIDTH))

        traces: list[dict[str, Any]] = []
        first_valid: np.ndarray | None = None
        trace_count = max(len(xerr_candidates), len(yerr_candidates), 1)
        for index in range(trace_count):
            xerr_candidate = xerr_candidates[index] if index < len(xerr_candidates) else None
            yerr_candidate = yerr_candidates[index] if index < len(yerr_candidates) else None
            valid = base_valid.copy()
            if xerr_candidate is not None:
                valid &= np.isfinite(xerr_candidate.values)
            if yerr_candidate is not None:
                valid &= np.isfinite(yerr_candidate.values)
            if first_valid is None:
                first_valid = valid
            color = ERROR_BAR_COLORS[index % len(ERROR_BAR_COLORS)]
            marker_size = 5 if index == 0 else 0
            trace: dict[str, Any] = {
                "type": "scatter",
                "mode": "markers",
                "name": self._trace_name(title, xerr_candidate, yerr_candidate),
                "x": x[valid].tolist(),
                "y": y[valid].tolist(),
                "marker": {"size": marker_size, "opacity": 0.85, "color": color},
                "showlegend": True,
            }
            error_bar_color = _rgba(color, error_bar_opacity)
            if xerr_candidate is not None:
                trace["error_x"] = {
                    "type": "data",
                    "array": xerr_candidate.values[valid].tolist(),
                    "visible": True,
                    "thickness": error_bar_thickness,
                    "width": error_bar_width,
                    "color": error_bar_color,
                }
            if yerr_candidate is not None:
                trace["error_y"] = {
                    "type": "data",
                    "array": yerr_candidate.values[valid].tolist(),
                    "visible": True,
                    "thickness": error_bar_thickness,
                    "width": error_bar_width,
                    "color": error_bar_color,
                }
            traces.append(trace)

        axis_valid = first_valid if first_valid is not None else base_valid
        x_valid = x[axis_valid]
        y_valid = y[axis_valid]

        layout: dict[str, Any] = {
            "title": {"text": title},
            "xaxis": {"title": {"text": f"Q ({x_units})" if x_units else "Q"}, "showgrid": True},
            "yaxis": {"title": {"text": f"Signal ({y_units})" if y_units else "Signal"}, "showgrid": True},
            "margin": {"l": 76, "r": 30, "t": 56, "b": 64},
            "template": "plotly_white",
            "showlegend": True,
        }
        if bool(cfg.get("auto_log_x", True)) and x_valid.size and np.all(x_valid > 0):
            layout["xaxis"]["type"] = "log"
        if bool(cfg.get("auto_log_y", True)) and y_valid.size and np.all(y_valid > 0):
            layout["yaxis"]["type"] = "log"

        payload = {
            "schema_version": "modacor.plotly_1d.v1",
            "data": traces,
            "layout": layout,
            "metadata": {
                "x_path": x_path,
                "y_path": y_path,
                "xerr_path": xerr_candidates[0].source if xerr_candidates else None,
                "yerr_path": yerr_candidates[0].source if yerr_candidates else None,
                "xerr_name": xerr_candidates[0].name if xerr_candidates else None,
                "yerr_name": yerr_candidates[0].name if yerr_candidates else None,
                "xerr_names": [candidate.name for candidate in xerr_candidates],
                "yerr_names": [candidate.name for candidate in yerr_candidates],
                "x_units": x_units,
                "y_units": y_units,
                "valid_points": int(axis_valid.sum()),
                "total_points": int(axis_valid.size),
            },
        }
        self.io_sinks.write_data(target, payload)
        return {}
