# SPDX-License-Identifier: BSD-3-Clause
# /usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

__coding__ = "utf-8"
__authors__ = ["Brian R. Pauw"]  # add names to the list as appropriate
__copyright__ = "Copyright 2026, The MoDaCor team"
__date__ = "12/02/2026"
__status__ = "Development"  # "Development", "Production"
# end of header and standard imports

"""HDF5 sink for writing processing results, pipeline metadata, and trace events."""

import re
from pathlib import Path
from typing import Any, Sequence

import h5py
from attrs import define, field, validators

from modacor.dataclasses.basedata import BaseData
from modacor.dataclasses.messagehandler import MessageHandler
from modacor.dataclasses.processing_data import ProcessingData
from modacor.io.io_sink import IoSink
from modacor.io.processing_path import parse_processing_path

__all__ = ["HDFProcessingSink"]


def _normalise_subpath(subpath: str | None) -> str:
    if subpath is None or str(subpath).strip() == "":
        return "default"
    return str(subpath).strip("/") or "default"


def _recreate_group(parent: h5py.Group | h5py.File, name: str) -> h5py.Group:
    if name in parent:
        del parent[name]
    return parent.create_group(name)


def _write_basedata(group: h5py.Group, basedata: BaseData, *, compression: str | None = None) -> None:
    signal_dataset = group.create_dataset(
        "signal",
        data=basedata.signal,
        compression=compression,
    )
    signal_dataset.attrs["units"] = str(basedata.units)
    signal_dataset.attrs["rank_of_data"] = int(basedata.rank_of_data)

    # Weights: store only if non-scalar or not equal to 1.0
    weights_array = basedata.weights
    if weights_array.size > 1:
        group.create_dataset("weights", data=weights_array, compression=compression)
    else:
        group.attrs["weight_scalar"] = float(weights_array.ravel()[0])

    # Store uncertainties under dedicated subgroup
    if basedata.uncertainties:
        unc_group = group.create_group("uncertainties")
        for key, values in basedata.uncertainties.items():
            dset = unc_group.create_dataset(key, data=values, compression=compression)
            dset.attrs["units"] = str(basedata.units)


def _json_ready(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_ready(v) for v in value]
    if hasattr(value, "to_dict") and callable(getattr(value, "to_dict")):
        try:
            return _json_ready(value.to_dict())
        except Exception:  # pragma: no cover - defensive
            pass
    if hasattr(value, "item") and callable(getattr(value, "item")):
        try:
            return _json_ready(value.item())
        except Exception:  # pragma: no cover - defensive
            pass
    return str(value)


def _json_dumps_bytes(payload: Any) -> bytes:
    import json

    return json.dumps(_json_ready(payload), indent=2, sort_keys=True, ensure_ascii=False).encode("utf-8")


def _json_dumps_text(payload: Any) -> str:
    return _json_dumps_bytes(payload).decode("utf-8")


def _write_text_dataset(group: h5py.Group, name: str, text: str) -> None:
    group.create_dataset(name, data=str(text).encode("utf-8"))


def _normalise_trace_events(trace_events: Any | None) -> list[dict[str, Any]]:
    if trace_events is None:
        return []
    if isinstance(trace_events, list):
        normalised: list[dict[str, Any]] = []
        for event in trace_events:
            event_dict = _json_ready(event)
            if isinstance(event_dict, dict):
                normalised.append(event_dict)
        return normalised
    return []


def _safe_hdf_key(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return safe or "dataset"


def _write_trace_indexed(parent: h5py.Group, trace_events: list[dict[str, Any]]) -> None:
    parent.attrs["schema_version"] = "1.1"

    if not trace_events:
        parent.attrs["empty"] = True
        return

    steps_group = parent.create_group("steps")
    index_group = parent.create_group("index")

    step_ids: list[str] = []
    modules: list[str] = []
    durations: list[float] = []
    any_change: list[bool] = []

    for idx, event in enumerate(trace_events, start=1):
        step_id = str(event.get("step_id", "unknown"))
        module = str(event.get("module", ""))
        step_key = f"{idx:04d}_{_safe_hdf_key(step_id)}"

        step_group = steps_group.create_group(step_key)
        step_group.attrs["step_id"] = step_id
        step_group.attrs["module"] = module
        step_group.attrs["label"] = str(event.get("label", ""))
        step_group.attrs["config_hash"] = str(event.get("config_hash", ""))

        duration_raw = event.get("duration_s")
        if isinstance(duration_raw, (int, float)):
            duration_value = float(duration_raw)
            step_group.attrs["duration_s"] = duration_value
        else:
            duration_value = float("nan")

        requires_steps = event.get("requires_steps", []) or []
        req_dtype = h5py.string_dtype(encoding="utf-8")
        step_group.create_dataset("requires_steps", data=[str(v) for v in requires_steps], dtype=req_dtype)

        _write_text_dataset(step_group, "config_json", _json_dumps_text(event.get("config", {})))
        _write_text_dataset(step_group, "messages_json", _json_dumps_text(event.get("messages", [])))

        datasets_payload = event.get("datasets", {}) or {}
        datasets_group = step_group.create_group("datasets")
        changed_any = False
        if isinstance(datasets_payload, dict):
            for raw_key, dataset_payload in datasets_payload.items():
                payload = dataset_payload if isinstance(dataset_payload, dict) else {"value": dataset_payload}
                dataset_group = datasets_group.create_group(_safe_hdf_key(str(raw_key)))
                dataset_group.attrs["path"] = str(raw_key)
                diff = payload.get("diff", []) or []
                if diff:
                    changed_any = True
                dataset_group.create_dataset("changed_kinds", data=[str(v) for v in diff], dtype=req_dtype)
                _write_text_dataset(dataset_group, "prev_json", _json_dumps_text(payload.get("prev")))
                _write_text_dataset(dataset_group, "now_json", _json_dumps_text(payload.get("now")))

        step_ids.append(step_id)
        modules.append(module)
        durations.append(duration_value)
        any_change.append(changed_any)

    str_dtype = h5py.string_dtype(encoding="utf-8")
    index_group.create_dataset("step_ids", data=step_ids, dtype=str_dtype)
    index_group.create_dataset("modules", data=modules, dtype=str_dtype)
    index_group.create_dataset("durations_s", data=durations)
    index_group.create_dataset("any_change", data=any_change, dtype=bool)


@define(kw_only=True)
class HDFProcessingSink(IoSink):
    """Write selected ProcessingData leaves and metadata into an HDF5 file."""

    resource_location: Path = field(converter=Path, validator=validators.instance_of(Path))
    iosink_method_kwargs: dict[str, Any] = field(factory=dict, validator=validators.instance_of(dict))
    logger: MessageHandler = field(init=False)

    def __attrs_post_init__(self) -> None:
        self.logger = MessageHandler(level=self.logging_level, name="HDFProcessingSink")

    def write(
        self,
        subpath: str,
        processing_data: ProcessingData,
        data_paths: Sequence[str] | str,
        *,
        pipeline_spec: dict[str, Any] | None = None,
        pipeline_yaml: str | None = None,
        trace_events: Any | None = None,
        override_resource_location: Path | None = None,
    ) -> Path:
        if isinstance(data_paths, str):
            data_paths = [data_paths]
        elif not data_paths:
            raise ValueError("HDFProcessingSink.write requires one or more data_paths.")

        out_path = (override_resource_location or self.resource_location).expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)

        compression = self.iosink_method_kwargs.get("compression")
        resolved_pipeline_spec = (
            pipeline_spec if pipeline_spec is not None else self.iosink_method_kwargs.get("pipeline_spec")
        )
        resolved_pipeline_yaml = (
            pipeline_yaml if pipeline_yaml is not None else self.iosink_method_kwargs.get("pipeline_yaml")
        )
        resolved_trace_events = (
            trace_events if trace_events is not None else self.iosink_method_kwargs.get("trace_events")
        )

        run_name = _normalise_subpath(subpath)

        with h5py.File(out_path, "a") as h5:
            processing_group = h5.require_group("processing")

            result_root = processing_group.require_group("result")
            run_result_group = _recreate_group(result_root, run_name)

            for path in data_paths:
                parsed = parse_processing_path(path)
                bundle_key = parsed.databundle_key
                basedata_name = parsed.basedata_name
                if bundle_key is None or basedata_name is None:
                    raise ValueError(f"Processing path '{path}' does not reference a BaseData entry.")

                try:
                    databundle = processing_data[bundle_key]
                except KeyError as exc:  # pragma: no cover - defensive
                    raise KeyError(f"ProcessingData missing bundle '{bundle_key}'.") from exc

                try:
                    basedata = databundle[basedata_name]
                except KeyError as exc:  # pragma: no cover - defensive
                    raise KeyError(f"DataBundle '{bundle_key}' missing BaseData '{basedata_name}'.") from exc

                if not isinstance(basedata, BaseData):
                    raise TypeError(
                        f"Processing path '{path}' did not resolve to a BaseData instance (got"
                        f" {type(basedata).__name__})."
                    )

                bundle_group = run_result_group.require_group(bundle_key)
                basedata_group = _recreate_group(bundle_group, basedata_name)
                _write_basedata(basedata_group, basedata, compression=compression)

            # Pipeline specification (stored as JSON string)
            pipeline_group = processing_group.require_group("pipeline")
            pipeline_run_group = _recreate_group(pipeline_group, run_name)
            if resolved_pipeline_spec is None and resolved_pipeline_yaml is None:
                pipeline_run_group.attrs["empty"] = True
            else:
                if resolved_pipeline_spec is not None:
                    pipeline_run_group.create_dataset("spec", data=_json_dumps_bytes(resolved_pipeline_spec))
                if resolved_pipeline_yaml is not None:
                    _write_text_dataset(pipeline_run_group, "yaml", str(resolved_pipeline_yaml))

            # Trace events: keep raw JSON + indexed structure for querying
            tracer_group = processing_group.require_group("tracer")
            tracer_run_group = _recreate_group(tracer_group, run_name)
            if resolved_trace_events is None:
                tracer_run_group.attrs["empty"] = True
            else:
                tracer_run_group.create_dataset("events", data=_json_dumps_bytes(resolved_trace_events))
                _write_trace_indexed(tracer_run_group, _normalise_trace_events(resolved_trace_events))

        self.logger.info(f"Wrote processing results to {out_path} (run={run_name}).")
        return out_path
