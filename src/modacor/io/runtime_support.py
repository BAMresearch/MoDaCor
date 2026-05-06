# SPDX-License-Identifier: BSD-3-Clause
# /usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import Any, Iterable, Mapping

from modacor.io.csv.csv_sink import CSVSink
from modacor.io.csv.csv_source import CSVSource
from modacor.io.hdf.hdf_processing_sink import HDFProcessingSink
from modacor.io.hdf.hdf_source import HDFSource
from modacor.io.io_sinks import IoSinks
from modacor.io.io_sources import IoSources
from modacor.io.yaml.yaml_source import YAMLSource

__all__ = ["build_sinks_from_specs", "build_sources_from_specs", "write_processing_data_hdf"]


def _load_custom_class(class_path: str):
    module_path, class_name = class_path.rsplit(".", 1)
    module = import_module(module_path)
    return getattr(module, class_name)


def build_sources_from_specs(specs: Iterable[Mapping[str, Any]]) -> IoSources:
    """
    Build an :class:`IoSources` registry from normalized source specifications.

    Each spec must provide `ref`, `type`, and `location`. Optional `kwargs`
    are passed through as `iosource_method_kwargs`, except for `custom`
    sources where `kwargs.class_path` selects the source class.
    """

    type_map: dict[str, Any] = {
        "csv": CSVSource,
        "hdf": HDFSource,
        "yaml": YAMLSource,
    }
    sources = IoSources()

    for spec in specs:
        ref = str(spec["ref"]).strip()
        source_type = str(spec["type"]).strip().lower()
        location = Path(str(spec["location"]))
        kwargs = dict(spec.get("kwargs", {}) or {})

        if source_type == "custom":
            class_path = kwargs.pop("class_path", None)
            if not class_path:
                raise ValueError(f"Custom source '{ref}' requires kwargs.class_path.")
            source_cls = _load_custom_class(str(class_path))
        else:
            try:
                source_cls = type_map[source_type]
            except KeyError as exc:
                raise ValueError(f"Unsupported source type '{source_type}' for ref '{ref}'.") from exc

        source = source_cls(
            source_reference=ref,
            resource_location=location,
            iosource_method_kwargs=kwargs.get("iosource_method_kwargs", kwargs),
        )
        sources.register_source(source)

    return sources


def build_sinks_from_specs(specs: Iterable[Mapping[str, Any]]) -> IoSinks:
    """
    Build an :class:`IoSinks` registry from normalized sink specifications.

    Each spec must provide `ref`, `type`, and `location`. Optional `kwargs`
    are passed through as `iosink_method_kwargs`, except for `custom` sinks
    where `kwargs.class_path` selects the sink class.
    """

    type_map: dict[str, Any] = {
        "csv": CSVSink,
        "hdf": HDFProcessingSink,
        "hdf_processing": HDFProcessingSink,
    }
    sinks = IoSinks()
    for spec in specs:
        ref = str(spec["ref"]).strip()
        sink_type = str(spec["type"]).strip().lower()
        location = Path(str(spec["location"]))
        kwargs = dict(spec.get("kwargs", {}) or {})

        if sink_type == "custom":
            class_path = kwargs.pop("class_path", None)
            if not class_path:
                raise ValueError(f"Custom sink '{ref}' requires kwargs.class_path.")
            sink_cls = _load_custom_class(str(class_path))
        else:
            try:
                sink_cls = type_map[sink_type]
            except KeyError as exc:
                raise ValueError(f"Unsupported sink type '{sink_type}' for ref '{ref}'.") from exc

        sink = sink_cls(
            sink_reference=ref,
            resource_location=location,
            iosink_method_kwargs=kwargs.get("iosink_method_kwargs", kwargs),
        )
        sinks.register_sink(sink)

    return sinks


def write_processing_data_hdf(
    write_hdf: dict[str, Any] | None,
    *,
    run_name: str,
    result: Any,
    pipeline_yaml: str,
) -> str | None:
    """
    Persist ProcessingData to HDF5 using the shared runtime request payload.
    """

    if not write_hdf:
        return None

    out_path_raw = write_hdf.get("path")
    if not out_path_raw:
        raise ValueError("write_hdf.path is required when write_hdf is provided.")

    data_paths = list(write_hdf.get("data_paths", []) or [])
    write_all = bool(write_hdf.get("write_all_processing_data", False))
    if not data_paths and not write_all:
        raise ValueError("write_hdf requires data_paths or write_all_processing_data=true.")

    out_path = Path(str(out_path_raw))
    sink = HDFProcessingSink(resource_location=out_path)
    sink.write(
        run_name,
        result.processing_data,
        data_paths=data_paths or None,
        write_all_processing_data=write_all,
        pipeline_spec=result.pipeline.to_spec(),
        pipeline_yaml=pipeline_yaml,
        trace_events=result.tracer.events if result.tracer is not None else None,
    )
    return str(out_path)
