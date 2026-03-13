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

from pathlib import Path
from typing import Any

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
        data_paths: list[str],
        *,
        pipeline_spec: dict[str, Any] | None = None,
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
            run_pipeline_group = _recreate_group(pipeline_group, run_name)
            if resolved_pipeline_spec is not None:
                json_bytes = str(self._as_json_string(resolved_pipeline_spec)).encode("utf-8")
                run_pipeline_group.create_dataset("spec", data=json_bytes)
            else:
                run_pipeline_group.attrs["empty"] = True

            # Trace events (stored as JSON string)
            tracer_group = processing_group.require_group("tracer")
            run_tracer_group = _recreate_group(tracer_group, run_name)
            if resolved_trace_events is not None:
                json_bytes = str(self._as_json_string(resolved_trace_events)).encode("utf-8")
                run_tracer_group.create_dataset("events", data=json_bytes)
            else:
                run_tracer_group.attrs["empty"] = True

        self.logger.info(f"Wrote processing results to {out_path} (run={run_name}).")
        return out_path

    @staticmethod
    def _as_json_string(data: Any) -> str:
        import json

        return json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False)
