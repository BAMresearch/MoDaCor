# SPDX-License-Identifier: BSD-3-Clause
# /usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Any

from modacor.io.io_sources import IoSources
from modacor.io.runtime_support import build_sources_from_specs, write_processing_data_hdf

from .session_manager import PipelineSession

__all__ = ["build_sources_from_session", "write_hdf_output"]


def build_sources_from_session(session: PipelineSession) -> IoSources:
    specs: list[dict[str, Any]] = []
    for ref in sorted(session.sources.keys()):
        reg = session.sources[ref]
        specs.append(
            {
                "ref": ref,
                "type": reg["type"],
                "location": reg["location"],
                "kwargs": dict(reg.get("kwargs", {}) or {}),
            }
        )
    return build_sources_from_specs(specs)


def write_hdf_output(
    write_hdf: dict[str, Any] | None,
    *,
    run_name: str,
    result: Any,
    pipeline_yaml: str,
) -> str | None:
    return write_processing_data_hdf(
        write_hdf,
        run_name=run_name,
        result=result,
        pipeline_yaml=pipeline_yaml,
    )
