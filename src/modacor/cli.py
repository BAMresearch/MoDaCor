# SPDX-License-Identifier: BSD-3-Clause
# /usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from pathlib import Path

from modacor.debug.pipeline_tracer import PlainUnicodeRenderer
from modacor.io.csv.csv_sink import CSVSink
from modacor.io.hdf.hdf_processing_sink import HDFProcessingSink
from modacor.io.hdf.hdf_source import HDFSource
from modacor.io.io_sinks import IoSinks
from modacor.io.io_sources import IoSources
from modacor.io.yaml.yaml_source import YAMLSource
from modacor.runner import run_pipeline_job

__all__ = ["main"]


def _parse_ref_path(spec: str) -> tuple[str, Path]:
    ref, sep, location = spec.partition("=")
    if sep == "" or not ref.strip() or not location.strip():
        raise argparse.ArgumentTypeError(f"Invalid mapping {spec!r}. Use the form 'reference=/path/to/resource'.")
    return ref.strip(), Path(location.strip())


def _parse_trace_watch(entries: list[str] | None) -> dict[str, list[str]]:
    watch: dict[str, list[str]] = {}
    if not entries:
        return watch

    for item in entries:
        bundle, sep, keys_raw = item.partition(":")
        if sep == "" or not bundle.strip() or not keys_raw.strip():
            raise ValueError(f"Invalid --trace-watch value {item!r}. Use 'bundle:key[,key...]'.")
        keys = [key.strip() for key in keys_raw.split(",") if key.strip()]
        if not keys:
            raise ValueError(f"Invalid --trace-watch value {item!r}: no keys supplied.")
        watch.setdefault(bundle.strip(), []).extend(keys)
    return watch


def _build_sources(
    hdf_sources: list[tuple[str, Path]] | None,
    yaml_sources: list[tuple[str, Path]] | None,
) -> IoSources:
    sources = IoSources()
    for source_reference, resource_location in hdf_sources or []:
        sources.register_source(HDFSource(source_reference=source_reference, resource_location=resource_location))
    for source_reference, resource_location in yaml_sources or []:
        sources.register_source(YAMLSource(source_reference=source_reference, resource_location=resource_location))
    return sources


def _build_sinks(csv_sinks: list[tuple[str, Path]] | None) -> IoSinks:
    sinks = IoSinks()
    for sink_reference, resource_location in csv_sinks or []:
        sinks.register_sink(CSVSink(sink_reference=sink_reference, resource_location=resource_location))
    return sinks


def _add_run_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    run_parser = subparsers.add_parser("run", help="Run a MoDaCor pipeline configuration.")
    run_parser.add_argument("--pipeline", required=True, type=Path, help="Path to a pipeline YAML file.")
    run_parser.add_argument(
        "--hdf-source",
        action="append",
        type=_parse_ref_path,
        default=[],
        metavar="REF=PATH",
        help="Register an HDF source (repeatable).",
    )
    run_parser.add_argument(
        "--yaml-source",
        action="append",
        type=_parse_ref_path,
        default=[],
        metavar="REF=PATH",
        help="Register a YAML source (repeatable).",
    )
    run_parser.add_argument(
        "--csv-sink",
        action="append",
        type=_parse_ref_path,
        default=[],
        metavar="REF=PATH",
        help="Register a CSV sink (repeatable).",
    )
    run_parser.add_argument("--trace", action="store_true", help="Enable tracing and attach trace events.")
    run_parser.add_argument(
        "--trace-watch",
        action="append",
        default=[],
        metavar="BUNDLE:KEY[,KEY...]",
        help="Tracer watch spec (repeatable), for example 'sample:signal,Q'.",
    )
    run_parser.add_argument(
        "--stop-after",
        default=None,
        metavar="STEP_ID",
        help="Stop execution after this step id (inclusive).",
    )
    run_parser.add_argument(
        "--write-hdf",
        default=None,
        type=Path,
        metavar="PATH",
        help="Write selected ProcessingData paths to an HDF5 output file.",
    )
    run_parser.add_argument(
        "--write-path",
        action="append",
        default=[],
        metavar="PROCESSING_PATH",
        help="ProcessingData path to export to HDF (repeatable).",
    )
    run_parser.add_argument(
        "--write-all-processing-data",
        action="store_true",
        help="Export all BaseData entries currently present in ProcessingData to HDF.",
    )
    run_parser.add_argument(
        "--run-name",
        default="default",
        help="Run label written under /processing/<...>/<run-name> in the HDF output.",
    )
    run_parser.add_argument(
        "--trace-report-lines",
        default=0,
        type=int,
        metavar="N",
        help="If tracing is enabled, print the last N trace lines (0 disables report output).",
    )


def _run_command(args: argparse.Namespace) -> int:
    trace_watch = _parse_trace_watch(args.trace_watch)
    sources = _build_sources(hdf_sources=args.hdf_source, yaml_sources=args.yaml_source)
    sinks = _build_sinks(csv_sinks=args.csv_sink)

    result = run_pipeline_job(
        args.pipeline,
        sources=sources,
        sinks=sinks,
        trace=args.trace,
        trace_watch=trace_watch,
        stop_after=args.stop_after,
    )

    if args.write_hdf is not None:
        if not args.write_path and not args.write_all_processing_data:
            raise ValueError("--write-hdf requires --write-all-processing-data or at least one --write-path.")
        hdf_sink = HDFProcessingSink(resource_location=args.write_hdf)
        hdf_sink.write(
            args.run_name,
            result.processing_data,
            data_paths=args.write_path or None,
            write_all_processing_data=args.write_all_processing_data,
            pipeline_spec=result.pipeline.to_spec(),
            pipeline_yaml=result.pipeline.to_yaml(),
            trace_events=result.tracer.events if result.tracer is not None else None,
        )
        print(f"Wrote HDF output: {args.write_hdf}")

    print(f"Executed {len(result.executed_steps)} step(s).")
    if result.stopped_after_step is not None:
        print(f"Stopped after step: {result.stopped_after_step}")

    if args.trace and args.trace_report_lines > 0 and result.tracer is not None:
        print(result.tracer.last_report(args.trace_report_lines, renderer=PlainUnicodeRenderer()))

    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="modacor", description="MoDaCor command-line interface.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_run_parser(subparsers)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "run":
        return _run_command(args)

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
