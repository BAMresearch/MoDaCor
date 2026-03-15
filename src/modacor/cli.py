# SPDX-License-Identifier: BSD-3-Clause
# /usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib import error, request

from modacor.debug.pipeline_tracer import PlainUnicodeRenderer
from modacor.io.io_sinks import IoSinks
from modacor.io.io_sources import IoSources
from modacor.io.runtime_support import build_sinks_from_specs, build_sources_from_specs, write_processing_data_hdf
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


def _http_request_json(base_url: str, method: str, path: str, payload: dict | None = None) -> dict | list | None:
    url = base_url.rstrip("/") + path
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    req = request.Request(url, method=method.upper(), data=data, headers=headers)
    try:
        with request.urlopen(req) as resp:  # noqa: S310
            raw = resp.read()
            if not raw:
                return None
            return json.loads(raw.decode("utf-8"))
    except error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} {method.upper()} {path}: {body}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"Request failed for {method.upper()} {path}: {exc.reason}") from exc


def _build_sources(
    hdf_sources: list[tuple[str, Path]] | None,
    yaml_sources: list[tuple[str, Path]] | None,
) -> IoSources:
    specs = [{"ref": ref, "type": "hdf", "location": path} for ref, path in hdf_sources or []]
    specs.extend({"ref": ref, "type": "yaml", "location": path} for ref, path in yaml_sources or [])
    return build_sources_from_specs(specs)


def _build_sinks(csv_sinks: list[tuple[str, Path]] | None) -> IoSinks:
    specs = [{"ref": ref, "type": "csv", "location": path} for ref, path in csv_sinks or []]
    return build_sinks_from_specs(specs)


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


def _add_serve_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    serve_parser = subparsers.add_parser("serve", help="Start the MoDaCor runtime API service.")
    serve_parser.add_argument("--host", default="127.0.0.1", help="Bind host for the HTTP server.")
    serve_parser.add_argument("--port", default=8000, type=int, help="Bind port for the HTTP server.")


def _add_session_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    session_parser = subparsers.add_parser("session", help="Manage runtime service sessions.")
    session_parser.add_argument(
        "--url",
        default="http://127.0.0.1:8000",
        help="Runtime service base URL.",
    )
    session_subparsers = session_parser.add_subparsers(dest="session_command", required=True)

    session_subparsers.add_parser("list", help="List sessions.")

    create_parser = session_subparsers.add_parser("create", help="Create a session.")
    create_parser.add_argument("--session-id", required=True)
    create_parser.add_argument("--name", default=None)
    create_parser.add_argument(
        "--source-template",
        default=None,
        help="Optional source template/profile name (for example 'mouse' or 'saxsess').",
    )
    create_group = create_parser.add_mutually_exclusive_group(required=True)
    create_group.add_argument("--pipeline-yaml-path", type=Path)
    create_group.add_argument("--pipeline-yaml-text")
    create_parser.add_argument("--trace", action="store_true")
    create_parser.add_argument(
        "--trace-watch",
        action="append",
        default=[],
        metavar="BUNDLE:KEY[,KEY...]",
        help="Tracer watch spec (repeatable).",
    )
    create_parser.add_argument(
        "--no-auto-full-reset-on-partial-error",
        action="store_true",
        help="Disable automatic full reset fallback after partial failures.",
    )

    delete_parser = session_subparsers.add_parser("delete", help="Delete a session.")
    delete_parser.add_argument("--session-id", required=True)

    status_parser = session_subparsers.add_parser("status", help="Get session details.")
    status_parser.add_argument("--session-id", required=True)

    last_error_parser = session_subparsers.add_parser("last-error", help="Get latest error diagnostics.")
    last_error_parser.add_argument("--session-id", required=True)

    set_source_parser = session_subparsers.add_parser("set-source", help="Upsert one source registration.")
    set_source_parser.add_argument("--session-id", required=True)
    set_source_parser.add_argument("--ref", required=True)
    set_source_parser.add_argument("--type", required=True, dest="source_type")
    set_source_parser.add_argument("--location", required=True, type=Path)
    set_source_parser.add_argument(
        "--kwargs-json",
        default="{}",
        help="JSON object with extra source kwargs (default '{}').",
    )

    set_sample_parser = session_subparsers.add_parser(
        "set-sample",
        help="Shortcut to upsert the 'sample' source reference.",
    )
    set_sample_parser.add_argument("--session-id", required=True)
    set_sample_parser.add_argument("--location", required=True, type=Path)
    set_sample_parser.add_argument("--type", default="hdf", dest="source_type")
    set_sample_parser.add_argument(
        "--kwargs-json",
        default="{}",
        help="JSON object with extra source kwargs (default '{}').",
    )

    del_source_parser = session_subparsers.add_parser("delete-source", help="Delete one source registration.")
    del_source_parser.add_argument("--session-id", required=True)
    del_source_parser.add_argument("--ref", required=True)

    process_parser = session_subparsers.add_parser("process", help="Trigger processing.")
    process_parser.add_argument("--session-id", required=True)
    process_parser.add_argument("--mode", required=True, choices=["partial", "full", "auto"])
    process_parser.add_argument("--changed-source", action="append", default=[], dest="changed_sources")
    process_parser.add_argument("--changed-key", action="append", default=[], dest="changed_keys")
    process_parser.add_argument("--run-name", default=None)
    process_parser.add_argument("--write-hdf-path", default=None, type=Path)
    process_parser.add_argument("--write-all-processing-data", action="store_true")
    process_parser.add_argument("--write-path", action="append", default=[])

    dry_run_parser = session_subparsers.add_parser("dry-run", help="Preview dirty-step plan without execution.")
    dry_run_parser.add_argument("--session-id", required=True)
    dry_run_parser.add_argument("--mode", required=True, choices=["partial", "full", "auto"])
    dry_run_parser.add_argument("--changed-source", action="append", default=[], dest="changed_sources")
    dry_run_parser.add_argument("--changed-key", action="append", default=[], dest="changed_keys")

    reset_parser = session_subparsers.add_parser("reset", help="Reset session runtime state.")
    reset_parser.add_argument("--session-id", required=True)
    reset_parser.add_argument("--mode", required=True, choices=["partial", "full"])

    runs_parser = session_subparsers.add_parser("runs", help="List runs or fetch one run.")
    runs_parser.add_argument("--session-id", required=True)
    runs_parser.add_argument("--run-id", default=None)

    session_subparsers.add_parser("templates", help="List available source templates/profiles.")


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
        write_processing_data_hdf(
            {
                "path": str(args.write_hdf),
                "data_paths": list(args.write_path),
                "write_all_processing_data": bool(args.write_all_processing_data),
            },
            run_name=args.run_name,
            result=result,
            pipeline_yaml=result.pipeline.to_yaml(),
        )
        print(f"Wrote HDF output: {args.write_hdf}")

    print(f"Executed {len(result.executed_steps)} step(s).")
    if result.stopped_after_step is not None:
        print(f"Stopped after step: {result.stopped_after_step}")

    if args.trace and args.trace_report_lines > 0 and result.tracer is not None:
        print(result.tracer.last_report(args.trace_report_lines, renderer=PlainUnicodeRenderer()))

    return 0


def _serve_command(args: argparse.Namespace) -> int:
    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError("uvicorn is not installed. Install server extras: pip install modacor[server].") from exc

    from modacor.server.api import create_app

    app = create_app()
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


def _print_json(data: object) -> None:
    print(json.dumps(data, indent=2, ensure_ascii=False))


def _session_list(base_url: str, args: argparse.Namespace) -> int:  # noqa: ARG001
    _print_json(_http_request_json(base_url, "GET", "/v1/sessions"))
    return 0


def _session_create(base_url: str, args: argparse.Namespace) -> int:
    pipeline_payload: dict[str, str]
    if args.pipeline_yaml_path is not None:
        pipeline_payload = {"yaml_path": str(args.pipeline_yaml_path)}
    else:
        pipeline_payload = {"yaml_text": str(args.pipeline_yaml_text)}

    payload = {
        "session_id": args.session_id,
        "name": args.name,
        "pipeline": pipeline_payload,
        "trace": {
            "enabled": bool(args.trace),
            "watch": _parse_trace_watch(args.trace_watch),
            "record_only_on_change": True,
        },
        "auto_full_reset_on_partial_error": not bool(args.no_auto_full_reset_on_partial_error),
    }
    if args.source_template is not None:
        payload["source_profile"] = str(args.source_template)
    _print_json(_http_request_json(base_url, "POST", "/v1/sessions", payload))
    return 0


def _session_delete(base_url: str, args: argparse.Namespace) -> int:
    _http_request_json(base_url, "DELETE", f"/v1/sessions/{args.session_id}")
    print(f"Deleted session: {args.session_id}")
    return 0


def _session_status(base_url: str, args: argparse.Namespace) -> int:
    _print_json(_http_request_json(base_url, "GET", f"/v1/sessions/{args.session_id}"))
    return 0


def _session_last_error(base_url: str, args: argparse.Namespace) -> int:
    _print_json(_http_request_json(base_url, "GET", f"/v1/sessions/{args.session_id}/errors/latest"))
    return 0


def _session_set_source(base_url: str, args: argparse.Namespace) -> int:
    try:
        kwargs_obj = json.loads(args.kwargs_json)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid --kwargs-json payload: {exc}") from exc
    if not isinstance(kwargs_obj, dict):
        raise ValueError("--kwargs-json must decode to an object/dict.")

    payload = {
        "sources": [
            {
                "ref": args.ref,
                "type": args.source_type,
                "location": str(args.location),
                "kwargs": kwargs_obj,
            }
        ]
    }
    _print_json(_http_request_json(base_url, "PUT", f"/v1/sessions/{args.session_id}/sources", payload))
    return 0


def _session_set_sample(base_url: str, args: argparse.Namespace) -> int:
    try:
        kwargs_obj = json.loads(args.kwargs_json)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid --kwargs-json payload: {exc}") from exc
    if not isinstance(kwargs_obj, dict):
        raise ValueError("--kwargs-json must decode to an object/dict.")

    payload = {
        "location": str(args.location),
        "type": args.source_type,
        "kwargs": kwargs_obj,
    }
    _print_json(_http_request_json(base_url, "POST", f"/v1/sessions/{args.session_id}/sample", payload))
    return 0


def _session_delete_source(base_url: str, args: argparse.Namespace) -> int:
    _http_request_json(base_url, "DELETE", f"/v1/sessions/{args.session_id}/sources/{args.ref}")
    print(f"Deleted source '{args.ref}' from session '{args.session_id}'.")
    return 0


def _session_process(base_url: str, args: argparse.Namespace) -> int:
    payload: dict[str, object] = {"mode": args.mode}
    if args.changed_sources:
        payload["changed_sources"] = args.changed_sources
    if args.changed_keys:
        payload["changed_keys"] = args.changed_keys
    if args.run_name is not None:
        payload["run_name"] = args.run_name

    if args.write_hdf_path is not None:
        write_hdf: dict[str, object] = {
            "path": str(args.write_hdf_path),
            "write_all_processing_data": bool(args.write_all_processing_data),
        }
        if args.write_path:
            write_hdf["data_paths"] = list(args.write_path)
        payload["write_hdf"] = write_hdf

    _print_json(_http_request_json(base_url, "POST", f"/v1/sessions/{args.session_id}/process", payload))
    return 0


def _session_dry_run(base_url: str, args: argparse.Namespace) -> int:
    payload: dict[str, object] = {"mode": args.mode}
    if args.changed_sources:
        payload["changed_sources"] = args.changed_sources
    if args.changed_keys:
        payload["changed_keys"] = args.changed_keys
    _print_json(_http_request_json(base_url, "POST", f"/v1/sessions/{args.session_id}/process/dry-run", payload))
    return 0


def _session_reset(base_url: str, args: argparse.Namespace) -> int:
    payload = {"mode": args.mode}
    _print_json(_http_request_json(base_url, "POST", f"/v1/sessions/{args.session_id}/reset", payload))
    return 0


def _session_runs(base_url: str, args: argparse.Namespace) -> int:
    path = f"/v1/sessions/{args.session_id}/runs"
    if args.run_id is not None:
        path = f"{path}/{args.run_id}"
    _print_json(_http_request_json(base_url, "GET", path))
    return 0


def _session_templates(base_url: str, args: argparse.Namespace) -> int:  # noqa: ARG001
    _print_json(_http_request_json(base_url, "GET", "/v1/source-templates"))
    return 0


def _session_command(args: argparse.Namespace) -> int:
    handlers = {
        "list": _session_list,
        "create": _session_create,
        "delete": _session_delete,
        "status": _session_status,
        "last-error": _session_last_error,
        "set-source": _session_set_source,
        "set-sample": _session_set_sample,
        "delete-source": _session_delete_source,
        "process": _session_process,
        "dry-run": _session_dry_run,
        "reset": _session_reset,
        "runs": _session_runs,
        "templates": _session_templates,
    }
    try:
        handler = handlers[args.session_command]
    except KeyError as exc:
        raise ValueError(f"Unknown session command: {args.session_command}") from exc
    return handler(args.url, args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="modacor", description="MoDaCor command-line interface.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_run_parser(subparsers)
    _add_serve_parser(subparsers)
    _add_session_parser(subparsers)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "run":
        return _run_command(args)
    if args.command == "serve":
        return _serve_command(args)
    if args.command == "session":
        return _session_command(args)

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
