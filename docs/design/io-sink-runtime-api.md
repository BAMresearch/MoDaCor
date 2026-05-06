# Runtime API IoSink Configuration

Design note for runtime-service `IoSink` configuration with the same
registration model already used for `IoSource`.

Implementation status: implemented in this branch.

## Pre-Implementation State

The runtime API stores source registrations on each session and rebuilds an
`IoSources` registry when processing starts. The flow is:

- `PipelineSession.sources` stores normalized registration dictionaries.
- `RuntimeService.upsert_sources(...)` and `patch_source(...)` expose the HTTP
  behavior through FastAPI routes.
- `build_sources_from_session(...)` adapts session state to
  `build_sources_from_specs(...)`.
- `run_pipeline_job(...)` receives the resulting `IoSources` registry.

The sink infrastructure already exists but is only partially connected to the
runtime API:

- `IoSinks` and `IoSink` mirror the source registry and base class contracts.
- `run_pipeline_job(...)` already accepts an optional `sinks` registry and
  assigns it to every process step.
- `build_sinks_from_specs(...)` exists, but currently supports only CSV sinks.
- The runtime service never stores session sinks, never builds an `IoSinks`
  registry from session state, and never passes sinks into `run_pipeline_job`.
- Process-level `write_hdf` is a separate artifact export shortcut using
  `HDFProcessingSink`; it is not currently a registered session sink.

## Goal

Allow clients to register, inspect, update, and delete output sinks through the
runtime API in the same shape as source registration:

```json
{
  "sinks": [
    {
      "ref": "export_csv",
      "type": "csv",
      "location": "/data/out/current.csv",
      "kwargs": {"delimiter": ","}
    }
  ]
}
```

Pipelines can then use `SinkProcessingData` or other sink-aware steps with
targets such as `export_csv::` while the concrete file path remains
runtime-configurable.

## API

The implementation adds these endpoints under the existing
`/v1/sessions/{session_id}` resource:

- `PUT /sinks`: upsert one or more sink registrations.
- `POST /sinks/patch`: convenience upsert for one sink registration.
- `DELETE /sinks/{ref}`: remove one sink registration.

Session detail responses should include both:

- `sources`: current source registrations.
- `sinks`: current sink registrations.

Use the same normalized registration shape for sinks as for sources:

- `ref`: stable runtime reference used before `::`.
- `type`: built-in type or `custom`.
- `location`: resource location passed to the sink constructor.
- `kwargs`: optional sink-specific keyword configuration.

Built-in sink types:

- `csv` -> `modacor.io.csv.csv_sink.CSVSink`
- `hdf` -> `modacor.io.hdf.hdf_processing_sink.HDFProcessingSink`
- `hdf_processing` -> `HDFProcessingSink` alias for clarity
- `custom` -> class selected by `kwargs.class_path`

For custom sinks, mirror custom source behavior: `kwargs.class_path` is consumed
as the fully qualified import path, and the remaining kwargs are forwarded as
sink method kwargs.

## Runtime Behavior

At process start, build sources and sinks from the current session snapshot:

1. Parse the pipeline YAML.
2. Build `IoSources` from `session.sources`.
3. Build `IoSinks` from `session.sinks`.
4. Call `run_pipeline_job(..., sources=sources, sinks=sinks, ...)`.

The auto-fallback path must reuse the same built sink registry when a failed
partial run falls back to a full run.

`write_hdf` should remain supported for now. It is a process-level artifact
shortcut that adds pipeline YAML/spec and trace metadata. Registered HDF sinks
serve pipeline steps that explicitly write through `IoSinks`; they should not
replace `write_hdf` in this enhancement.

API-registered HDF sinks may optionally request runtime metadata, but metadata
inclusion must not be the default. The process-level `write_hdf` path remains
the main artifact export path and continues to include pipeline and trace
metadata by default when requested. Registered HDF sinks should only receive
pipeline/trace metadata when their sink registration explicitly opts in through
sink-specific kwargs. Because in-pipeline sink writes happen before the complete
run result exists, complete trace metadata is naturally better served by
`write_hdf`; any HDF-sink metadata opt-in should document exactly which metadata
is available at the time of the write.

If an API-registered sink has the same ref as a pipeline `AppendSink` step, the
existing `AppendSink` behavior means the pre-registered sink wins because
`AppendSink` skips refs that are already present. This mirrors the current
`AppendSource` pattern and keeps runtime-provided I/O definitions authoritative.

Sink writes are external side effects. A failed partial run can restore
`ProcessingData`, but it cannot roll back files already written by a sink step.
This is already true for pipeline-defined sinks; the API feature should document
the same non-transactional behavior rather than hiding it.

## Implementation Summary

1. Extend session state.

Add `PipelineSession.sinks` alongside `sources`. Add `SessionManager` methods
for upsert/delete sink registrations. Prefer a small shared registration
normalization helper so source and sink handling stay aligned and malformed
payload errors are consistently reported.

2. Extend runtime I/O helpers.

Add `build_sinks_from_session(session)` to `src/modacor/server/io_utils.py` and
use existing `modacor.io.runtime_support.build_sinks_from_specs(...)`.

3. Make sink specs as capable as source specs.

Update `build_sinks_from_specs(...)` to use a type map and support `custom`
with `kwargs.class_path`. Forward `kwargs.get("iosink_method_kwargs", kwargs)`
to mirror source builder behavior. For HDF sinks, keep any metadata inclusion
explicit via kwargs; do not silently add pipeline or trace metadata.

4. Wire runtime execution.

Update `RuntimeService.process(...)`, `_execute_process_run(...)`,
`_handle_process_failure(...)`, and `_run_auto_fallback(...)` so both the
normal execution path and auto fallback receive the configured `IoSinks`.

5. Add API methods and routes.

Add service methods matching the source methods: `upsert_sinks`,
`patch_sink`, and `delete_sink`. Add FastAPI routes under
`src/modacor/server/api.py`.

6. Update CLI wrappers.

Add `modacor session set-sink` and `modacor session delete-sink` for parity
with `set-source` and `delete-source`. Keep `modacor run --csv-sink` unchanged.

7. Update public docs.

Updated public docs:

- `docs/pipeline_operations/runtime_service_api.md`
- `docs/pipeline_operations/runtime_service_openapi.yaml`
- `docs/pipeline_operations/pipeline_basics.md`
- `docs/extending/io_source_sink_guide.md`
- `docs/getting_started/cli_and_runner.md`

## Test Coverage

Focused unit tests cover:

- `build_sinks_from_specs(...)` builds CSV and HDF sinks.
- `build_sinks_from_specs(...)` supports `custom` sinks with
  `kwargs.class_path`.
- unsupported sink types fail with a clear `ValueError`.
- nested `iosink_method_kwargs` and flat kwargs both work.
- `SessionManager` can upsert/delete sinks without disturbing sources.

Service-level tests cover:

- session detail includes `sinks`.
- `upsert_sinks`, `patch_sink`, and `delete_sink` mirror source behavior.
- `RuntimeService.process(...)` passes configured sinks to `run_pipeline_job`.
- auto fallback also receives configured sinks.
- a seeded partial run with `SinkProcessingData` writes to an API-registered
  CSV sink.

CLI tests cover:

- `modacor session set-sink` sends `PUT /v1/sessions/{id}/sinks`.
- `modacor session delete-sink` sends `DELETE /v1/sessions/{id}/sinks/{ref}`.

Verification commands:

- targeted runtime/sink tests first,
- server API tests with FastAPI available,
- full project tests if the environment has the required dependencies.

Targeted check on 2026-05-06:

```text
./.venv-dev/bin/python -m pytest \
  tests/io/test_runtime_support.py \
  tests/server/test_io_utils.py \
  tests/server/test_api_e2e.py \
  tests/server/test_session_manager.py \
  tests/modules/base_modules/test_append_sink.py \
  tests/modules/base_modules/test_sink_processing_data.py \
  tests/io/hdf/test_hdf_processing_sink.py \
  tests/test_cli.py
```

Result: 42 passed.

Full check on 2026-05-06:

```text
./.venv-dev/bin/python -m pytest
```

Result: 425 passed, with 3 existing numerical runtime warnings.

## Resolved Decisions

1. Should the first implementation include the CLI session wrappers?

Decision: yes. They are small and keep session source/sink ergonomics
consistent.

2. Should API-registered HDF sinks include trace and pipeline metadata
automatically?

Decision: no by default. API-registered HDF sinks may optionally include
pipeline/trace metadata through explicit sink configuration, but the main
`write_hdf` path remains responsible for the default metadata-rich HDF artifact.

3. Should sink changes participate in partial-run invalidation?

Decision: not in the first implementation. Sink-only re-export is not a main
use case, and full invalidation is acceptable for now. A dedicated
`changed_sinks` field can be designed later if sink-only re-export becomes an
operational requirement.
