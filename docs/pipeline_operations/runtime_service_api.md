# Runtime Service API (Design Contract)

This document defines the proposed daemon/runtime interface for long-lived MoDaCor processing sessions.

Status: design contract with an active implementation refresh.

Companion draft OpenAPI schema:

- `docs/pipeline_operations/runtime_service_openapi.yaml`
- `docs/pipeline_operations/backlog.md` (upgrade and maintenance tracker)

## Current implementation status

The first structural refactor tranche between `U6` and `U8` is now in place:

- `src/modacor/server/api.py` is a thin FastAPI binding layer.
- `src/modacor/server/runtime_service.py` owns session orchestration and run lifecycle behavior.
- `src/modacor/server/planning.py` owns dry-run planning and dirty-step calculations.
- `src/modacor/server/io_utils.py` adapts session source/sink registrations into runtime IO objects.
- `src/modacor/server/errors.py` defines framework-neutral service exceptions.
- `src/modacor/io/runtime_support.py` provides shared source/sink builders and HDF export handling for both the CLI and runtime service.

This keeps the design contract below aligned with the codebase structure that is
now stable through the latest diagnostics work (`U10`), with persistence and
advanced streaming still pending.

## Goals

- Keep named MoDaCor instances alive (`PipelineSession`) and accept new files over time.
- Support partial rerun when only specific sources change (for example a new `sample` file).
- Provide deterministic fallback to full reset/rerun after partial-run failure.
- Support multiple concurrent sessions (for multiple instruments / detector streams).
- Expose an integration-friendly API for orchestration systems such as Bluesky.

## Core concepts

## `PipelineSession`

A session is an isolated runtime instance with:

- `session_id` (string, unique)
- `name` (human-readable)
- `pipeline_yaml` (canonical configuration text)
- `pipeline` (instantiated graph)
- `io_sources` / `io_sinks`
- `processing_data`
- `tracer` and last run metadata
- session lock (single active run per session)

Sessions are managed by a `SessionManager` in-process.

## Source registry model

Sources are dynamic and unbounded in count:

- `source_ref` (e.g. `sample`, `background`, `ical`, `ical_bg`, `defaults`)
- `type` (e.g. `hdf`, `yaml`, `csv`, future extensions)
- `resource_location` (path/URI)
- optional constructor kwargs

This preserves flexibility for pipelines with 2 to 6+ source files.

## Sink registry model

Sinks are dynamic output targets bound to stable refs:

- `sink_ref` (e.g. `export_csv`, `export_hdf`)
- `type` (e.g. `csv`, `hdf`, `hdf_processing`, `custom`)
- `resource_location` (path/URI)
- optional constructor kwargs

Pipeline steps can write through targets such as `export_csv::` while the
runtime API controls the concrete destination path. API-registered HDF sinks can
opt into available runtime metadata via sink kwargs, but metadata-rich final HDF
artifacts remain the responsibility of the process-level `write_hdf` shortcut.

## Reset modes

- `partial` reset: recompute only dirty subgraph.
- `full` reset: rebuild pipeline/session state and rerun from scratch.
- `auto` mode: try `partial`, fallback to `full` on failure (if enabled).

## State machine

Session states:

- `idle`
- `running_partial`
- `running_full`
- `error_partial`
- `error_full`

Transitions:

1. `idle -> running_partial` on partial process request.
2. `running_partial -> idle` on success.
3. `running_partial -> error_partial` on failure.
4. `error_partial -> running_full` on explicit full reset/process request, or automatic fallback.
5. `running_full -> idle` on success.
6. `running_full -> error_full` on failure.

## Partial invalidation contract

When sources change:

1. Compute seed steps affected by changed source refs.
2. Expand to descendants in DAG (dirty set).
3. Restore `processing_data` snapshot captured before earliest dirty step.
4. Call `reset()` on dirty steps.
5. Execute dirty subgraph in topological order.

Guarantee:

- No mixed old/new outputs in affected downstream chain.

Initial policy is conservative (descendant closure). Fine-grained optimizations can be added later.

## API schema (REST)

Base path: `/v1`

## Service status

### `GET /health`

Liveness probe for orchestration systems. This endpoint only answers the
question "is the process alive enough to respond?" and currently returns:

```json
{
  "status": "ok"
}
```

### `GET /readiness`

Readiness probe for runtime usability. This endpoint reports whether the
service can still accept work and includes high-level runtime metrics.

Example response:

```json
{
  "status": "degraded",
  "ready": true,
  "metrics": {
    "session_count": 3,
    "active_run_count": 1,
    "error_session_count": 1,
    "error_session_ids": ["sess-error"],
    "last_updated_utc": "2026-03-15T10:10:00+00:00"
  }
}
```

Current semantics:

- `ready: true` means the service can accept requests.
- `status: "degraded"` means one or more sessions are in an error state, even
  though the API itself remains usable.

## Sessions

### `GET /source-templates`

List built-in source templates/profiles (for example `mouse`, `saxsess`) with required/optional source refs.

### `POST /sessions`

Create a named session.

Request:

```json
{
  "session_id": "mouse-main",
  "name": "MOUSE production",
  "source_profile": "mouse",
  "pipeline": {
    "yaml_text": "name: ...",
    "yaml_path": "/opt/pipelines/MOUSE_solids.yaml"
  },
  "trace": {
    "enabled": true,
    "watch": {"sample": ["signal"]},
    "record_only_on_change": true
  },
  "auto_full_reset_on_partial_error": true
}
```

Rules:

- Exactly one of `yaml_text` or `yaml_path` must be provided.
- `session_id` must be unique.
- If `source_profile` is set, required source refs must be registered before `/process` is allowed.

### `GET /sessions`

List sessions and summary status.

### `GET /sessions/{session_id}`

Get full session metadata, current sources/sinks, state, and last run summary.

### `GET /sessions/{session_id}/errors/latest`

Return the current session error state plus the latest recorded failed-run
diagnostics.

Example response after a failed full run:

```json
{
  "session_id": "mouse-main",
  "state": "error_full",
  "active_run_id": null,
  "updated_utc": "2026-03-15T10:10:00+00:00",
  "current_error": {
    "code": "RUN_FAILED",
    "message": "synthetic failure",
    "details": {"exception_type": "RuntimeError"},
    "run_id": "run-abc123",
    "recorded_utc": "2026-03-15T10:10:00+00:00",
    "effective_mode": "full"
  },
  "latest_error": {
    "code": "RUN_FAILED",
    "message": "synthetic failure",
    "details": {"exception_type": "RuntimeError"},
    "run_id": "run-abc123",
    "recorded_utc": "2026-03-15T10:10:00+00:00",
    "effective_mode": "full"
  },
  "latest_failed_run": {
    "run_id": "run-abc123",
    "mode": "full",
    "effective_mode": "full",
    "status": "failed"
  }
}
```

Semantics:

- `current_error` reflects the session's current error state and becomes `null`
  after a successful recovery or full reset.
- `latest_error` is derived from the most recent failed run in history and
  remains available after recovery for post-mortem inspection.

### `DELETE /sessions/{session_id}`

Stop and remove session.

## Sources

### `PUT /sessions/{session_id}/sources`

Upsert one or more sources.

Request:

```json
{
  "sources": [
    {"ref": "sample", "type": "hdf", "location": "/data/new_sample.nxs"},
    {"ref": "background", "type": "hdf", "location": "/data/bg.nxs"},
    {"ref": "defaults", "type": "yaml", "location": "/data/defaults.yaml"}
  ]
}
```

Response includes accepted refs and current source map.

### `POST /sessions/{session_id}/sources/patch`

Convenience endpoint to upsert a single source.

Request:

```json
{
  "ref": "sample",
  "type": "hdf",
  "location": "/data/new_sample.nxs",
  "kwargs": {}
}
```

Equivalent to `PUT /sources` with a single-item `sources` list.

### `POST /sessions/{session_id}/sample`

Convenience shortcut for the common "new sample file arrived" workflow.
This endpoint updates source ref `sample` directly.

Request:

```json
{
  "location": "/data/new_sample.nxs",
  "type": "hdf",
  "kwargs": {}
}
```

### `DELETE /sessions/{session_id}/sources/{ref}`

Remove one source registration.

## Sinks

### `PUT /sessions/{session_id}/sinks`

Upsert one or more sinks.

Request:

```json
{
  "sinks": [
    {"ref": "export_csv", "type": "csv", "location": "/data/out/current.csv", "kwargs": {"delimiter": ","}},
    {"ref": "export_hdf", "type": "hdf", "location": "/data/out/current.h5"}
  ]
}
```

Response includes accepted refs and the current sink map.

### `POST /sessions/{session_id}/sinks/patch`

Convenience endpoint to upsert a single sink.

Request:

```json
{
  "ref": "export_csv",
  "type": "csv",
  "location": "/data/out/current.csv",
  "kwargs": {"delimiter": ","}
}
```

Equivalent to `PUT /sinks` with a single-item `sinks` list.

For HDF sinks, runtime metadata is opt-in:

```json
{
  "ref": "export_hdf",
  "type": "hdf",
  "location": "/data/out/current.h5",
  "kwargs": {
    "include_runtime_metadata": {
      "pipeline_yaml": true,
      "pipeline_spec": true,
      "trace_events": false
    }
  }
}
```

### `DELETE /sessions/{session_id}/sinks/{ref}`

Remove one sink registration.

## Processing

### `POST /sessions/{session_id}/process`

Trigger run.

Request:

```json
{
  "mode": "partial",
  "changed_sources": ["sample"],
  "changed_keys": ["sample.signal"],
  "run_name": "sample_2026_03_13_153045",
  "write_hdf": {
    "path": "/data/out/sample_2026_03_13_153045.h5",
    "write_all_processing_data": true,
    "data_paths": []
  }
}
```

`mode` enum:

- `partial`
- `full`
- `auto`

Notes:

- `changed_sources` or `changed_keys` is required for `partial`; both are optional for `auto`.
- `changed_keys` enables key-aware invalidation (e.g. `sample.signal`, `sample.Q`) for tighter partial reruns.
- `write_hdf` is optional; if provided, pipeline spec/yaml and trace are persisted.

### `POST /sessions/{session_id}/process/dry-run`

Preview what would run without executing any pipeline steps.

Request (same shape as `/process`, excluding `write_hdf` in typical usage):

```json
{
  "mode": "partial",
  "changed_sources": ["sample"],
  "changed_keys": ["sample.signal"]
}
```

Response includes:

- `effective_mode`
- `dirty_steps`
- `skipped_steps`
- `checkpoint_boundary_step`
- `missing_required_sources`
- `warnings`
- `can_process`

### `POST /sessions/{session_id}/reset`

Reset without immediate processing.

Request:

```json
{
  "mode": "full"
}
```

### `POST /sessions/{session_id}/recover`

Explicit recovery path after error.

Request:

```json
{
  "strategy": "full_reset_then_process",
  "changed_sources": ["sample"]
}
```

## Runs

### `GET /sessions/{session_id}/runs`

List run history metadata.

### `GET /sessions/{session_id}/runs/{run_id}`

Run details include:

- `dirty_steps`
- `skipped_steps`
- `step_durations_s`
- `elapsed_s`
- fallback metadata (`fallback_reason`, `recovered_from_run_id`) for auto recovery cases
- output artifact locations (e.g. `hdf_output`)

## WebSocket events

Endpoint: `/v1/sessions/{session_id}/events`

Event envelope:

```json
{
  "event": "step_finished",
  "session_id": "mouse-main",
  "run_id": "run-000123",
  "ts_utc": "2026-03-13T15:30:45.123Z",
  "payload": {}
}
```

Event types:

- `session_state_changed`
- `run_started`
- `run_finished`
- `run_failed`
- `step_started`
- `step_finished`
- `trace_event`
- `recovery_started`
- `recovery_finished`

## Error model

REST errors:

```json
{
  "error": {
    "code": "PARTIAL_RUN_FAILED",
    "message": "Step GX failed: ...",
    "details": {
      "session_id": "mouse-main",
      "run_id": "run-000123",
      "failed_step_id": "GX"
    }
  }
}
```

Recommended codes:

- `SESSION_NOT_FOUND`
- `SESSION_BUSY`
- `INVALID_REQUEST`
- `SOURCE_NOT_FOUND`
- `PIPELINE_LOAD_FAILED`
- `PARTIAL_RUN_FAILED`
- `FULL_RUN_FAILED`
- `RECOVERY_FAILED`

## Concurrency and consistency contracts

- One active run per session (guarded by lock).
- Multiple sessions can run in parallel.
- Each run receives immutable run metadata (`run_id`, `run_name`, timestamps).
- Writes to HDF artifacts are per-run file path or per-run group to avoid collisions.

## Bluesky integration notes

- Bluesky callback/plan can call:
  1. `PUT /sources` with new file path(s)
  2. `POST /process` with `mode=partial`, `changed_sources=["sample"]`
- Live monitoring consumes WebSocket events.
- Keep MoDaCor runtime API transport-agnostic internally so future Kafka/Redis event buses are optional.

## Implementation phases

1. `MVP service`
- Session manager
- Create/list/delete session
- Source upsert
- `process` with full rerun only
- WebSocket run start/finish

2. `Partial invalidation`
- Dirty subgraph computation
- Snapshot + partial reset/rerun
- run metadata for dirty set

3. `Recovery and hardening`
- auto full-reset fallback
- explicit `/recover`
- richer errors + metrics

4. `Production polish`
- authn/authz
- persistence for session definitions
- rate limits, structured logs, OpenTelemetry

## Scaffold status in this repository

A first API scaffold is available under:

- `src/modacor/server/session_manager.py`
- `src/modacor/server/api.py`

It provides route skeletons and an in-memory session manager aligned with this contract.
`/process` is now wired to execute MoDaCor runs with registered sources,
registered sinks, and optional HDF output writing.
The scaffold now includes dirty-step detection by changed source references and executes selected subgraphs for
`partial` mode when prior `ProcessingData` exists. `auto` mode attempts partial first and falls back to full rerun on
partial failure.
When partial mode runs, the service records a boundary checkpoint before the first dirty step and restores it if
partial execution fails.
The scaffold also includes the `U8` health/readiness split for operational
probes and basic runtime metrics, plus the `U10` latest-error diagnostics
endpoint for post-failure inspection.

Run the scaffold service:

```bash
pip install "modacor[server]"
modacor serve --host 127.0.0.1 --port 8000
```

Optional convenience wrapper for API usage:

```bash
modacor session --url http://127.0.0.1:8000 list
```

## Quick use example

The example below shows a complete session lifecycle with a MOUSE-style pipeline.

1. Create a session from a pipeline file:

```bash
curl -X POST "http://127.0.0.1:8000/v1/sessions" \
  -H "content-type: application/json" \
  -d '{
    "session_id": "mouse-main",
    "name": "MOUSE production",
    "pipeline": {
      "yaml_path": "/Users/bpauw/Documents/BAM/Projects/2025/MOUSE_MoDaCor/processing_pipelines/MOUSE_solids.yaml"
    },
    "trace": {
      "enabled": true,
      "watch": {"sample": ["signal"], "background": ["signal"]},
      "record_only_on_change": true
    },
    "auto_full_reset_on_partial_error": true
  }'
```

2. Register/update sources (repeat this step whenever a new sample file arrives):

```bash
curl -X PUT "http://127.0.0.1:8000/v1/sessions/mouse-main/sources" \
  -H "content-type: application/json" \
  -d '{
    "sources": [
      {"ref": "sample", "type": "hdf", "location": "/data/MOUSE_sample_latest.nxs"},
      {"ref": "background", "type": "hdf", "location": "/data/MOUSE_background.nxs"}
    ]
  }'
```

3. Register/update sinks used by sink-aware pipeline steps:

```bash
curl -X PUT "http://127.0.0.1:8000/v1/sessions/mouse-main/sinks" \
  -H "content-type: application/json" \
  -d '{
    "sinks": [
      {"ref": "export_csv", "type": "csv", "location": "/tmp/mouse_latest.csv", "kwargs": {"delimiter": ","}}
    ]
  }'
```

4. Trigger a partial run using changed keys, and write a full HDF artifact:

```bash
curl -X POST "http://127.0.0.1:8000/v1/sessions/mouse-main/process" \
  -H "content-type: application/json" \
  -d '{
    "mode": "partial",
    "changed_sources": ["sample"],
    "changed_keys": ["sample.signal"],
    "run_name": "mouse_run_2026_03_13T1530",
    "write_hdf": {
      "path": "/tmp/mouse_run_2026_03_13T1530.h5",
      "write_all_processing_data": true
    }
  }'
```

5. Use auto mode to attempt partial first and fallback to full on failure:

```bash
curl -X POST "http://127.0.0.1:8000/v1/sessions/mouse-main/process" \
  -H "content-type: application/json" \
  -d '{
    "mode": "auto",
    "changed_sources": ["sample"],
    "changed_keys": ["sample.signal"]
  }'
```

6. If needed, force a complete reset without processing:

```bash
curl -X POST "http://127.0.0.1:8000/v1/sessions/mouse-main/reset" \
  -H "content-type: application/json" \
  -d '{"mode":"full"}'
```
