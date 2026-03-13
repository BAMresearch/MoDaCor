# Runtime Service API (Design Contract)

This document defines the proposed daemon/runtime interface for long-lived MoDaCor processing sessions.

Status: design contract (target architecture), not final implementation.

Companion draft OpenAPI schema:

- `docs/pipeline_operations/runtime_service_openapi.yaml`

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

## Sessions

### `POST /sessions`

Create a named session.

Request:

```json
{
  "session_id": "mouse-main",
  "name": "MOUSE production",
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

### `GET /sessions`

List sessions and summary status.

### `GET /sessions/{session_id}`

Get full session metadata, current sources, state, and last run summary.

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

### `DELETE /sessions/{session_id}/sources/{ref}`

Remove one source registration.

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

Run details: timings, dirty set, status, output artifact locations.

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
- rate limits, health probes, structured logs, OpenTelemetry

## Scaffold status in this repository

A first API scaffold is available under:

- `src/modacor/server/session_manager.py`
- `src/modacor/server/api.py`

It provides route skeletons and an in-memory session manager aligned with this contract.
`/process` is now wired to execute MoDaCor runs with registered sources and optional HDF output writing.
The scaffold now includes dirty-step detection by changed source references and executes selected subgraphs for
`partial` mode when prior `ProcessingData` exists. `auto` mode attempts partial first and falls back to full rerun on
partial failure.
When partial mode runs, the service records a boundary checkpoint before the first dirty step and restores it if
partial execution fails.

Run the scaffold service:

```bash
pip install "modacor[server]"
modacor serve --host 127.0.0.1 --port 8000
```
