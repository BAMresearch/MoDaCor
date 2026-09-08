# Pipeline Operations Backlog

This page tracks runtime-service feature upgrades and the structural refactors
that keep the pipeline-operations code maintainable as the service grows.

Status legend:

- `planned`
- `in_progress`
- `done`

## Current focus

The `U6` to `U8` maintenance window is now closed.

- `U6` is complete.
- `U8` is now complete.
- `U10` is now complete.
- `U9` and `U7` remain planned feature upgrades.
- The immediate follow-up is documentation tightening and the next operational
  persistence and streaming features.

## Upgrade ledger

1. `U1` `modacor session` CLI wrapper for session lifecycle and processing calls
Status: `done`
Notes: implemented in `src/modacor/cli.py` as `modacor session ...` with commands:
`list`, `create`, `delete`, `status`, `set-source`, `delete-source`, `process`, `reset`, `runs`.
Benefit: high
Complexity: medium
Reason: removes raw HTTP friction and improves day-to-day operator ergonomics.

2. `U2` Source patch convenience endpoint (`POST /sessions/{id}/sources/patch`)
Status: `done`
Notes: implemented in `src/modacor/server/api.py` with OpenAPI + docs updates.
Benefit: medium
Complexity: low
Reason: simplifies common single-source update workflow.

3. `U3` Source templates/profiles (e.g. MOUSE, SAXSess)
Status: `done`
Notes: implemented as built-in profiles with `GET /v1/source-templates` and session-level `source_profile` validation.
Benefit: high
Complexity: medium
Reason: prevents misconfiguration and standardizes source expectations per instrument family.

4. `U4` Dry-run endpoint for invalidation preview
Status: `done`
Benefit: very high
Complexity: low-medium
Reason: immediate visibility into dirty-step decisions; improves trust and debugging without executing a run.

5. `U5` Richer run summaries (dirty/skipped steps, fallback reason, timings)
Status: `done`
Notes: run metadata now includes `dirty_steps`, `skipped_steps`, `step_durations_s`, `elapsed_s`, and fallback fields.
Benefit: high
Complexity: medium
Reason: improves observability and post-mortem diagnostics.

6. `U6` “Last sample” shortcut endpoint
Status: `done`
Benefit: high
Complexity: low
Reason: maps the most frequent operation to a single focused API call.
Notes: implemented as `POST /v1/sessions/{id}/sample` plus CLI `modacor session set-sample ...`.

7. `U8` Health/readiness split endpoints with runtime metrics
Status: `done`
Benefit: high
Complexity: low
Reason: improves deployability and operational safety (orchestration probes, monitoring).
Notes: implemented as `GET /v1/health` for liveness and `GET /v1/readiness`
for service usability plus runtime metrics (`session_count`,
`active_run_count`, `error_session_count`, `error_session_ids`,
`last_updated_utc`).

8. `U10` Latest error diagnostics endpoint
Status: `done`
Benefit: medium-high
Complexity: low-medium
Reason: speeds triage and recovery after failed runs.
Notes: implemented as `GET /v1/sessions/{id}/errors/latest`, returning the
current session error state plus the latest recorded failed-run diagnostics.
The CLI wrapper now exposes the same payload via
`modacor session last-error --session-id ...`.

9. `U9` Persistent session store
Status: `planned`
Benefit: high
Complexity: medium-high
Reason: preserves runtime definitions/state across restarts; requires careful state/version handling.

10. `U7` Improved event streaming (persistent WS + SSE option)
Status: `planned`
Benefit: medium-high
Complexity: high
Reason: valuable for real-time UIs and remote control loops but introduces more protocol/runtime complexity.

## Structural maintenance window

These tasks are intentionally scheduled between `U6` and `U8` so the runtime
service grows on a cleaner foundation.

1. `S1` Runtime-service module split
Status: `done`
Scope: `src/modacor/server/api.py` now stays focused on FastAPI route bindings
and HTTP translation, while `src/modacor/server/runtime_service.py` owns
session orchestration, `src/modacor/server/planning.py` owns dry-run and dirty
step planning, `src/modacor/server/io_utils.py` adapts session source
registrations, and `src/modacor/server/errors.py` defines framework-agnostic
service errors.
Benefit: high
Complexity: medium
Reason: separates transport concerns from runtime behavior and makes service
logic easier to test in isolation.

2. `S2` Python 3.12 floor alignment
Status: `done`
Scope: package metadata, CI, tox, and installation documentation all target
Python `3.12+`.
Benefit: high
Complexity: low
Reason: removes unsupported-version ambiguity and matches the codebase's
typing/runtime requirements.

3. `S3` Test-package separation
Status: `done`
Scope: package-internal tests were moved out of `src/modacor/tests` into the
top-level `tests/` tree, and distribution discovery now excludes
`modacor.tests*`.
Benefit: high
Complexity: medium
Reason: keeps shipped packages lean and avoids coupling runtime imports to test
helpers.

4. `S4` Process-step export and docs unification
Status: `done`
Scope: `modacor.modules.__all__` now exports every supported `ProcessStep`, and
the module-doc generation tests assert that exported steps and generated docs
stay in sync.
Benefit: high
Complexity: medium
Reason: removes the split between runtime discovery and reference
documentation.

5. `S5` Shared runtime I/O helpers for CLI and service execution
Status: `done`
Scope: `src/modacor/io/runtime_support.py` now centralizes source builders,
sink builders, and shared HDF export handling used by both `modacor run` and
the runtime service.
Benefit: high
Complexity: medium
Reason: eliminates duplicated source/HDF handling and makes future runtime
upgrades less error-prone.

## Recommended next feature order

1. `U9` (persistent session store)
2. `U7` (advanced streaming)

## Notes

- Upgrades are implemented incrementally, with structural maintenance work
  allowed between feature upgrades when it reduces future implementation risk.
- This file is the source of truth for progress tracking.
- Companion design notes live in `docs/pipeline_operations/runtime_service_api.md`.
