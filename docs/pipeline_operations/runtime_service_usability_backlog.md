# Runtime Service Usability Backlog

This document tracks usability upgrades for the runtime service and API ergonomics.

Status legend:

- `planned`
- `in_progress`
- `done`

## Upgrade list

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
Status: `planned`
Benefit: high
Complexity: low
Reason: improves deployability and operational safety (orchestration probes, monitoring).

8. `U10` Latest error diagnostics endpoint
Status: `planned`
Benefit: medium-high
Complexity: low-medium
Reason: speeds triage and recovery after failed runs.

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

## Recommended execution order (remaining)

1. `U4` (dry-run)
2. `U6` (last-sample shortcut)
3. `U8` (health/readiness)
4. `U10` (error diagnostics)
5. `U9` (persistence)
6. `U7` (advanced streaming)

## Notes

- Upgrades are implemented incrementally, one item at a time.
- This file is the source of truth for progress tracking.
