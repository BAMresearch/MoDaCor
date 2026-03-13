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

2. `U2` Source patch convenience endpoint (`POST /sessions/{id}/sources/patch`)
Status: `done`
Notes: implemented in `src/modacor/server/api.py` with OpenAPI + docs updates.

3. `U3` Source templates/profiles (e.g. MOUSE, SAXSess)
Status: `done`
Notes: implemented as built-in profiles with `GET /v1/source-templates` and session-level `source_profile` validation.

4. `U4` Dry-run endpoint for invalidation preview
Status: `planned`

5. `U5` Richer run summaries (dirty/skipped steps, fallback reason, timings)
Status: `done`
Notes: run metadata now includes `dirty_steps`, `skipped_steps`, `step_durations_s`, `elapsed_s`, and fallback fields.

6. `U6` “Last sample” shortcut endpoint
Status: `planned`

7. `U7` Improved event streaming (persistent WS + SSE option)
Status: `planned`

8. `U8` Health/readiness split endpoints with runtime metrics
Status: `planned`

9. `U9` Persistent session store
Status: `planned`

10. `U10` Latest error diagnostics endpoint
Status: `planned`

## Notes

- Upgrades are implemented incrementally, one item at a time.
- This file is the source of truth for progress tracking.
