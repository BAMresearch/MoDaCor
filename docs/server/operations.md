# Server operations

A normal session lifecycle is:

1. Verify `/v1/health` and `/v1/readiness`.
2. Create or replace a session with pipeline YAML and trace configuration.
3. Register sources and sinks, or upload source-buffer content.
4. Preview invalidation with a dry run when using partial execution.
5. Process in `full`, `partial`, or `auto` mode.
6. Inspect the run record, trace, sink output, and latest error when applicable.
7. Reset state, recover a failed output, or delete the session.

Full mode rebuilds processing state and runs the complete graph. Partial mode
requires a prior result and declared changed sources or keys. Auto mode attempts
partial execution and falls back to a full run when configured to do so.

Session operations are serialized per session. Multiple sessions may run in
parallel, subject to service limits and the behavior of their external storage.
Use readiness metrics and run history instead of inferring completion from file
appearance alone.

For exact payloads, status codes, and error fields, use the
[server API reference](../reference/server-api.md).

## Logs and troubleshooting

`LocalRuntimeServer` writes captured server output to
`./modacor_output/modacor_server.log` by default, relative to the working
directory when it starts. It creates the directory, prints the resolved path,
and exposes it as `local_server.log_path`. Pass `log_path=...` to choose another
file. Errors from its managed Python client include this path. Reusing an
already running external server does not create or claim its log file.

`modacor serve` logs to stderr. Use `--log-file PATH` to also write to a file,
and `--log-level debug|info|warning|error|critical` to set the console and file
threshold (default: `info`). Both destinations use
`%(asctime)s - %(name)s - %(levelname)s - %(message)s`. LocalRuntimeServer
captures the subprocess console output itself; it does not also pass
`--log-file` for that destination.

Source and sink registrations and updates return an additive `warnings` list.
Each entry contains `ref`, `kind`, `location` and `problem`. Missing, non-file
or unreadable file sources and unwritable file destinations produce warnings
without rejecting registration, allowing files to arrive during acquisition.
File sinks require an existing parent directory, or a writable existing file.
Buffer, Tiled and custom backends are not checked at registration. The Python
runtime/session client displays these problems as `RuntimeWarning`.

Runtime source/sink construction failures return `SOURCE_UNAVAILABLE` or
`SINK_UNAVAILABLE`, naming the ref and location. Their error `details` include
`ref`, `type`, `location` and `exception_type` alongside `session_id` and
`run_id`. Unexpected API handler failures return structured `INTERNAL_ERROR`
with the exception type and message. API errors, registration problems and
run failures are logged; server errors include tracebacks and run failures
include the failed step ID when known.

Inspect `GET /v1/sessions/{id}/errors/latest` before deleting a failed session
to retrieve its in-memory error record and traceback. Deleting a session
discards its error records; the server log remains available on disk.
