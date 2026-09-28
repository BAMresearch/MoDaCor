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
