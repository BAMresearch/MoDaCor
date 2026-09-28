# Buffers and chunked outputs

Buffers transfer arrays and metadata through the runtime API without requiring
the server to open a caller-owned file. Register a buffer source, upload its
arrays/attributes/metadata through `SessionClient.source_buffer(...)`, then run
the pipeline normally. A buffer sink exposes selected results for download.

Chunked outputs are server-level resources rather than session-local temporary
objects. The lifecycle is:

1. Create a fixed or provisional output from a `ChunkPlan`.
2. Resolve a provisional layout when the first representative result supplies
   missing shape information.
3. Process and publish chunks with matching plan hash, chunk ID, shapes, units,
   and placements.
4. Inspect completion and errors.
5. Finalize only when all expected chunks are present.

`RuntimeClient.chunked_outputs` creates or reopens handles. A
`ChunkedOutputHandle` can fetch a chunk spec, submit a chunk, inspect progress,
finalize, recover, and detach.

Destination paths must not be shared by concurrently writable resources.
Server-side target locks and plan hashes protect against accidental mixing, but
an external orchestrator remains responsible for assigning each chunk once and
handling retries idempotently.

See [Chunked processing](../processing/chunked-processing.md) for the processing
model and the [I/O capability matrix](../reference/io-capabilities.md) for
backend support.
