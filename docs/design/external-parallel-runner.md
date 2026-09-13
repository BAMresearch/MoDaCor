# External Parallel Runner

Status: recommended orchestration approach; implementation is deferred.

## Responsibility boundary

An external runner should own chunk planning, source slicing, worker
scheduling, retries, backpressure, and the decision to finalize. The MoDaCor
server owns the output assembly mechanics: it initializes a chunked-output
resource, processes one already resolved chunk per worker request, writes the
successful result directly to the registered sink, and finalizes only when
asked by the orchestrator. The chunk contract is defined in
[Chunked Operation](chunked-operation.md).

This keeps pipeline modules independent of execution policy and allows the
runner to choose local processes, runtime API sessions, distributed workers,
or a facility scheduler without changing the correction graph.

## Source delivery modes

Workers may push already resolved chunks into session `BufferSource`
registrations, or ask the server to pull slices directly from registered
`HDFSource` and `TiledSource` inputs. These are complementary operating modes,
not different output protocols: both submit the same `ChunkSpec` and publish
through the same server-managed chunked-output resource.

Buffer delivery is suitable for remote runners or
storage that is inaccessible to the server. It incurs HTTP serialization and
memory copies, and the runner is responsible for uploading mutually aligned
arrays. Direct HDF5 or Tiled access avoids transporting chunk arrays through
the orchestrator and keeps storage identity closer to the read. The server now
projects a `ChunkSpec` onto exact datasets declared by the plan's typed source
bindings. The HDF5 route has end-to-end coverage and the Tiled route has
deterministic server coverage; representative deployed-Tiled validation remains
part of beamline readiness. The binding contract is specified in
[Chunked Operation](chunked-operation.md).

A deployment may use both modes. For example, a facility server may read raw
detector frames directly from Tiled while receiving a runner-generated dynamic
mask through a buffer. The plan must give every input an explicit `aligned`,
`static`, or `explicit` binding and the execution record must preserve the
effective source selection or uploaded-source identity.

## Recommended worker model

Use a fixed number of long-lived workers. Each worker owns an independent
pipeline instance, `ProcessingData`, source and sink registries, trace state,
and runtime session when the API is used.

```text
chunk planner -> bounded work queue -> independent sessions -> server-managed output
```

One active run is allowed per runtime session. Reuse one session per worker
rather than creating one session per chunk. Replace or clear its buffer input,
run the required pipeline portion with its `output_id` and `ChunkSpec`, and
release large input-buffer entries before accepting another chunk. The server
writes the in-memory result as a post-run action, so the orchestrator does not
normally download `ProcessingData`. The first chunk normally uses a complete
run; later chunks may use partial reruns as described below.

Pipeline objects should be reconstructed from YAML in each worker. Live
`ProcessStep` instances contain mutable processing and prepared state and must
not be shared between concurrent runs.

The runner should bound both worker count and queued chunks:

```text
peak memory ~= active workers * peak memory per chunk + queued buffer copies
```

Tracing may remain enabled, but full `ProcessingData` snapshots should normally
be disabled for large chunks.

## Reusing invariant pipeline results

A long-lived worker session may retain `ProcessingData` from its preceding
chunk. After the first complete run, subsequent chunks can use the runtime
service's partial-rerun support so unchanged work is not repeated. Typical
invariants include static mask loading, detector pixel coordinates, geometry
maps, and pixel-index maps.

After replacing the chunk-dependent buffer arrays, a request may resemble:

```json
{
  "mode": "partial",
  "changed_keys": ["sample.signal"],
  "rollback_snapshot": false
}
```

Use `changed_sources` instead when every pipeline input associated with a
source reference changed. Keep invariant and chunk-dependent arrays under
different source references where possible; otherwise source-level
invalidation may rerun static loaders unnecessarily.

Partial execution uses process-step dependency contracts and graph descendants
to find the dirty subgraph. For effective reuse, static calculations should be
on independent graph branches rather than placed linearly downstream of a
chunk loader. The eventual merge or correction steps will rerun when their
chunk-dependent input changes, while an independent static branch can remain
unchanged.

This optimization requires every chunk-dependent output to be overwritten or
invalidated. A module that appends to an existing result or depends on
undeclared state may retain data from the preceding chunk. The runner should
periodically compare partial results with a clean complete run, and fall back
to a complete run whenever dependency coverage is uncertain.

`rollback_snapshot: false` avoids a deep copy of the retained
`ProcessingData`, which is desirable for large chunks. The tradeoff is that a
failed partial run cannot restore the previous in-memory state; the next
attempt should then start with a complete reset. Each parallel worker has its
own session, so reuse and failure recovery remain isolated per worker.

## Execution options

### Direct local process pool

A process pool calling `run_pipeline_job(...)` directly remains a supported
library-level option. It avoids HTTP serialization and gives each run
independent Python and module state, but it also requires the external runner
to construct sources and sinks and call the chunk lifecycle itself. The
server-oriented workflow does not depend on this option.

### Multiple API sessions

Separate sessions in one MoDaCor service isolate buffers and processing state
and can support modest concurrency. They still share the Python process and
some service infrastructure, so scaling depends on whether the active modules
release the GIL and on backend I/O contention. This is the recommended first
execution model because sessions can write their results directly while the
server serializes output lifecycle operations.

### Service replicas

Multiple single-process service replicas provide stronger isolation. Because
the current session registry and buffers are process-local, the external
runner must consistently route a worker session to the same replica. Generic
load balancing without session affinity is not sufficient.

## Result writing

Before scheduling work, the orchestrator creates one server-level chunked
output and receives an opaque `output_id`. Each session includes this id and
its immutable `ChunkSpec` in the process request. After successful processing,
the server passes the session's in-memory `ProcessingData` directly to the
chunk-capable sink. A process request succeeds only after the chunk manifest is
durable.

The orchestrator inspects output progress and calls the server's finalize
operation after it has stopped scheduling chunks. Finalization reads the
stored plan, manifest, and datasets; it does not require `ProcessingData` to be
returned to the orchestrator. The full server API and sink lifecycle are
specified in
[Chunked Sink Implementation Plan](chunked-sink-implementation-plan.md).

Concurrent HDF5 writes can work in installations where every process opens and
closes the file for each operation and the HDF5 library and filesystem locking
correctly serialize writers. This may be a useful deployment option, but it
must be verified on the actual filesystem. Locking alone does not make a
multi-object update transactional. The initial server implementation should
therefore serialize initialization, slice writes, and finalization with one
lock per output. Multiple service processes or replicas require a shared lock
or designated writer before targeting the same HDF5 file.

Tiled is the preferred backend for distributed or multi-host runners when the
facility deployment supports concurrent clients, authentication, and suitable
array-write operations. It avoids sharing local file handles and can expose
chunk artifacts immediately through unique catalog paths. The current MoDaCor
`TiledSink` still writes complete arrays and does not provide partial-array
assembly or atomic publication of several arrays. Therefore, use unique
per-chunk paths plus a completion manifest unless the deployed Tiled service
provides a verified destination-slice workflow.

## Correctness and retries

Each submitted job should contain the immutable `ChunkSpec`. The execution
record adds attempt number, worker identity, timestamps, pipeline hash, source
registration or revision, outcome, and error details.

The server-managed output should:

- reject results whose plan hash is unexpected;
- make writes idempotent by `chunk_id` and destination selection;
- verify result shape and dtype before placement;
- record completion only after all arrays for a chunk are durable;
- allow a failed or lost chunk to be submitted again; and
- publish a plan-level completion marker only after every expected chunk has
  succeeded.

The orchestrator remains responsible for deciding when to request
finalization and for resubmitting chunks whose process or publication attempt
failed. It does not need destination credentials or a complete
`ProcessingData` transport protocol.

Parallel workers must not independently average chunk-level averages when the
desired result is a global reduction. Such pipelines require sufficient
statistics or another mergeable reducer state.

## Initial recommendation

Start with a sequential external orchestrator using one or more long-lived API
sessions and one server-managed chunked output. Once numerical equivalence,
memory bounds, and idempotent assembly are demonstrated, add a bounded pool of
sessions. Keep server-side writes serialized initially, then validate
concurrent HDF5 or Tiled publication separately against representative
facility infrastructure. Buffer-fed and direct HDF5 reads are available;
validate direct Tiled reads as a separate beamline deployment profile before
relying on them operationally.
