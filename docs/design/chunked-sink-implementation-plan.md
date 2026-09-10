# Chunked Sink Implementation Plan

Status: proposed design for review before implementation.

## Decision summary

Chunk writing belongs to `IoSink` and `IoSinks`; `IoSource` remains the read
interface. The design must keep ordinary laboratory-scale operation unchanged:
a normal sink continues to accept `write(...)`, and neither pipeline YAML nor
the caller needs a `ChunkPlan` or `ChunkSpec`.

Chunk writing will be an optional, discoverable sink capability. The initial
implementation will use separate public sink classes:

- `HDFProcessingSink` for the existing complete, replace-style write; and
- `HDFChunkedProcessingSink` for preallocation, destination-slice writes,
  resume, and finalization.

`TiledSink` remains the complete-array writer. A separate `TiledChunkedSink`
can implement the same generic capability after the required Tiled deployment
operations have been verified.

The public classes should share private layout and metadata helpers rather
than one high-level sink dispatching between modes. This avoids mode-dependent
configuration in ordinary writes while keeping the finalized HDF5 data layout
and metadata directly comparable.

Do not introduce `HDFStraightSink`: the existing `HDFProcessingSink` already
has that role and its public name should remain stable. Do not add an
`HDFSink` facade in the first implementation either. A facade would still
need a mode switch and would conceal the initialize/write/finalize lifecycle
from callers that need to handle retries. The combination of a capability
flag, two explicit public classes, and shared private layout helpers gives a
smaller compatibility surface without duplicating the HDF5 format logic.

Likewise, a separate `ChunkWritableSink` base class or mixin is unnecessary.
Chunk support is one optional capability of a sink, not a second kind of sink
that processing modules need to know about. Keeping the optional methods on
`IoSink` also leaves room for a backend to support both complete and chunked
writes in one implementation later if that becomes natural.

Pipeline processing modules will not select an output style. The existing
`SinkProcessingData` step and `IoSinks.write_data(...)` path remain unchanged.
For runtime-service operation, the server owns the chunk-specific sink
lifecycle: the orchestrator initializes a server-level output resource, each
session writes its successful chunk as a post-run action, and the orchestrator
asks the server to finalize the assembled result. The orchestrator retains
authority over planning, scheduling, retries, and the decision to finalize,
but it does not need to receive `ProcessingData` or access the destination
directly.

Direct Python callers may still invoke the same `IoSinks` lifecycle locally.
That is an alternate transport boundary, not a different sink contract or a
requirement for the HTTP workflow.

## Sink capability discovery

Add a class-level capability flag to `IoSink`:

```python
class IoSink:
    supports_chunked_writes: ClassVar[bool] = False

    def write(self, subpath: str, *args, **kwargs):
        raise NotImplementedError

    def initialize_chunked(self, subpath: str, plan: ChunkPlan, **kwargs):
        raise UnsupportedSinkCapability(type(self), "chunked_writes")

    def write_chunk(
        self,
        subpath: str,
        processing_data: ProcessingData,
        *,
        plan: ChunkPlan,
        chunk: ChunkSpec,
        **kwargs,
    ):
        raise UnsupportedSinkCapability(type(self), "chunked_writes")

    def finalize_chunked(self, subpath: str, plan: ChunkPlan, **kwargs):
        raise UnsupportedSinkCapability(type(self), "chunked_writes")
```

Chunk-capable subclasses set it to `True`. A class-level value is preferable
to mutable instance configuration because support is an implementation
property, not a user choice.

These default methods keep existing sink subclasses source-compatible and
produce a domain-specific error instead of an `AttributeError`. The flag
makes discovery simple for runtime APIs, graph editors, external runners, and
tests. It does not by itself provide the contract: a test must require every
class declaring support to override the chunk lifecycle methods. If more
optional sink operations are added later, this can evolve into a
`capabilities` set while retaining `supports_chunked_writes` as a convenience
property.

`IoSinks` should expose routed methods and reject unsupported sinks before
performing I/O:

```python
io_sinks.initialize_chunked("result::run1", plan)
io_sinks.write_chunk(
    "result::run1", processing_data, plan=plan, chunk=chunk
)
io_sinks.finalize_chunked("result::run1", plan=plan)
```

The error should name the selected sink class and state that it does not
support chunked writes. Normal `write_data(...)` must not inspect chunk
capabilities or require chunk metadata.

## Public lifecycle

`HDFChunkedProcessingSink` implements three explicit operations.

### Initialize

```python
initialize_chunked(
    subpath: str,
    plan: ChunkPlan,
    *,
    collision: Literal["error", "resume", "replace"] = "error",
) -> ChunkWriteResult
```

Initialization:

1. validates the complete plan and output layouts;
2. creates or validates the run and plan groups;
3. stores the canonical `ChunkPlan` and its hash;
4. preallocates every destination dataset at its final shape and dtype;
5. selects HDF5 chunk shapes, compression, and fill values;
6. writes invariant units, rank, and axis metadata;
7. creates the expected-chunk manifest;
8. records MoDaCor and schema versions; and
9. marks the plan `writing` before flushing and closing the file.

Collision behavior is explicit:

- `error` rejects an existing destination;
- `resume` requires matching plan hash, layouts, and software/schema
  compatibility; and
- `replace` deletes and recreates the destination and must be explicitly
  requested.

### Write one chunk

```python
write_chunk(
    subpath: str,
    processing_data: ProcessingData,
    *,
    plan: ChunkPlan,
    chunk: ChunkSpec,
) -> ChunkWriteResult
```

One write:

1. verifies the plan id and hash;
2. verifies that the chunk id and ordinal belong to the plan;
3. resolves each declared output from `ProcessingData`;
4. derives the expected shape from that output's destination selector;
5. validates actual shape, dtype, units, `rank_of_data`, weights,
   uncertainties, and axes;
6. marks the chunk `writing` and stores its `ChunkSpec`;
7. writes all destination slices;
8. flushes the arrays;
9. marks the chunk `complete`; and
10. flushes again before closing the file.

A smaller edge chunk is valid because its expected shape is derived from the
resolved selector, not compared with the plan's nominal `chunk_size`.

Retries are idempotent. A chunk left in `writing` is rewritten completely. A
chunk already marked `complete` with the same spec and execution identity may
return a no-op result; a conflicting spec is rejected. Reprocessing completed
data requires an explicit replacement policy.

### Finalize

```python
finalize_chunked(
    subpath: str,
    *,
    plan: ChunkPlan,
) -> ChunkWriteResult
```

Finalization:

1. verifies that all expected chunk ids are complete;
2. rejects incomplete or conflicting manifest entries;
3. validates final output layouts and required metadata;
4. writes plan-level pipeline and provenance records;
5. establishes the NeXus default chain;
6. marks the plan `complete`; and
7. flushes and closes the file.

Consumers must treat only a plan marked `complete` as a published assembled
result. Finalization does not receive `ProcessingData`: it operates entirely
on the stored plan, output layouts, manifest, and already written datasets.
For a global reduction, it may consume persisted reducer states, but it still
does not require the original per-chunk `ProcessingData` objects.

## Chunk and output contracts

`ChunkPlan` stores information shared by every chunk. `ChunkSpec` remains a
compact immutable record for one unit of work. Their selection semantics are
defined in [Chunked Operation](chunked-operation.md).

Before implementation, destination placement in `ChunkSpec` should be changed
from one optional selector to output-keyed placements, because pipeline outputs
may have different shapes:

```python
@dataclass(frozen=True, slots=True)
class ChunkPlacement:
    output_id: str
    destination_selection: tuple[AxisSelector, ...]
    expected_shape: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class ChunkSpec:
    schema_version: str
    plan_id: str
    plan_hash: str
    chunk_id: str
    ordinal: int
    grid_index: tuple[int, ...]
    source_selection: tuple[AxisSelector, ...]
    expected_input_shape: tuple[int, ...]
    placements: tuple[ChunkPlacement, ...]
```

Each plan output describes one `BaseData` root:

```python
@dataclass(frozen=True, slots=True)
class ChunkArrayLayout:
    component: str  # signal, weights, uncertainties/<name>, axes/<name>
    final_shape: tuple[int, ...]
    dtype: str
    placement_binding: PlacementBinding


@dataclass(frozen=True, slots=True)
class ChunkOutputLayout:
    output_id: str
    processing_path: str
    destination_path: str
    units: str
    rank_of_data: int
    arrays: tuple[ChunkArrayLayout, ...]
```

`ChunkPlacement.destination_selection` describes the output signal's
coordinate system. Each array component's static `placement_binding` in the
plan says whether it uses that selection directly, broadcasts across it, is
invariant, or projects selected axes through an explicit axis map. This keeps
each `ChunkSpec` compact while still making placement of weights,
uncertainties, and batch-dependent axes deterministic. A component that
cannot be related to the signal selection must use an explicit binding in the
plan; shape inference is not sufficient when it is ambiguous.

The first implementation should use a complete declared output schema when it
is available. When dtype, units, uncertainties, or axes are not predictable
from pipeline configuration, the server may create the output in an
`awaiting_schema` state. The first successful chunk then acts as a pilot: under
the output lock, the server derives and validates the component schema,
preallocates the datasets using the final shapes from the plan, writes the
pilot, and changes the state to `writing`. Only one worker may perform this
transition.

## `BaseData` storage rules

The chunked sink accepts output paths resolving to complete `BaseData` objects
so their numerical and metadata contract cannot be split accidentally.

- Signal is always a preallocated dataset.
- Array-valued weights are broadcast and written consistently with signal.
- A scalar weight may remain an attribute only when it is invariant for the
  complete plan; otherwise weights require a destination dataset.
- Each uncertainty key has a preallocated dataset.
- The uncertainty-key set is fixed by the output layout and cannot change
  between chunks.
- Units and `rank_of_data` are stored once and validated on every write.
- Static axes are stored once and validated when repeated.
- Batch-dependent axes require their own layout and destination placement.

The first vertical slice may support signal plus array-valued uncertainties
before adding batch-dependent axes, but it must not claim complete `BaseData`
support until all these rules are implemented.

## HDF5 layout

The finalized data tree should match an equivalent ordinary write:

```text
/processing/result/<run_name>/
    <bundle>/
        <basedata>/
            signal
            weights
            uncertainties/
                <name>
            <axis datasets>
```

Chunk administration is stored separately:

```text
/processing/chunk_plans/<plan_id>/
    plan_json
    status
    chunks/
        <chunk_id>/
            spec_json
            status
            execution_json
```

Pipeline metadata remains under `/processing/pipeline/<run_name>/`. Per-chunk
trace data may be stored under
`/processing/tracer/<run_name>/chunks/<chunk_id>/` so one chunk does not replace
another's trace.

The plan JSON is stored once. Each chunk stores its compact spec plus execution
status. Large arrays are never copied into the manifest.

## MoDaCor version provenance

Both ordinary and chunked HDF5 outputs must record the version imported from
`modacor.__version__`. The current writer records `program_name` but not the
MoDaCor version.

Store:

- `/processing/program_name` as `MoDaCor`;
- `/processing/program_version` as the current version; and
- `modacor_version` on `/processing/result/<run_name>` for an ordinary run;
  and
- `modacor_version` on `/processing/chunk_plans/<plan_id>` for a chunked run.

The per-run or per-plan value is authoritative because one HDF5 file may
contain results created by different MoDaCor versions over time. The
processing-level field is a convenient default and must not erase historical
per-run values.

Pipeline YAML/specification, pipeline hash, `ChunkPlan`, `ChunkSpec`, source
registration or revision, and MoDaCor version together form the reproducibility
record. Worker id, attempt number, timestamps, and errors belong to the chunk
execution record rather than the immutable `ChunkSpec`.

## Shared implementation without a mode-heavy facade

Do not make `HDFProcessingSink` choose between complete and chunked behavior
based on configuration. Keep the two public classes explicit and extract only
the reusable internals, for example:

```text
modacor/io/hdf/
    hdf_processing_layout.py
    hdf_processing_sink.py
    hdf_chunked_processing_sink.py
```

Shared layout helpers should handle:

- processing-path resolution;
- safe HDF5 names;
- NeXus group attributes and default links;
- `BaseData` component discovery;
- units, rank, and axis metadata;
- compression and fill-value policy; and
- JSON/text provenance fields.

`HDFProcessingSink.write()` continues to create a complete result using those
helpers. `HDFChunkedProcessingSink` creates the same destination layout first
and fills it through slices. Neither public class delegates wholesale to the
other, and the chunked writer never uses the current `_recreate_group()` helper
during a normal chunk write.

`HDFChunkedProcessingSink` is an external-runner sink and intentionally does
not make the ordinary pipeline step infer a one-chunk plan. Its inherited
`write()` remains unsupported. If experience later shows that one registered
HDF sink should expose both operations, the shared helpers make combining the
public classes mechanical without changing either lifecycle contract.

This structure makes a high-value equivalence test possible: write the same
logical `ProcessingData` once through `HDFProcessingSink` and once through
several `HDFChunkedProcessingSink` calls, then compare the finalized result
trees, values, dtypes, units, axes, uncertainties, and NeXus metadata.

## Fixed-size destination datasets

Preallocate fixed-size datasets rather than relying on append-only extensible
datasets:

```python
dataset = group.create_dataset(
    "signal",
    shape=layout.final_shape,
    dtype=layout.dtype,
    chunks=layout.hdf_chunk_shape,
    compression=compression,
    fillvalue=fillvalue,
)
dataset[destination_selection] = chunk_array
```

Destination-slice writing supports multiple batch axes, out-of-order chunks,
smaller edge chunks, and idempotent retries. Append-only datasets assume one
ordered growth dimension and are therefore not the general contract.

Suggested fill values are NaN for floating signal and uncertainties, zero for
weights, and a declared value for integer data. Completion is determined only
from the manifest, never by inspecting fill values.

## Module and runtime boundary

No processing module should need `write` versus `write_chunk` configuration.
The recommended runtime-service workflow is:

1. the external orchestrator asks the server to initialize a chunked output;
2. the server returns an opaque `output_id`;
3. each pipeline request supplies that `output_id` and one `ChunkSpec`;
4. the server executes the ordinary pipeline and then passes its in-memory
   `RunResult.processing_data` to `IoSinks.write_chunk(...)`;
5. a successful process response acknowledges both successful computation and
   a durable chunk write; and
6. after scheduling has stopped, the orchestrator inspects status and asks the
   server to finalize the output.

The `ChunkSpec` is part of the run execution envelope, like a run id or retry
identity. Passing it to the session does not make pipeline modules responsible
for chunking. A post-run action is preferable to a
`SinkChunkProcessingData` module because it has the complete `RunResult`,
trace, pipeline specification, and failure status, and it cannot run before a
later pipeline step fails.

Ordinary `SinkProcessingData` and ordinary `write_hdf` requests remain valid
without any new fields. Registration may use an explicit sink type such as
`hdf_chunked`; selecting ordinary `hdf` retains current behavior.

The current HTTP process response returns run identity, status, execution
mode, notes, and an optional HDF path; it does not serialize the complete
`ProcessingData`. Selected arrays can be exposed through `BufferSink`, but the
current public buffer API is not a complete `ProcessingData` round trip: it
does not expose generic sink attribute or metadata reads, and the buffer sink
does not preserve every axis relationship. Server-side post-run writing is
therefore both the lower-copy route and the smaller initial API change. A
complete result-download protocol may be added later for deployments in which
workers cannot access the destination through the server.

This plan therefore treats the runtime service as the primary operational
surface for chunked processing. It does not remove `run_pipeline_job(...)`:
that function remains the in-process execution engine, a useful library and
test interface, and the implementation used by the server. Whether broader
command-line execution should become a thin server launcher and API client is
a separate architecture decision and is not required for chunk writing.

## Server-managed chunked-output resource

A chunked output is a server-level resource rather than a child of one worker
session. It may receive chunks from several sessions and must remain valid if
a worker session is reset or deleted.

### Initialize

```text
POST /v1/chunked-outputs
```

The request contains a chunk-capable sink registration or a reference to one,
the sink subpath, the complete `ChunkPlan`, and the collision policy. The
server validates its write policy, snapshots the resolved sink configuration,
calls `initialize_chunked(...)`, and returns an opaque `output_id` plus plan
identity and progress counts.

If a session-registered sink is accepted for convenience, its registration is
copied into the output resource. Later changes to or deletion of that session
must not silently redirect an active output.

### Inspect

```text
GET /v1/chunked-outputs/<output_id>
```

The response reports state and counts for expected, complete, writing, failed,
and missing chunks. Detailed chunk entries should be paginated for large
plans. The manifest in HDF5 or Tiled is authoritative; server memory may cache
but must not be the only copy of this state.

### Process and publish one chunk

```json
{
  "mode": "partial",
  "changed_keys": ["sample.signal"],
  "chunk_output": {
    "output_id": "out-8c57c4",
    "chunk_spec": {
      "plan_id": "i22-saxs-42",
      "plan_hash": "sha256:...",
      "chunk_id": "c000017"
    }
  }
}
```

The server resolves the output resource only after pipeline success and calls
`write_chunk(...)` with the session's in-memory result. A write failure must be
reported distinctly from a processing failure, for example as
`CHUNK_WRITE_FAILED`, and the run is not acknowledged as successful overall
until the manifest records the chunk as complete. A retry uses the same stable
chunk id and remains idempotent.

### Finalize

```text
POST /v1/chunked-outputs/<output_id>/finalize
```

The request repeats the expected `plan_hash` as an optimistic concurrency
check. The server prevents new writes, loads the stored plan and manifest,
calls `finalize_chunked(...)`, and returns the completed state. A missing or
failed chunk produces `409 Conflict`, reports progress, and leaves the output
writable. Repeating finalization for an already completed output with the same
plan hash is a successful no-op.

A dedicated action endpoint is clearer than setting `status: complete`: the
operation validates coverage, constructs final metadata and NeXus links,
flushes storage, and publishes the result rather than merely changing a field.

The server-side `output_id` mapping must eventually be persistent or
reconstructable so an incomplete assembly can resume after a server restart.
That persistence requirement is part of the broader runtime-service lifecycle
design; the plan and manifest themselves are always persisted in the output
backend.

## Concurrency and durability

Sequential chunk writing is the acceptance target. Every operation opens,
flushes, and closes the file. The server maintains a lock per `output_id`
covering initialization, every chunk write, and finalization.

Finalization changes the state to `finalizing` while holding that lock so a
late chunk cannot race with publication. If completeness validation fails, the
output returns to `writing`; after successful publication it becomes
`complete` and rejects later writes.

The contract should also permit later concurrent writers when the deployed
HDF5 library and filesystem locking have been validated. Concurrent workers
must use the same plan hash, write disjoint destination slices, and update
separate chunk manifest entries. A status transition to `complete` occurs only
after every component for that chunk has been flushed.

HDF5 writes are not transactional. A process can fail after modifying some
datasets but before completing its manifest entry. Idempotent rewrite of every
component is therefore required when retrying a `writing` chunk.

An in-process lock coordinates sessions in one server process. Multiple
service processes or replicas require a shared lock or a designated writer;
session affinity alone does not serialize access to a shared HDF5 file.

The generic capability must not expose `h5py` objects or HDF-specific locking
details. A future `TiledChunkedSink` can use native destination-slice writes or
unique per-chunk nodes plus a completion manifest, depending on verified
server capabilities.

## Implementation phases

### Phase 1: contracts and version provenance

- Implement and test immutable `ChunkPlan`, `ChunkSpec`, selectors, placements,
  serialization, normalization, and hashing.
- Add `supports_chunked_writes` to `IoSink`.
- Add capability-checked routing methods to `IoSinks`.
- Add MoDaCor version metadata to ordinary HDF5 output.
- Keep all existing non-chunked tests and public behavior unchanged.

### Phase 2: signal-only vertical slice

- Add `HDFChunkedProcessingSink` registration as `hdf_chunked`.
- Implement initialize, fixed-shape signal allocation, slice writes, manifest,
  resume, and finalize.
- Cover multiple batch axes, out-of-order writes, and smaller edge chunks.
- Compare finalized signal output directly with `HDFProcessingSink` output.

### Phase 3: complete `BaseData`

- Add weights, all uncertainties, units, `rank_of_data`, and static axes.
- Add batch-dependent axis layouts.
- Extract shared HDF layout helpers only where duplication is demonstrated.
- Extend whole-versus-chunked equivalence tests to the complete result tree.

### Phase 4: runtime integration

- Carry optional `ChunkSpec` through `run_pipeline_job` and `RunResult`.
- Attach chunk identity to trace events without copying plan-wide metadata into
  every event.
- Add a server-level chunked-output manager and opaque `output_id`.
- Add initialize, inspect, and finalize API operations.
- Add `chunk_output` to process requests and call `write_chunk(...)` as a
  post-run action using the session's in-memory `ProcessingData`.
- Distinguish pipeline failures from chunk-publication failures and acknowledge
  success only after the chunk manifest is durable.
- Support the `awaiting_schema` pilot-chunk transition when layouts cannot be
  declared completely in advance.
- Preserve full and partial pipeline-run behavior.

### Phase 5: beamline readiness

- Verify bounded memory on representative I22-scale data.
- Demonstrate numerical and metadata equivalence with a whole-array run.
- Test interruption, retry, resume, missing chunks, and duplicate requests.
- Measure HDF5 layout and compression performance.
- Validate serialized concurrent writes on the target filesystem separately.
- Define operational cleanup and incomplete-plan recovery procedures.

### Phase 6: Tiled implementation

- Confirm destination-slice and publication behavior against the facility
  Tiled deployment.
- Implement `TiledChunkedSink` using the same contracts and capability flag.
- Run backend-neutral conformance tests plus Tiled integration tests.

## Required tests

Contract tests cover selector conversion, edge-chunk shapes, plan hashing,
stable ids, output-keyed placement, overlap detection, and JSON round trips.

Generic sink tests verify that normal sinks report no chunk support, capable
sinks implement the complete lifecycle, unsupported routed calls fail before
I/O, and `write_data(...)` remains unchanged.

HDF5 tests cover initialization, multidimensional placement, smaller final
chunks, out-of-order writes, duplicate and conflicting chunks, interruption
and retry, reopen and resume, shape/dtype/unit/rank mismatches, uncertainty and
weight schemas, static and batch axes, missing-chunk finalization, unrelated
file-content preservation, NeXus defaults, version provenance, and complete
tree equivalence with an ordinary write.

Runtime tests cover optional chunk metadata, post-run writes after full and
partial reruns, output initialization and inspection, finalize-without-
`ProcessingData`, incomplete-finalization conflicts, idempotent finalization,
pilot schema initialization, session recreation, trace correlation, distinct
processing and publication errors, and the absence of new requirements for
ordinary requests. They must also verify that deleting a worker session does
not invalidate a server-level output resource.

## Initial acceptance criteria

- Existing non-chunked pipelines and sink configurations run unchanged.
- Ordinary output records the MoDaCor version.
- Peak assembly memory is proportional to one chunk, excluding the fixed HDF5
  dataset storage.
- Final signal, weights, uncertainties, units, ranks, axes, and NeXus metadata
  match an equivalent ordinary write.
- An edge chunk smaller than the nominal chunk size writes successfully.
- Chunks may be written out of order and retried idempotently.
- Incomplete output cannot be mistaken for a finalized plan.
- The orchestrator can initialize, inspect, and finalize an output without
  receiving `ProcessingData` or accessing the destination directly.
- A successful chunk process response means the corresponding chunk is
  durably marked complete.
- Plan, chunk, pipeline, source, trace, and MoDaCor-version provenance can be
  correlated without storing array payloads in metadata.
- The generic interface is suitable for a later Tiled implementation without
  exposing HDF5-specific behavior.
