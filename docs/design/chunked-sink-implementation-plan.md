# Chunked Sink Implementation Plan

Status: implementation in progress.

## Implementation progress

As of 2026-09-12:

- Phase 1 is implemented: immutable chunk contracts, normalized selectors,
  canonical plan hashing and serialization, optional `IoSink` capability
  routing, `hdf_chunked` registration, and ordinary HDF5 MoDaCor-version
  provenance are covered by focused tests.
- Phase 2 established the initial signal-only vertical slice. It preallocates
  fixed-shape HDF5 datasets, writes contiguous destination slices across one
  or more batch axes, accepts smaller edge chunks and out-of-order delivery,
  detects overlap, supports idempotent retry and resume, validates complete
  coverage, and finalizes the NeXus default chain.
- Phase 3 is implemented: layouts now carry explicit component-level axis
  units/rank and ordered axis names, and the HDF5 writer supports direct,
  broadcast, static, and axis-mapped placement for weights, all declared
  uncertainties, invariant axes, and batch-dependent axes. It rejects
  undeclared `BaseData` components and changes to static values. Finalized
  complete `BaseData` trees are tested against the ordinary writer.
- The complete-schema server path in Phase 4 is implemented. Server-level
  output resources snapshot sink registrations, use opaque `output_id` values
  and destination locks, expose initialize/inspect/finalize operations, and
  publish each successful run's in-memory `ProcessingData` before acknowledging
  the request. Outputs remain usable after a worker session is deleted.
- The generic runtime sink builder and server registration model recognize
  `hdf_chunked`, including normal write-root enforcement. This is capability
  discovery and configuration, and requests containing `chunk_output` now drive
  the post-run write lifecycle. Requests without `chunk_output` are unchanged.
- Successful server publications persist lightweight, array-free trace events
  per chunk under `/processing/tracer/<run_name>/chunks/<chunk_id>/`.
  Finalization preserves those groups, and each event carries its chunk
  identity. Processing-data snapshots remain opt-in and are not part of this
  lightweight path.
- `HDFSource` caches complete-array reads only. Explicit slice reads bypass the
  cache so a sequential reader does not retain every previously delivered
  chunk; focused tests distinguish this from the reusable full-read cache.
- The fixture-independent part of Phase 5 is implemented: a new server can
  reconstruct an output handle from the persisted HDF plan, operators can
  reconcile, abandon, resume, or detach assemblies without deleting data,
  target-level locks are exercised with concurrent handles, and an opt-in
  subprocess benchmark measures HDF layout, throughput, validation, and peak
  RSS for generated or external HDF datasets.
- The first external-data checkpoint is implemented in the I22 notebook in the
  `MoDaCor_examples` repository. A zero-copy virtual view selects ten real SAXS
  frames, the ordinary sink writes them in one operation, and the chunked sink
  assembles five two-frame chunks. The stored arrays match exactly. This
  validates representative input decoding and HDF assembly, but not yet
  full-scale memory behavior or correction-pipeline equivalence.
- The I22 notebook now includes a server-driven SAXS/WAXS example for four
  100-frame measurements split into ten-frame chunks. Independent detector
  plans store two run groups in one physical HDF5 file and retain lightweight
  per-chunk traces. A reduced real-data run completes two chunks for each
  detector in the shared file and exercises buffer replacement, pilot schema
  extraction, initialization, partial reruns, publication, inspection, and
  finalization. The configured 80-pipeline-run exercise remains an interactive
  validation rather than a CI test.
- `ChunkPlan.source_bindings` now provides typed `aligned`, `static`, and
  `explicit` mappings from a chunk driver to registered source datasets. The
  runtime applies request-scoped selectors to direct HDF5 or Tiled reads,
  automatically invalidates affected partial-run branches, rejects double
  slicing of staged buffers, and persists the effective selectors with chunk
  execution metadata. Direct HDF5 assembly is covered end to end and a
  deterministic Tiled test reaches backend sliced reads without filling its
  complete-array cache; validation against a deployed Tiled service remains
  open.
- Verification at the current Phase 5 checkpoint passes 770 tests, with the
  opt-in RSS regression skipped by default; running that check explicitly also
  passes. The three reported test warnings are pre-existing numerical-domain
  warnings in `BaseData` tests. The documentation builds cleanly with Sphinx
  warnings treated as errors, and the OpenAPI YAML parses successfully.

The optional `awaiting_schema` path is deliberately still open. The current
immutable `ChunkPlan` hash includes dtype, units, uncertainty, axis, and
component layout. Deriving those fields from a pilot chunk would therefore
change the plan identity after chunks had already been issued. Supporting this
cleanly needs a separate provisional-plan/schema-resolution contract; the
server rejects incomplete plans instead of mutating their identity.

The current HDF chunk writer requires contiguous destination slices with
stride 1. Strided destination writes remain deliberately unsupported and are
rejected rather than silently reinterpreted.

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

    def inspect_chunked(self, subpath: str, *, plan: ChunkPlan, **kwargs):
        raise UnsupportedSinkCapability(type(self), "chunked_writes")

    def finalize_chunked(self, subpath: str, *, plan: ChunkPlan, **kwargs):
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
io_sinks.inspect_chunked("result::run1", plan=plan)
io_sinks.finalize_chunked("result::run1", plan=plan)
```

The error should name the selected sink class and state that it does not
support chunked writes. Normal `write_data(...)` must not inspect chunk
capabilities or require chunk metadata.

## Public lifecycle

`HDFChunkedProcessingSink` implements four explicit operations.

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
    trace_events: list[TraceEvent] | None = None,
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
7. writes all destination slices and any lightweight per-chunk trace events;
8. flushes the arrays and trace record;
9. marks the chunk `complete`; and
10. flushes again before closing the file.

A smaller edge chunk is valid because its expected shape is derived from the
resolved selector, not compared with the plan's nominal `chunk_size`.

Retries are idempotent. A chunk left in `writing` is rewritten completely. A
storage exception after the manifest enters `writing` changes that entry to
`failed`; it retains its placement ownership and is also rewritten completely
on retry. A
chunk already marked `complete` with the same spec and execution identity may
return a no-op result; a conflicting spec is rejected. Reprocessing completed
data requires an explicit replacement policy.

### Inspect

`inspect_chunked(...)` opens the backend read-only and reports overall state;
expected, complete, writing, failed, and missing counts; and a pageable range
of compact chunk entries. It validates the requested plan and initialized
layout before returning, so the API does not treat server memory as the
authoritative manifest.

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

The implemented contract uses output-keyed placements rather than one optional
destination selector, because pipeline outputs may have different shapes:

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


@dataclass(frozen=True, slots=True)
class ChunkSourceBinding:
    source_ref: str
    data_key: str
    role: Literal["aligned", "static", "explicit"]
    axis_map: tuple[int | None, ...] = ()
```

`ChunkPlan.source_bindings` stores these executable mappings once. The
free-form `ChunkPlan.bindings` field remains planning provenance and is not
used to drive source reads.

Each plan output describes one `BaseData` root:

```python
@dataclass(frozen=True, slots=True)
class ChunkArrayLayout:
    component: str  # signal, weights, uncertainties/<name>, axes/<name>
    final_shape: tuple[int, ...]
    dtype: str
    placement_binding: PlacementBinding
    units: str | None = None
    rank_of_data: int | None = None


@dataclass(frozen=True, slots=True)
class ChunkOutputLayout:
    output_id: str
    processing_path: str
    destination_path: str
    units: str
    rank_of_data: int
    arrays: tuple[ChunkArrayLayout, ...]
    axis_names: tuple[str, ...] = ()
```

`ChunkPlacement.destination_selection` describes the output signal's
coordinate system. Each array component's static `placement_binding` in the
plan says whether it uses that selection directly, broadcasts across it, is
invariant, or projects selected axes through an explicit axis map. This keeps
each `ChunkSpec` compact while still making placement of weights,
uncertainties, and batch-dependent axes deterministic. A component that
cannot be related to the signal selection must use an explicit binding in the
plan; shape inference is not sufficient when it is ambiguous.

For an `axes/<name>` component, `axis_names` records which signal dimension or
dimensions refer to that dataset, including repeated names for a
multidimensional coordinate. Component-level `units` and `rank_of_data`
describe the axis dataset itself. The tuple must have the signal's final rank,
and every non-`.` name must have one corresponding declared axis component.
This separates axis identity from placement and avoids guessing from array
lengths.

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

Phase 3 implements all of these rules. Signal-only plans remain valid for
pipelines whose results contain only the signal plus the default scalar weight;
non-default weights, uncertainties, or axes must be declared and are rejected
if omitted from the plan.

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

Pipeline metadata remains under `/processing/pipeline/<run_name>/`. Lightweight
per-chunk trace data is stored under
`/processing/tracer/<run_name>/chunks/<chunk_id>/` so one chunk does not replace
another's trace. These groups contain array-free `TraceEvent` data and indexed
step summaries. Processing-data snapshots are deliberately excluded from the
default chunk publication path because their volume can defeat bounded
operation. The plan-level pipeline specification omits run-specific trace
events so the latest chunk is not duplicated or mistaken for a plan-wide
trace.

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

- [x] Implement and test immutable `ChunkPlan`, `ChunkSpec`, selectors, placements,
  serialization, normalization, and hashing.
- [x] Add `supports_chunked_writes` to `IoSink`.
- [x] Add capability-checked routing methods to `IoSinks`.
- [x] Add MoDaCor version metadata to ordinary HDF5 output.
- [x] Keep all existing non-chunked tests and public behavior unchanged.

### Phase 2: signal-only vertical slice

- [x] Add `HDFChunkedProcessingSink` registration as `hdf_chunked`.
- [x] Implement initialize, fixed-shape signal allocation, slice writes, manifest,
  resume, and finalize.
- [x] Cover multiple batch axes, out-of-order writes, and smaller edge chunks.
- [x] Compare finalized signal output directly with `HDFProcessingSink` output.

### Phase 3: complete `BaseData`

- [x] Extend and test array layouts with component units/rank and output layouts
  with ordered axis names.
- [x] Add weights, all uncertainties, units, `rank_of_data`, and static axes.
- [x] Add batch-dependent axis layouts.
- [x] Extract shared HDF layout helpers only where duplication is demonstrated.
- [x] Extend whole-versus-chunked equivalence tests to the complete result tree.

### Phase 4: runtime integration

- [x] Carry optional `ChunkSpec` through `run_pipeline_job` and `RunResult`.
- [x] Attach chunk identity to trace events without copying plan-wide metadata into
  every event.
- [x] Persist lightweight trace events at a stable per-run, per-chunk HDF5 path.
- [x] Add a server-level chunked-output manager and opaque `output_id`.
- [x] Add initialize, inspect, and finalize API operations.
- [x] Add `chunk_output` to process requests and call `write_chunk(...)` as a
  post-run action using the session's in-memory `ProcessingData`.
- [x] Distinguish pipeline failures from chunk-publication failures and acknowledge
  success only after the chunk manifest is durable.
- [ ] Support the `awaiting_schema` pilot-chunk transition when layouts cannot be
  declared completely in advance.
- [x] Preserve full and partial pipeline-run behavior.

### Phase 5: beamline readiness

- [x] Add an opt-in subprocess RSS scaling regression using generated data.
- [x] Add an external-HDF benchmark harness and machine-readable report format.
- [x] Test interruption, retry, restart reconstruction, recovery actions,
  missing chunks, duplicate requests, and in-process target serialization with
  small deterministic data.
- [x] Define non-destructive operational detach, abandon, resume, and
  incomplete-plan reconciliation procedures.
- [ ] Verify bounded memory on representative I22-scale data.
- [ ] Demonstrate numerical and metadata equivalence with a whole-array run.
- [ ] Measure and select HDF5 layout and compression defaults on representative
  data and storage.
- [ ] Validate serialized concurrent writes on the target filesystem separately.

### Follow-up: server-side source slice binding

The runtime accepts chunks staged through `BufferSource` and can now project a
`ChunkSpec` through typed plan bindings for direct HDF5 or Tiled reads.

- [x] Implement request-level slice bindings without mutating pipeline YAML or
  re-registering sources for each chunk.
- [x] Resolve and validate `aligned`, `static`, and `explicit` bindings across
  signal and all chunk-dependent companion arrays.
- [x] Integrate binding changes with partial-run dependency invalidation.
- [x] Preserve explicit-slice cache bypass for HDF5 and Tiled while retaining
  reusable static reads.
- [x] Persist effective selectors plus source type, location, and dataset/node
  identity in execution provenance.
- [x] Test direct HDF5 reads against the server output and partial-rerun
  lifecycle.
- [x] Test direct Tiled reads through a deterministic server integration.
- [ ] Test mixed-source sessions and a deployed Tiled service against the same
  output and retry lifecycle, including a stable Tiled revision identifier.
- [x] Keep non-chunked sessions configuration-free and behaviorally unchanged.

The executable workflow and the boundary between repository tests and external
beamline data are documented in
[Chunked Beamline Validation](chunked-beamline-validation.md).

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
