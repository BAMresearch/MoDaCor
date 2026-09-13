# Chunked Operation

Status: recommended design for externally orchestrated chunk processing.

## Scope

MoDaCor's array sources already support explicit reads through
`IoSources.get_data(..., load_slice=...)`. In particular, `HDFSource`,
`TiledSource`, and `BufferSource` accept NumPy-compatible integer and slice
selectors. HDF and Tiled can perform the selection at the storage backend, so
the complete source array does not need to be materialized and sliced locally.

Ordinary pipeline YAML remains free of execution-specific slices. For chunked
requests, however, a `ChunkPlan` can now declare typed source bindings. The
server projects the submitted `ChunkSpec.source_selection` through those
bindings and applies the resolved selectors when the pipeline reads registered
HDF5 or Tiled inputs. The buffer route remains available when an external
runner supplies arrays that are already sliced. In both modes MoDaCor receives
the resolved chunk description so traces and outputs retain enough information to
identify and reproduce the operation.

This design initially covers chunks along non-image, or batch, dimensions.
For an array shaped `(measurement, frame, slow, fast)` with
`rank_of_data: 2`, axes 0 and 1 are batch axes and axes 2 and 3 are data axes.
Spatial detector tiling is a separate extension because it requires geometry
offsets, halos, and module-specific correctness rules.

## Source-ingestion modes

Chunked operation must support two complementary deployment modes. They share
the same `ChunkPlan`, `ChunkSpec`, pipeline execution, chunked-output, retry,
and finalization contracts; only ownership of the source read differs.

### Externally staged chunks with `BufferSource`

The external runner reads and slices the source, then uploads the resolved
signal and every chunk-dependent companion array through the runtime buffer
API. The session registers that logical input as a `BufferSource`. MoDaCor
must treat these arrays as already sliced and must not apply the
`ChunkSpec.source_selection` a second time.

This is the appropriate mode when the runner owns source credentials, the
server cannot access the raw storage, a nonstandard source needs custom
decoding, or chunks are sent to the server over HTTP. It also makes arbitrary
source-specific preprocessing possible before submission.

The costs are HTTP serialization, additional memory copies, and external
responsibility for keeping signal, weights, uncertainties, normalization data,
and other frame-dependent inputs aligned. A buffer registration does not by
itself identify an immutable source file or catalog revision, so the runner
must include that identity in plan and execution provenance.

Replacing a buffer value gives bounded latest-value retention within the
session, but the runner must still bound queued uploads and release any
client-side copies. Partial reruns should identify the changed buffer source so
that reusable static branches are not recalculated.

### Server-side reads with `HDFSource` or `TiledSource`

In a pull deployment, the server registers the original HDF5 or Tiled source
and reads only the source selections needed for the current chunk. This avoids
uploading detector arrays through the client and naturally associates reads
with a file path, dataset path, or Tiled node and revision. It is the preferred
mode when the server has direct storage access and suitable credentials.

The low-level source operations already support these reads. Explicit
`HDFSource` slices bypass its complete-array cache, and Tiled requests slices
from the remote node and bypasses its local full-array cache. Complete reads
remain cacheable for genuinely static inputs such as compact calibration data.
Unchanged HDF5 and Tiled registrations reuse their source instances across
session runs, including the Tiled connection; re-registration invalidates that
session cache.

Chunked server execution now binds a process request's `ChunkSpec` to exact
source datasets declared in `ChunkPlan.source_bindings`. This is deliberately
request-scoped: it does not mutate pipeline YAML or source registrations, and
ordinary processing remains unchanged. The HDF5 path is covered by an
end-to-end server test. A deterministic server test also verifies that Tiled
receives a backend slice and does not populate its full-array cache; validation
against a representative deployed Tiled service remains outstanding.

### Server-side slice-binding contract

The implemented contract:

- accept structured selectors in the process request without evaluating Python
  slice expressions;
- resolve the driver selection through the plan's `aligned`, `static`, and
  `explicit` input bindings, producing one effective selector per registered
  source dataset;
- validate ranks, bounds, edge-chunk shapes, plan identity, and ambiguous axis
  mappings before reading data;
- apply selectors at source-read time without mutating pipeline YAML or
  re-registering an `IoSource` for every chunk;
- mark the affected source references or processing keys as changed so partial
  execution invalidates the correct dependency subgraph;
- preserve cache policy: explicit HDF5 and Tiled slices are not accumulated,
  while explicitly static complete reads may be reused;
- persist the resolved source reference, type, location, dataset or node, and
  selection alongside the chunk execution record; and
- distinguish direct-source inputs from already staged `BufferSource` inputs,
  preventing accidental double slicing.

Bindings are plan-wide because their projection rules do not change between
chunks. A driver and two companion inputs can be declared as:

```yaml
driver:
  source: sample::/entry/data
  full_shape: [1, 100, 1679, 1475]
source_bindings:
  - source_ref: sample
    data_key: /entry/data
    role: aligned
  - source_ref: sample
    data_key: /entry/monitor
    role: explicit
    axis_map: [0, 1]
  - source_ref: mask
    data_key: /entry/mask
    role: static
```

`aligned` requires the input rank to equal the driver rank and applies the
driver selectors unchanged. `explicit.axis_map` has one entry per source axis;
each integer selects the corresponding driver axis and `null` leaves that
source axis complete. `static` records the relationship but applies no slice.
The driver dataset itself must have exactly one `aligned` binding. Bindings use
exact source references and dataset keys rather than shape inference.

Non-static bindings currently accept registered `hdf` and `tiled` sources.
Targeting a `buffer` source is rejected because its values must already be
sliced before upload. The server adds the affected source references to partial
run invalidation automatically and records every effective selector in the run
and chunk execution metadata. Ordinary, non-chunked sessions do not require a
`ChunkPlan` or additional source configuration.

## Separate selection from chunk size

An axis rule has two independent purposes:

- `start`, `stop`, and optional `stride` select source indices; and
- `chunk_size` sets the maximum number of selected indices in one chunk.

`stride` has exactly the meaning of `step` in a Python or NumPy slice. It is
the distance between selected source indices and defaults to 1. It is not the
distance between chunk starts and normally should be omitted. The planner
advances chunk boundaries automatically according to `chunk_size`.

For example, `start: 1`, `stop: 600`, and `chunk_size: 30` produces contiguous
selections `1:31`, `31:61`, and so on. Setting `stride: 5` instead first selects
indices `1, 6, 11, ...` and then partitions those selected indices into chunks.
The name `stride` avoids confusing this source-selection behavior with either
a pipeline process step or chunk advancement.

```yaml
driver: sample::/entry1/detector/data
rank_of_data: 2
batch_axes:
  0:
    index: 0
  1:
    start: 1
    stop: 600
    chunk_size: 30
```

For a source shaped `(outer, frame, slow, fast)`, this fixes axis 0 at index 0
and selects axis 1 from 1 through 599 in chunks containing at most 30 frames.
The data axes are read completely. A simpler `0:20` selection in chunks of 5
is expressed as:

```yaml
batch_axes:
  0:
    start: 0
    stop: 20
    chunk_size: 5
```

Omitted batch axes and all data axes default to a complete selection. Integer
selection removes an axis; a one-element slice preserves it. Negative indices
and open stops may be accepted in user configuration, but the planner should
normalize them to non-negative, bounded values before creating a `ChunkSpec`.

`chunk_size` counts selected elements, not the source-coordinate span. For
example, `stride: 5` and `chunk_size: 20` reads 20 selected elements spanning
up to 100 source positions. Resolved `ChunkSpec` slice records include the
effective stride, including the default value 1, for unambiguous provenance.

## Edge chunks

The final chunk is allowed to contain fewer than `chunk_size` elements. Given
20 selected elements and `chunk_size: 6`, the actual chunk extents are
`6, 6, 6, 2`.

Validation must calculate an actual shape from the normalized selectors. It
must not require every chunk to equal the nominal chunk size. For every
chunked axis:

```text
0 < actual_extent <= chunk_size
```

Non-chunked dimensions must match their resolved selections exactly. The
external runner should validate the array it supplies against this actual
shape before starting MoDaCor.

## `ChunkPlan` and `ChunkSpec`

Static information should not be repeated in every chunk record. A
`ChunkPlan` stores the source layout and planning rules once; each `ChunkSpec`
stores one resolved unit of work.

### ChunkPlan

Required plan fields are:

- `schema_version`: version of the serialized contract;
- `plan_id`: stable human-readable identifier;
- `driver`: logical source reference, full shape, dtype, and `rank_of_data`;
- `batch_axes` and `data_axes`: original source-axis numbers;
- `axis_rules`: normalized selection and partition rules;
- `bindings`: optional free-form planning provenance retained for compatibility;
- `source_bindings`: optional typed, executable mappings from the driver
  selection to exact registered source datasets;
- `total_chunks`: number of generated chunks;
- `expected_chunk_ids`: stable chunk ids in ordinal order, used to initialize
  and validate the completion manifest; and
- `plan_hash`: hash of the canonical plan representation.

Executable source bindings use explicit roles:

- `aligned`: project applicable driver batch selectors onto the input;
- `static`: always read the complete input; or
- `explicit`: use a separately declared axis mapping.

Shape inference may propose these roles, but ambiguous shapes must require an
explicit binding. Shape alone cannot determine which driver axis corresponds
to an input shaped `(100,)`.

Output layouts and the rules that project a chunk onto each stored array are
kept in the plan when destination placement is known before execution. This
avoids repeating static shape, dtype, units, and component-mapping information
in every chunk record.

Each output layout also carries ordered `axis_names`, with `.` for signal
dimensions that do not have a named axis. Array layouts for `axes/<name>` carry
the axis dataset's own `units` and `rank_of_data`; signal and uncertainty units
remain output-level invariants. This separates axis identity and metadata from
the `PlacementBinding` that controls how its values are placed.

### ChunkSpec

A compact chunk record contains:

```yaml
schema_version: "1.0"
plan_id: i22-978609-saxs
plan_hash: sha256:...
chunk_id: c000001
ordinal: 1
grid_index: [0, 1]
source_selection:
  - {kind: index, value: 0}
  - {kind: slice, start: 31, stop: 61, stride: 1}
  - {kind: all}
  - {kind: all}
expected_input_shape: [30, 1679, 1475]
placements:
  - output_id: corrected_signal
    destination_selection:
      - {kind: index, value: 0}
      - {kind: slice, start: 30, stop: 60, stride: 1}
      - {kind: all}
      - {kind: all}
    expected_shape: [30, 1679, 1475]
```

Supported selector records are:

```yaml
{kind: all}
{kind: index, value: 0}
{kind: slice, start: 31, stop: 61, stride: 1}
```

`expected_input_shape` is the actual input shape after applying
`source_selection`; it is therefore smaller for an edge chunk when necessary.
Each placement identifies one declared output and its expected chunk-result
shape. Output-keyed placements are required because pipeline outputs may have
different shapes. The implemented assembly contract requires one placement
for every declared output. A future output kind may make placement optional
for independent per-chunk artifacts that are not part of an assembled array.

Weights, uncertainties, and axes do not each need to repeat selectors in the
`ChunkSpec`. Their static output-layout records in the `ChunkPlan` define how
the signal placement projects onto those components. Ambiguous components
require an explicit axis mapping in the plan.

`chunk_id` must be stable across retries. Attempt number, worker identity,
timestamps, status, errors, and performance measurements belong to an
execution record, not the immutable `ChunkSpec`. Pipeline hashes and source
registrations should likewise be linked run metadata instead of being copied
into every chunk record.

An optional `parent_chunk_id` may be added later for true nested plans without
changing the selection representation.

## Validation

The planner should reject a plan or chunk when:

- a stride is zero or a chunk size is not positive;
- an axis is outside the driver's rank;
- batch and data axes overlap;
- a data axis is sliced without an explicit spatial-tiling opt-in;
- normalized selections exceed the full shape;
- the supplied array differs from the derived actual shape;
- an aligned binding cannot project the driver selection unambiguously;
- chunk identifiers or destination regions overlap unexpectedly; or
- the plan hash referenced by a chunk does not match the active plan.

For a finite plan, validation should also confirm deterministic order and
complete coverage of the requested selection. Complete coverage does not mean
covering indices excluded by `stride`.

## MoDaCor handoff

For `BufferSource`, the external runner supplies already sliced arrays and
MoDaCor does not reinterpret the `ChunkSpec` to perform another slice. For
direct HDF5 or Tiled operation, the runner supplies only the `ChunkSpec`; the
server resolves the plan's typed source bindings and applies them during source
reads.

The recommended server integration puts `output_id` and `chunk_spec` in the
process request. `RunResult`, trace events, and persisted result metadata can
then carry `plan_id`, `chunk_id`, `ordinal`, and `plan_hash`, with the complete
serialized spec stored once per chunk result. After successful processing, the
server passes its in-memory `ProcessingData` directly to the chunk-capable sink
and acknowledges success only after the chunk manifest is durable.

The current HTTP process response does not return complete `ProcessingData`.
Selected arrays can be downloaded through `BufferSink`, but that API is not a
complete transport for all `BaseData` metadata and axis relationships. Chunked
output should therefore not depend on returning result arrays to the
orchestrator. The orchestrator initializes, inspects, and finalizes a
server-level output resource without accessing `ProcessingData` or the storage
backend directly.

This complete-schema workflow is now available through
`POST /v1/chunked-outputs`, `GET /v1/chunked-outputs/<output_id>`, the optional
`chunk_output` process-request field, and
`POST /v1/chunked-outputs/<output_id>/finalize`. The initialized resource owns a
snapshot of the sink registration, so it is not invalidated or redirected by
later worker-session changes. Opaque output ids remain in-process handles,
while the HDF5 plan and chunk manifest are persistent and authoritative. A new
server process can reconstruct a handle with
`POST /v1/chunked-outputs/reopen`; recovery operations can reconcile, abandon,
or resume an incomplete assembly.

Retries use the same `ChunkSpec`. The result collector should place or replace
that chunk idempotently using its output placements. Global reductions need
mergeable reducer state; averaging already averaged chunk results is not a
correct substitute.

Explicit HDF source slices bypass the complete-array cache, so sequential
external reads do not accumulate all previously processed chunks. Complete
reads remain cached for reusable static data.

## Reusable background working set

A source reused unchanged by partial runs is not automatically chunked. In the
I22 correction pipelines, the pilot run reads the complete background detector
stack, applies its normalization and uncertainty steps, and reduces its leading
measurement/frame axes. Later sample-chunk runs reuse that reduced
`ProcessingData` branch.

Consequently, the detector stack and the intermediate arrays needed by the
background branch must fit comfortably in worker memory. The HDF source also
retains the complete raw read in its reusable full-array cache for the life of
the session. Account for that cached array, the processing copy, propagated
uncertainties and masks, and numerical temporaries rather than budgeting only
the on-disk dataset size.

Large backgrounds need an explicit policy. Recommended options are:

- prepare one corrected, reduced background product with a mergeable weighted
  reduction and reuse that compact product; or
- define a scientifically meaningful sample-to-background chunk mapping,
  stage both sources, and invalidate both branches for each run.

Do not combine background chunk means with an unweighted mean unless the
weights, valid counts, masks, and uncertainty propagation make that operation
mathematically equivalent. A full-background aggregate and frame-paired
subtraction are different experimental policies and should remain explicit.

The proposed output capability and HDF5 implementation are specified in
[Chunked Sink Implementation Plan](chunked-sink-implementation-plan.md).
Parallel orchestration is documented separately in
[External Parallel Runner](external-parallel-runner.md).
