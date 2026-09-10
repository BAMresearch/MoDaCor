# Chunked Operation

Status: recommended design for externally orchestrated chunk processing.

## Scope

MoDaCor's array sources already support explicit reads through
`IoSources.get_data(..., load_slice=...)`. In particular, `HDFSource`,
`TiledSource`, and `BufferSource` accept NumPy-compatible integer and slice
selectors. HDF and Tiled can perform the selection at the storage backend, so
the caller does not need to materialize the complete source array and slice it
locally.

MoDaCor does not currently schedule those reads or pass a slice from ordinary
pipeline YAML into `AppendProcessingData`. An external runner should plan the
slices and request them through `IoSources`, supply the resulting chunk to the
normal pipeline, and assemble the results. It should not need to implement
HDF5 or Tiled slicing itself. MoDaCor should receive the resolved chunk
description so traces and outputs retain enough information to identify and
reproduce the operation.

This design initially covers chunks along non-image, or batch, dimensions.
For an array shaped `(measurement, frame, slow, fast)` with
`rank_of_data: 2`, axes 0 and 1 are batch axes and axes 2 and 3 are data axes.
Spatial detector tiling is a separate extension because it requires geometry
offsets, halos, and module-specific correctness rules.

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
- `bindings`: other inputs and how the driver selection projects onto them;
- `total_chunks`: number of generated chunks;
- `expected_chunk_ids`: stable chunk ids in ordinal order, used to initialize
  and validate the completion manifest; and
- `plan_hash`: hash of the canonical plan representation.

Bindings should use explicit roles:

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

The external runner supplies already sliced arrays and passes the corresponding
`ChunkSpec` as execution metadata. MoDaCor should not reinterpret the spec to
perform another slice.

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

Retries use the same `ChunkSpec`. The result collector should place or replace
that chunk idempotently using its output placements. Global reductions need
mergeable reducer state; averaging already averaged chunk results is not a
correct substitute.

Explicit HDF source slices bypass the complete-array cache, so sequential
external reads do not accumulate all previously processed chunks. Complete
reads remain cached for reusable static data.

The proposed output capability and HDF5 implementation are specified in
[Chunked Sink Implementation Plan](chunked-sink-implementation-plan.md).
Parallel orchestration is documented separately in
[External Parallel Runner](external-parallel-runner.md).
