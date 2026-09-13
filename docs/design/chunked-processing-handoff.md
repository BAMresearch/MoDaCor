# Chunked Processing Handoff

Status: design handoff for a future implementation branch.

## Goal

Allow MoDaCor to process datasets that do not fit comfortably in memory,
including:

- detectors with many pixels;
- acquisitions with many frames; and
- datasets that are large in both dimensions.

The implementation should keep memory use proportional to a configured chunk
size while preserving the numerical result, metadata, weights, uncertainties,
masks, and detector geometry of an equivalent whole-dataset run.

This work is separate from the Tiled I/O revival. The Tiled source is ready for
complete and explicitly sliced reads, and the sink is ready for complete-array
writes. What is missing is a pipeline-level mechanism that selects, processes,
combines, and writes chunks.

## Current capabilities

### Source slicing

The `IoSource` contract accepts `get_data(data_key, load_slice=...)`.
`HDFSource`, `TiledSource`, and `BufferSource` implement slicing. For Tiled,
explicit slices are requested from the remote node and bypass the local
full-array cache. A compatibility fallback downloads and slices locally only
when a node does not implement sliced reads.

Shape and dtype inspection are available without a complete read for the main
array sources. These methods can be used to construct a chunk plan before
loading data.

### External chunk processing through buffers

The existing `BufferSource`, `BufferSink`, and runtime buffer API form the
currently working bounded-memory route. An external client can:

1. select and read one source chunk;
2. upload signal, weights, uncertainties, masks, and metadata to a session
   buffer;
3. execute the normal MoDaCor pipeline;
4. download the selected results from the sink buffer;
5. write or assemble those results externally; and
6. replace the buffer contents with the next chunk.

Sink keys have latest-value retention. Partial execution can use
`rollback_snapshot: false` to avoid a deep copy of large `ProcessingData`.
Separate sessions are the supported way to process chunks concurrently.

The full buffer protocol and API example are documented in
[API Buffer Source/Sink](api-buffer-source-sink.md).

### Module-local chunking

Individual numerical kernels may already limit their temporary allocations.
For example, capillary self-absorption calculations have a
`detector_chunk_size`. This reduces intermediate working memory inside that
calculation, but it is not a general file or pipeline chunking mechanism: the
input detector data may still have been loaded in full.

## Pipeline-level support

Ordinary pipeline YAML does not contain execution-specific source intervals.
Chunked server requests can now coordinate slices across exact HDF5 or Tiled
datasets using typed `ChunkPlan.source_bindings`; staged `BufferSource` values
remain already-sliced inputs. See [Chunked Operation](chunked-operation.md) for
the current contract. Direct Python callers can also use source slicing.

MoDaCor also has no general:

- graph-native loop or scheduler construct;
- lazy or Dask-backed `BaseData` representation;
- reducer protocol for combining chunk-level statistics.

Server-managed chunking now provides normalized source work, a shared chunk
descriptor, incremental HDF5 reassembly, retry/resume, completion manifests,
and finalization. A `ProvisionalChunkPlan` can discover HDF5/Tiled source
extents and use the first processed result to resolve output schema. This is an
external execution lifecycle rather than a loop embedded in the processing
graph.

`TiledSink` writes complete arrays one at a time. It can overwrite an existing
array only when shape and dtype are unchanged; it does not append, resize, or
publish a multi-array result atomically. Its full writes should therefore not
be mistaken for a chunk assembly API.

## Chunking strategies

### 1. Frame chunks

Frame batching slices leading, non-detector dimensions while retaining the
complete detector image in every chunk. This is the recommended first target.
`rank_of_data` already defines the trailing dimensions that describe one data
item, so a signal shaped `(frames, slow, fast)` with `rank_of_data: 2` can be
chunked along `frames` without changing the detector coordinate system.

Frame chunks are straightforward when each frame produces an independent
result. The client or sink can concatenate those results in frame order.
Metadata and static detector geometry can be reused, while frame-dependent
arrays must be sliced with the signal.

The current `IndexedAverager` is a specific compatibility concern. It requires
signal, Q, Psi, pixel index, and mask arrays to have identical shapes and then
flattens them. A two-dimensional geometry map is not automatically broadcast
over a batch of frames. Initially, run this stage one frame at a time or change
it to understand leading batch dimensions.

### 2. Spatial detector tiles

Spatial tiling slices one or more trailing detector dimensions. It is needed
when a single detector frame or a calculation's temporaries are too large.
This requires more coordination than frame batching:

- every pixel-dependent array must use the same tile;
- geometry calculations must preserve the tile's global pixel offsets;
- algorithms using neighboring pixels may require a halo around the tile;
- masks and corrections must be cropped or broadcast consistently; and
- output tiles need deterministic placement in the destination array.

Spatial tiles are safe for purely pixel-local corrections. Global operations
such as normalization, background statistics, median filters, outlier
detection, and integration may require a preliminary pass, halos, or a reducer.

### 3. Hierarchical chunks

Datasets with many large frames can use a frame batch containing one spatial
tile at a time. This combines the rules above and should follow only after both
individual strategies have explicit dimension semantics and tests.

### 4. Lazy arrays

Dask or another lazy array implementation could build a task graph from source
chunks. Most MoDaCor modules currently use eager NumPy operations and assume
ordinary `ndarray` values, so adopting lazy arrays would touch `BaseData` and a
large part of the module library. This is a possible longer-term direction,
not the recommended first implementation.

## Aggregation and numerical correctness

Chunk results can be concatenated only when outputs are independent along the
chunked dimension. Reductions need mergeable intermediate state. For indexed
weighted averaging, useful sufficient statistics include:

- `sum_w`;
- `sum_wx` for signal;
- `sum_wq` for Q;
- `sum_wcos` and `sum_wsin` for circular Psi averages;
- sample counts and valid counts; and
- the corresponding uncertainty accumulators required by each configured
  propagation rule.

Averaging already averaged chunk outputs generally gives the wrong answer,
especially when chunks contain different counts or weights. Reducer state must
be merged and finalized only after all chunks have contributed. Operations
whose exact state cannot be merged may require two passes or an explicit
approximation policy.

## Implementation options

### External orchestration with the existing buffer API

This works now and is the lowest-risk path for an early large-data workflow.
The external process owns file slicing, chunk order, retries, concurrency, and
final assembly. MoDaCor processes one complete chunk at a time through the
ordinary pipeline.

Advantages are bounded memory and limited changes to the core. Costs are extra
buffer transfer and the need to maintain orchestration outside MoDaCor.

### Expose source slices in pipeline configuration

Add a safe slice or chunk descriptor to `AppendProcessingData` and pass it to
every relevant `IoSources.get_data()` call. This lets HDF and Tiled read only
the requested interval and avoids staging input through the buffer API.

A single shared descriptor should drive signal, weights, uncertainties, masks,
and frame-dependent metadata. It should use structured integers and ranges;
do not evaluate Python slice expressions from YAML. This option still needs an
external loop unless accompanied by a native runner.

### Native chunk runner and incremental sinks

A MoDaCor runner could inspect source shapes, create chunk descriptors, execute
the pipeline repeatedly, and publish results. This provides the best integrated
experience but requires clear contracts for chunk-aware modules, resumability,
result assembly, reducer state, and atomic completion.

Incremental HDF or Tiled output should distinguish these operations:

- allocate a destination with final shape and dtype;
- write a known destination slice idempotently;
- append along a declared extensible dimension; and
- finalize or publish a completed dataset.

These operations do not fit the current generic `IoSink.write()` method and
should be designed as an explicit optional capability rather than inferred
from repeated whole-array writes.

## Recommended development sequence

1. Define a structured `ChunkSpec` containing a chunk id, source slices,
   destination slices, full shape, and which axes are batch versus data axes.
2. Implement frame slicing in `AppendProcessingData`, applying one shared
   chunk consistently to signal, weights, uncertainties, masks, and other
   frame-dependent inputs.
3. Add a sequential frame-chunk runner. Use existing full-shape and dtype
   inspection to build the plan without materializing the source array.
4. Add an output collector for independent per-frame results. Start with the
   existing buffer API or an HDF destination-slice writer.
5. Make `IndexedAverager` batch-aware or introduce an explicit reducer state,
   depending on whether the desired output is one result per frame or one
   result over all frames.
6. Add retry/resume metadata and a completion marker before enabling parallel
   workers.
7. Implement spatial tiling for a small set of verified pixel-local modules,
   including global pixel offsets and halo declarations.
8. Consider hierarchical chunks and lazy arrays only after the eager chunk
   contracts are stable.

Frame chunking should be delivered as a narrow vertical slice before designing
every possible scheduler and backend. It exercises source slicing, coordinated
inputs, pipeline execution, and result assembly while avoiding most geometry
boundary problems.

## Decisions for the new branch

The next discussion should settle these points before fixing the public API:

- Is the first use case independent processing of every frame, or a global
  reduction over many frames?
- Should the first runner be internal to MoDaCor, or should the existing
  external buffer orchestration remain the scheduler?
- Which source and destination pair is required first: HDF-to-HDF,
  Tiled-to-Tiled, or buffer-mediated processing?
- Should chunk size be configured as a frame count, a byte budget, or both?
- Which metadata vary by frame, and which arrays may be reused across chunks?
- What retry and idempotency guarantees are required?
- Is spatial tiling required in the first milestone, or can it follow frame
  batching?

## Initial acceptance criteria

- Peak data memory is proportional to the configured chunk size rather than
  the full dataset size.
- A representative frame-chunked pipeline agrees numerically with a complete
  in-memory run.
- Signal, weights, uncertainties, masks, units, axes, and `rank_of_data` remain
  consistent.
- Tiled and HDF sliced reads do not materialize the complete source array.
- Output order and destination placement are deterministic.
- Failed chunks can be retried without duplicating or corrupting completed
  output.
- Reducer-based results are independent of chunk size within the documented
  floating-point tolerance.

## Suggested prompt for the next chat

> Read `docs/design/chunked-processing-handoff.md` and the linked buffer design.
> Create a new branch for chunked processing. Start by reviewing the current
> source slicing, `AppendProcessingData`, `IndexedAverager`, and sink contracts.
> Propose and implement the smallest frame-chunking vertical slice, with tests
> showing bounded reads and agreement with whole-dataset processing. Keep
> spatial detector tiling and lazy-array support out of the first milestone
> unless the code review shows they are required.
