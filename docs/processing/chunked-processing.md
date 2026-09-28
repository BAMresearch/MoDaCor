# Chunked processing

Chunking bounds working memory and lets an external orchestrator distribute
large frame series or detector regions. It does not change the scientific
meaning of a pipeline step.

MoDaCor separates:

- a `ChunkPlan`, which defines the complete source and destination layout;
- a `ChunkSpec`, which identifies one source selection and its output
  placements; and
- source/sink capabilities, which determine whether slices can be read or
  written directly.

Two supported delivery patterns are common:

- the runtime reads a configured HDF5 or Tiled source slice; or
- an external orchestrator uploads one chunk through a buffer source.

Each chunk is processed with the normal runner. Its identity can be attached to
the run result and trace. Incremental sinks validate plan hashes, chunk IDs,
shapes, units, and destination slices before writing. Finalization verifies that
the expected chunks are complete.

Chunk size is an operational choice; source selection and physical output
placement are correctness contracts. Edge chunks therefore have explicit
expected shapes instead of relying on a fixed nominal size.

See [Buffers and chunked outputs](../server/buffers-and-chunked-outputs.md) for
server operations and the public classes in the
[I/O API reference](../reference/python-api/io.md).
