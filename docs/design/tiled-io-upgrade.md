# Tiled I/O Upgrade

This document records the revival of Tiled-backed I/O after the
`Tiled-IoSource` branch was updated to the September 2026 MoDaCor architecture.
It describes the implemented contract, verification, completed compatibility
work, and the remaining steps before facility deployment.

Status values are `Done`, `Open`, and `Deferred`.

## Scope and current design

The upgrade retains the shared `IoSource` and `IoSink` contracts. Tiled is an
optional backend and is imported only when a URL or profile connection is
created. Supplying an existing client through `root_node` or a
`resource_location` mapping keeps tests and embedded use independent of the
optional package import.

`TiledSource` implements:

- full and sliced NumPy array reads;
- shape and dtype inspection using the Tiled array structure, without first
  downloading the array;
- data attributes and static metadata through `path@attribute` references;
- an optional `base_path`/`base_item_path` prefix;
- URL, `profile:name`, `profile://name`, descriptor mapping, and existing-client
  connections;
- local caches for resolved nodes, complete arrays, attributes, and structure;
- `clear_cache()` for data that may have changed remotely.

Cached arrays and attribute dictionaries are owned by the source. Public reads
return copies, matching `HDFSource`, so in-place pipeline processing cannot
modify later reads from the cache. Explicit slices bypass the full-array cache.

`TiledSink` implements the current ProcessingData sink contract:

- `IoSinks.write_data("sink::subpath", processing_data, data_paths=...)`
  routing;
- export of BaseData `signal`, `weights`, and named `uncertainties`;
- export of numeric leaves as arrays and nonnumeric leaves as metadata;
- units and `rank_of_data` metadata on exported arrays;
- creation of missing containers below an optional base path;
- rejection of existing targets by default;
- opt-in updates with `overwrite=True` when node type, shape, and dtype match.

Both classes are exported from `modacor.io.tiled` and `modacor.io`. Runtime
specs accept `type: tiled` for sources and sinks. Tiled locations remain strings
so URI schemes survive runtime and CLI registration.

## Packaging and tests

The `tiled` extra installs the client needed for ordinary source and sink use.
The `tiled-tests` extra adds the minimal server and its runtime dependencies.
The `all` extra contains both sets of dependencies, so `pip install '.[all]'`
installs every supported optional feature.

Unit tests use small protocol-compatible objects to cover connection parsing,
path handling, caching, metadata, slicing fallback, writes, and overwrite
validation. Integration tests use Tiled's in-process ASGI server and temporary
array storage. They exercise real client serialization without opening a
network port. A dedicated `tox -e tiled` environment installs `tiled-tests` and
runs these tests. CI runs that environment with each active Python 3.12–3.14
matrix interpreter.

## Upgrade items

1. `Done` Rebase onto the current I/O contracts

   The source was reconciled with the current `IoSource`, registry, runtime
   builder, and package exports. A `TiledSink` was added for the current
   ProcessingData-oriented `IoSink.write()` contract.

2. `Done` Current Tiled client compatibility

   Structure dtype inspection now supports Tiled's `data_type.to_numpy_dtype()`
   API. Metadata accepts general read-only mappings. Slice fallback applies the
   requested slice locally when a compatible node lacks remote slicing.
   Logging calls now match MoDaCor's `MessageHandler` interface.

3. `Done` URI-safe CLI registration

   Session `set-source`, `set-sink`, and `set-sample` locations are parsed as
   strings. Filesystem backends still convert their locations in the runtime
   builder, while Tiled URLs and profiles remain unchanged.

4. `Done` Cache ownership

   Full reads cache an internal copy and return a separate array. Repeated reads
   and attribute reads also return copies. This prevents a processing step from
   altering the source's view through an in-place NumPy operation.

5. `Done` Real Tiled coverage in CI

   The focused tox environment includes the full integration-test dependency
   set and is run by every Python 3.12–3.14 CI job. Import or server
   compatibility failures therefore fail CI instead of silently skipping all
   real-server coverage.

6. `Open` Facility connection and authorization validation

   Test at least one real facility deployment using its actual URL/profile,
   TLS configuration, authentication provider, token refresh behavior, and
   write permissions. Credentials belong in the Tiled profile or deployment
   secret mechanism; they should not be committed to pipeline YAML or passed in
   recorded command output.

7. `Open` Representative scale and cache policy

   Test representative detector arrays and repeated pipeline execution. The
   current full-read cache has no size or entry limit. Decide whether facility
   workflows should disable it, bound it, or rely on explicit `clear_cache()`.
   Sliced reads already avoid populating that cache. The I22 notebook now
   contains an opt-in four-measurement SAXS/WAXS workload served by a local
   Tiled service; its one-chunk SAXS production-path smoke check passes, while
   the complete scale and facility-deployment measurements remain open.

8. `Open` Multi-array write atomicity

   A sink call validates all ProcessingData paths before writing, but Tiled
   creates arrays one at a time. A connection failure can leave part of an
   export present. Prefer a unique run subpath for each export. If consumers
   require atomic visibility, add a completion marker or a facility-specific
   staging-and-publish protocol.

9. `Deferred` Resizing and replacement semantics

   Overwrite currently updates only arrays with the same shape and dtype. It
   does not delete or replace nodes. This conservative behavior avoids
   accidental remote data loss. Add explicit replacement only when a concrete
   workflow defines retention, authorization, and failure recovery.

10. `Deferred` Non-array Tiled structures

    MoDaCor's source contract currently returns NumPy arrays and the sink is
    designed around BaseData arrays. Native Tiled tables, sparse arrays, and
    other structure families remain outside this implementation until the core
    I/O contract has a use case for them.

## Validation record

- 2026-09-08: Tiled 0.2.18 source reads and source/sink round trips passed
  against an in-process Tiled server.
- 2026-09-08: Focused I/O, append-source, append-sink, sink-processing,
  runtime-server, and pipeline-runner tests passed after the initial revival.
- 2026-09-08: CLI URI preservation and cache-copy regression tests were added.
- 2026-09-08: The isolated Python 3.12 `tox -e tiled` environment passed all
  21 Tiled tests, including both in-process server tests.
- 2026-09-08: Tiled CI coverage was expanded to Python 3.13 and 3.14. Clean
  `tox -e tiled` runs passed all 21 tests on both versions; the environment
  selects the active matrix interpreter through `TOXPYTHON`.
- 2026-09-13: A notebook-owned Tiled 0.2.18 service streamed a real externally
  linked I22 SAXS detector slice through the runtime. One ten-frame production
  chunk completed through both HDFSource and TiledSource and produced matching
  finalized HDF5 results with one trace record each.

The remaining open items require representative facility infrastructure or a
product decision; they do not block local array-based source and sink use.
