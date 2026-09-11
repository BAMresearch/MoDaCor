# Chunked Beamline Validation

This chapter defines the Phase 5 validation workflow for chunked HDF5 output.
Representative beamline files are intentionally external to the MoDaCor
repository. The repository contains only the harness, small deterministic
tests, and the expected report contract.

## What can run without representative data

The lightweight suite covers manifest interruption, retry, restart
reconstruction, missing chunks, duplicate delivery, abandoning and resuming an
assembly, and serialization of two server output ids targeting the same HDF5
file. These tests use small generated arrays and run in normal CI.

The optional RSS test runs two fresh benchmark subprocesses with the same chunk
shape and an eightfold difference in total frame count:

```bash
MODACOR_RUN_CHUNK_MEMORY_TESTS=1 \
  .venv-dev/bin/python -m pytest tests/integration/test_chunked_memory.py -q
```

It measures process peak RSS rather than Python allocations, so NumPy and HDF5
native allocations are included. Its generous fixed allowance makes it a
regression guard, not a replacement for measurement with realistic detector
data.

## Benchmark harness

`scripts/benchmark_chunked_hdf.py` exercises the chunked sink without loading a
complete source into memory. It supports generated arrays and slice-by-slice
reads from an external HDF5 or NeXus dataset.

A quick generated run is:

```bash
.venv-dev/bin/python scripts/benchmark_chunked_hdf.py \
  --shape 24,1679,1475 \
  --chunk-axis 0 \
  --chunk-size 2 \
  --storage-chunks 1,256,256 \
  --compression gzip \
  --compression-level 1 \
  --output /tmp/modacor-chunk-benchmark.h5 \
  --report /tmp/modacor-chunk-benchmark.json
```

To exercise an external file:

```bash
.venv-dev/bin/python scripts/benchmark_chunked_hdf.py \
  --source-hdf /data/i22/example.nxs \
  --source-dataset entry/instrument/detector/data \
  --chunk-axis 0 \
  --chunk-size 2 \
  --output /scratch/modacor/i22-identity-assembly.h5 \
  --report /scratch/modacor/i22-identity-assembly.json
```

The current harness measures identity assembly, not pipeline computation. It
validates the result slice by slice and reports:

- MoDaCor version and plan id/hash;
- source identity, shape, dtype, and logical byte count;
- chunk axis, size, count, and delivery order;
- actual HDF5 chunk and compression properties;
- initialization, writing, finalization, validation, and total time;
- logical write throughput and output file size; and
- baseline and peak process RSS.

Run each storage configuration in a fresh subprocess. Peak RSS is a lifetime
maximum, so comparing several configurations inside one process would retain
the largest earlier value. Use a new output path for each run, or pass
`--overwrite` deliberately. The harness refuses to overwrite its source file
or replace the HDF5 output with the JSON report.

## Restart and recovery operations

Opaque output ids are process-local handles. The HDF5 plan, manifest, run
subpath, and staged pipeline provenance are persistent. After a server restart,
reconstruct a handle with:

```text
POST /v1/chunked-outputs/reopen
```

The request supplies the chunk-capable sink registration, `plan_id`, and an
optional expected `plan_hash`. The server reads and validates the stored plan
and returns a new output id. It does not need a worker session or the original
complete plan document.

Recovery uses:

```text
POST /v1/chunked-outputs/<output_id>/recover
```

with one of these actions:

- `reconcile`: change stale `writing` chunks/components to `failed`, return an
  interrupted `finalizing` plan to `writing`, and retain chunk placement
  ownership for safe retry;
- `abandon`: persistently stop writes to an incomplete assembly; or
- `resume`: return an explicitly abandoned assembly to `writing`.

A chunk marked `failed` is retried by submitting the same `ChunkSpec`; every
component is rewritten before the manifest returns to `complete`.
Initialization interrupted before layout creation is not reconciled because
the fixed dataset schema may be incomplete. Inspect it and explicitly
initialize with `collision: replace` if discarding that partial assembly is
appropriate.

`DELETE /v1/chunked-outputs/<output_id>` only detaches the in-process handle.
It never removes datasets or files and is recoverable through `reopen`. File
deletion is intentionally outside the runtime API because an HDF5 file may
contain raw data, unrelated results, or other chunk plans.

## Work that remains data- or facility-dependent

After publishing the representative data collection, add a versioned case
manifest with the Zenodo DOI, checksums, source dataset paths, expected output
paths, shapes, dtypes, units, and relevant axes. The normal test suite should
not download it automatically.

The remaining validation must then:

1. compare whole-array and chunked pipeline results numerically and structurally;
2. measure RSS and throughput for representative I22 SAXS and WAXS cases;
3. sweep chunk shapes and compression settings on the intended storage system;
4. repeat interrupted-run recovery against copies of the representative files;
5. validate HDF5 locking with the deployed library and filesystem; and
6. record the chosen operational defaults and performance evidence.

The in-process server lock is deliberately the supported concurrency boundary
for now. Multiple server processes or replicas still require a shared lock or a
single designated HDF5 writer and must not be inferred safe from the local
threaded test.
