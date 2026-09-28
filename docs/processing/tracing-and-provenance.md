# Tracing and debugging

`PipelineTracer` records lightweight per-step dataset probes by default:
shape, units, dimensionality, rank, NaN counts, and optional min/max values.
These probes are serialized into the HDF trace tree under
`/processing/tracer/<run>/steps/<step>/datasets`.

Full `ProcessingData` snapshots are separate and opt-in because they can be
large. Enable them only for debugging runs where the intermediate arrays matter:

```bash
modacor run \
  --pipeline pipeline.yaml \
  --trace \
  --trace-watch sample:signal,Q \
  --trace-snapshot-processing-data \
  --trace-snapshot-step DC_bg \
  --write-hdf debug.h5 \
  --write-all-processing-data
```

If no `--trace-snapshot-step` is supplied, MoDaCor captures a snapshot after
every executed step. HDF exports store snapshots at:

```text
/processing/tracer/<run>/steps/<step>/processing_data
```

The final result remains under `/processing/result/<run>`, and the file-level
NeXus `default` chain points there rather than to an intermediate snapshot.
Snapshot arrays are compressed with HDF5 `lzf` by default; set
`tracer_processing_data_compression` in the HDF sink kwargs to override this.
