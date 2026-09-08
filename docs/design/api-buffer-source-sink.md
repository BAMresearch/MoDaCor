# API Buffer Source/Sink

Status: implementation plan for the keyed in-memory buffer MVP.

## Goal

MoDaCor should be able to run as a correction engine for externally chunked
data while keeping the normal `IoSource` and `IoSink` contracts. The external
pipeline remains responsible for unpacking, chunk selection, retry policy, and
final repacking. MoDaCor receives the data currently available in a session
buffer, processes it through a normal pipeline, and writes selected results back
to a session buffer.

## Data Model

The runtime buffer is session-scoped and keyed by:

```text
(session_id, kind, ref, data_key)
```

where `kind` is either `source` or `sink`, `ref` is the registered source/sink
reference, and `data_key` is the same key passed through `IoSource.get_data()`
or `IoSink.write()` subpaths.

Array entries are stored internally as `numpy.ndarray`. Metadata and attributes
are JSON-compatible Python values. Source keys are exact keys, for example:

```text
sample/signal/signal
sample/signal/weights
sample/signal/uncertainties/poisson
sample/signal@units
sample/mask/signal
sample/Q/signal
sample/Psi/signal
```

## Wire Format

The MVP supports `.npy` for array upload/download and JSON for metadata.
`.npy` preserves array shape, signed/unsigned integer dtypes, floating dtypes,
and array order metadata.

JSON/base64 array transfer is intentionally deferred. The buffer store and
buffer IO classes only work with `numpy.ndarray`; codecs live at the API
boundary so additional wire formats can be added later.

## Runtime Registration

Buffer IO is registered like other runtime IO:

```json
{
  "sources": [
    {"ref": "chunk_input", "type": "buffer", "location": "buffer://session"}
  ],
  "sinks": [
    {"ref": "chunk_output", "type": "buffer", "location": "buffer://session"}
  ]
}
```

The `location` value is only a placeholder for registration parity. The actual
storage is the runtime session buffer.

## Example Pipeline

```yaml
steps:
  load_signal:
    module: AppendProcessingData
    requires_steps: []
    configuration:
      processing_key: sample
      databundle_output_key: signal
      signal_location: "chunk_input::sample/signal/signal"
      units_location: "chunk_input::sample/signal/signal@units"
      weights_location: "chunk_input::sample/signal/weights"
      uncertainties_sources:
        poisson: "chunk_input::sample/signal/uncertainties/poisson"
      rank_of_data: 2

  load_mask:
    module: AppendProcessingData
    requires_steps: []
    configuration:
      processing_key: sample
      databundle_output_key: mask
      signal_location: "chunk_input::sample/mask/signal"
      units_override: dimensionless
      rank_of_data: 2

  export:
    module: SinkProcessingData
    requires_steps:
      - corrected_step
    configuration:
      target: "chunk_output::current"
      data_paths:
        - /sample/corrected
```

Masks remain separate `BaseData` entries because `BaseData` has no `mask`
attribute and existing modules look for `mask`/`Mask` entries in a
`DataBundle`. Generic axes attachment is deferred for the MVP; axes can be
loaded as separate `BaseData` entries.

## API

Array upload:

```text
PUT /v1/sessions/{session_id}/buffers/sources/{source_ref}/arrays/{data_key:path}
Content-Type: application/x-npy
```

Array attributes:

```text
PUT /v1/sessions/{session_id}/buffers/sources/{source_ref}/attrs/{data_key:path}
Content-Type: application/json
```

Standalone metadata:

```text
PUT /v1/sessions/{session_id}/buffers/sources/{source_ref}/metadata/{data_key:path}
Content-Type: application/json
```

The metadata payload is an object containing `value`.

Sink array fetch:

```text
GET /v1/sessions/{session_id}/buffers/sinks/{sink_ref}/arrays/{data_key:path}
Accept: application/x-npy
```

Manifest:

```text
GET /v1/sessions/{session_id}/buffers/{kind}/{ref}/manifest
```

Clear:

```text
DELETE /v1/sessions/{session_id}/buffers
DELETE /v1/sessions/{session_id}/buffers/sinks/{sink_ref}
DELETE /v1/sessions/{session_id}/buffers/sinks/{sink_ref}/arrays/{data_key:path}
```

The general clear endpoint supports optional `kind`, `ref`, and `data_key`
query parameters.

## Memory Behavior

The sink buffer uses latest-only retention. Writing the same sink key overwrites
the previous value.

Chunk clients should call `/process` with:

```json
{
  "mode": "partial",
  "changed_keys": ["sample.signal"],
  "rollback_snapshot": false
}
```

This avoids deep-copying large chunk-derived `ProcessingData` during partial
execution. Existing clients keep the previous rollback behavior because the
default remains `rollback_snapshot: true`.

## Limitations

- Buffer data is in-process memory only and is lost on server restart.
- MoDaCor does not schedule chunks or repack final arrays.
- One active run per session remains unchanged.
- Parallel chunk processing should use separate sessions in the MVP.
- JSON/base64 array transfer and generic axes attachment are deferred.
