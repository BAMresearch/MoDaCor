# IO source and sink extension

MoDaCor separates data access from processing steps through `IoSource`,
`IoSources`, `IoSink`, and `IoSinks`. New readers and writers should follow the
same contracts used by the built-in HDF, YAML, CSV, and HDF-processing
implementations.

## Source contract

Subclass `modacor.io.io_source.IoSource` and implement the methods your format
supports:

- `get_data(data_key, load_slice=...)`
- `get_data_shape(data_key)`
- `get_data_dtype(data_key)`
- `get_data_attributes(data_key)`
- `get_static_metadata(data_key)`

Built-in examples:

- `src/modacor/io/hdf/hdf_source.py`
- `src/modacor/io/yaml/yaml_source.py`
- `src/modacor/io/csv/csv_source.py`
- `src/modacor/io/tiled/tiled_source.py`

`IoSources` exposes all sources through a shared `ref::path` syntax. For
example:

- `sample::entry1/instrument/detector00/data`
- `defaults::detector/darkcurrent/value`

HDF attributes can be addressed with `@attribute`, for example
`sample::entry1/instrument/detector00/frame_exposure_time@units`.

## Sink contract

Subclass `modacor.io.io_sink.IoSink` and implement:

- `write(subpath, *args, **kwargs)`

Sinks that support incremental assembly may additionally set
`supports_chunked_writes = True` and implement `initialize_chunked(...)`,
`write_chunk(...)`, `inspect_chunked(...)`, and `finalize_chunked(...)`.
Persistent backends should also implement `load_chunked_plan(...)` so a new
server process can reconstruct a handle, and `recover_chunked(...)` for
backend-specific reconciliation, abandonment, and resumption. Ordinary sinks
inherit clear unsupported-capability errors for all optional chunk methods.

`IoSinks` routes writes through `sink_ref::subpath`. The current built-in sink
examples are:

- `src/modacor/io/csv/csv_sink.py`
- `src/modacor/io/hdf/hdf_processing_sink.py`
- `src/modacor/io/hdf/hdf_chunked_processing_sink.py`
- `src/modacor/io/tiled/tiled_sink.py`

## Registration paths

There are three supported ways to add sources or sinks:

1. Register them directly in Python with `IoSources.register_source(...)` or
   `IoSinks.register_sink(...)`.
2. Add them dynamically inside a pipeline with the `AppendSource` and
   `AppendSink` process steps.
3. Build them from normalized runtime specs via
   `modacor.io.runtime_support.build_sources_from_specs(...)` and
   `build_sinks_from_specs(...)`.

The shared CLI/runtime builder currently supports:

- source types: `hdf`, `yaml`, `csv`, `buffer`, `tiled`, and `custom`
- sink types: `csv`, `hdf`, `hdf_chunked`, `hdf_processing`, `buffer`,
  `plotly_json`, `tiled`, and `custom`

`hdf_chunked` currently provides the complete programmatic `BaseData`
lifecycle, including weights, uncertainties, and static or batch-dependent
axes. Destination selections must currently be contiguous. The runtime builder
can construct it, but the server endpoints that coordinate its
initialize/write/finalize operations are not implemented yet; track that work
in `docs/design/chunked-sink-implementation-plan.md`.

For `custom` sources or sinks, trusted/local builders can use
`kwargs.class_path` with the fully qualified class import path. Runtime services
running with the restricted policy reject arbitrary `class_path` imports; for
that mode, register allowed custom classes in the service process and refer to
them with `kwargs.class_alias`.

Runtime-service sink registrations use the same normalized shape as source
registrations:

```json
{
  "ref": "export_csv",
  "type": "csv",
  "location": "/data/out/current.csv",
  "kwargs": {"delimiter": ","}
}
```

HDF sink registrations may opt into available runtime metadata with
`kwargs.include_runtime_metadata`, but this is not enabled by default. The
process-level `write_hdf` path remains the main metadata-rich HDF artifact
export.

## Runtime-service profiles

If you are adding facility-facing runtime workflows, also check
`src/modacor/server/source_profiles.py`. That file defines the named source
profiles exposed by the API and CLI, such as `mouse` and `saxsess`.

## Testing expectations

Add targeted tests under `tests/io/...` for format behavior and under
`tests/modules/...` if the new class is used by `AppendSource` or `AppendSink`.
Current examples include:

- `tests/io/hdf/test_hdf_source.py`
- `tests/io/yaml/test_yaml_source.py`
- `tests/io/csv/test_csv_source.py`
- `tests/modules/base_modules/test_append_source.py`
- `tests/modules/base_modules/test_append_sink.py`

## Practical guidance

- Keep source references stable and descriptive; they become part of pipeline
  configuration.
- Prefer explicit internal paths and metadata keys over implicit defaults.
- If the format only supports a subset of the full interface, raise
  `NotImplementedError` rather than silently guessing.
- Reuse the existing registry and runtime-support helpers instead of creating a
  parallel configuration path.

## Tiled source and sink

Install the optional client with `pip install 'modacor[tiled]'`. Both
`modacor.io.tiled.TiledSource` and `modacor.io.tiled.TiledSink` accept a Tiled
URL, `profile:profile-name`, or an existing client via `root_node=client`.
The runtime builders accept `type: tiled` and preserve the URL. Connection
options, including authentication, go in `connection_kwargs` inside
`iosource_method_kwargs` or `iosink_method_kwargs` (or runtime `kwargs`).
See the [Tiled Python client reference](https://blueskyproject.io/tiled/reference/python-client.html)
for connection and catalog methods.

```python
from modacor.io.tiled import TiledSource, TiledSink

sink = TiledSink(
    sink_reference="corrected",
    resource_location="profile:beamline",
    iosink_method_kwargs={"base_path": "processed"},
)
sink.write("run_001", processing_data, data_paths=["/sample/signal"])

source = TiledSource(
    source_reference="result",
    resource_location="profile:beamline",
    iosource_method_kwargs={"base_path": "processed/run_001"},
)
array = source.get_data("sample/signal/signal")
units = source.get_static_metadata("sample/signal/signal@units")
```

A BaseData path such as `/sample/signal` exports its `signal`, `weights`, and
named `uncertainties` arrays. Numeric leaf paths such as
`/sample/signal/variances/poisson` are also supported. Units are stored in
array metadata under `attrs.units`; nonnumeric leaves such as
`/sample/signal/units` are stored in a metadata container and can be read with
`get_static_metadata("sample/signal/units@value")`. Select BaseData roots or
individual leaves; whole DataBundle roots are not supported by this sink.

The sink needs a writable Tiled catalog and write credentials. It creates
missing containers under the configured base path and rejects existing targets
by default. Set `iosink_method_kwargs={"overwrite": True}` to update existing
arrays with the same shape and dtype. Writes occur one array at a time and
are not transactional; use a distinct run subpath for separate exports.

The source caches full-array reads and metadata locally and returns caller-owned
copies, so in-place processing does not modify later reads. Call
`source.clear_cache()` after server-side updates. Explicit slices bypass the
full-array cache. An empty key or `@attribute` resolves relative to `base_path`.

Install `pip install 'modacor[tiled-tests]'` and run
`python -m pytest tests/io/tiled` for source and sink integration tests against
an in-process Tiled server with temporary storage. This exercises real client
requests and serialization without an external service or listening port.
