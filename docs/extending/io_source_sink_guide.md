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

`IoSinks` routes writes through `sink_ref::subpath`. The current built-in sink
examples are:

- `src/modacor/io/csv/csv_sink.py`
- `src/modacor/io/hdf/hdf_processing_sink.py`

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

- source types: `hdf`, `yaml`, `csv`, and `custom`
- sink types: `csv`

For `custom` sources, the runtime spec must include `kwargs.class_path` with
the fully qualified class import path.

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
