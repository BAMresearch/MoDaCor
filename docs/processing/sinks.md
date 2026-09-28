# I/O sinks

`IoSinks` registers named output adapters independently from the processing
graph. Steps can publish selected values through a configured sink, while the
runner and runtime service can export final `ProcessingData` and provenance.

Current sinks cover CSV, HDF5 processing results, in-memory buffers, optional
Tiled output, and Plotly JSON live output. Some sinks support incremental or
chunked lifecycle operations; others accept complete results only. Consult the
[I/O capability matrix](../reference/io-capabilities.md) before choosing an
operational pattern.

`AppendSink` registers a sink from configuration. `SinkProcessingData` selects
processing paths for output. CLI and server registrations are often preferable
when the same pipeline should write to different destinations in different
runs.

HDF exports can include the pipeline specification, YAML, trace events, and
selected processing results. Treat output data and provenance as one
reproducibility record.
