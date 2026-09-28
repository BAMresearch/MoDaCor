# I/O capability reference

Runtime registration uses the type names below. `custom` is available only when
the runtime policy permits an approved class alias or import path.

## Sources

| Type | Implementation | Data and attributes | Slice reads | Extra |
| --- | --- | --- | --- | --- |
| `hdf` | `HDFSource` | Yes | Yes | Base |
| `csv` | `CSVSource` | Yes | Yes | Base |
| `yaml` | `YAMLSource` | Yes | Yes for array values | Base |
| `buffer` | `BufferSource` | Yes | Yes | Server-managed memory |
| `tiled` | `TiledSource` | Yes | Yes | `tiled` extra |

## Sinks

| Type | Implementation | Complete writes | Chunk lifecycle | Extra |
| --- | --- | --- | --- | --- |
| `csv` | `CSVSink` | Yes | No | Base |
| `hdf`, `hdf_processing` | `HDFProcessingSink` | Yes | No | Base |
| `hdf_chunked` | `HDFChunkedProcessingSink` | No | Yes | Server workflows |
| `buffer` | `BufferSink` | Yes | No | Server-managed memory |
| `tiled` | `TiledSink` | Yes | No | `tiled` extra |
| `plotly`, `plotly_json`, `visualization` | `PlotlyJSONSink` | Plot payload | No | `plotting` plus server buffer store |

The base `IoSource` exposes data, shape, dtype, attributes, and static metadata.
Unsupported capabilities return the documented neutral value or raise the
explicit capability error. The base `IoSink` rejects chunk operations unless
the implementation advertises `supports_chunked_writes`.

See [Sources](../processing/sources.md), [Sinks](../processing/sinks.md), and the
[I/O extension guide](../development/io-source-sink-guide.md).
