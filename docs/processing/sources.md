# I/O sources

`IoSources` is a runtime registry of named external inputs. A registration binds
a reference such as `sample`, `background`, or `calibration` to an `IoSource`
implementation and resource location.

Current source families include HDF5/NeXus, YAML, CSV, in-memory buffers, and
optional Tiled access. Exact capabilities are listed in the
[I/O capability reference](../reference/io-capabilities.md).

Steps address source content with `ref::path`, for example:

```text
sample::/entry/instrument/detector/data
sample::/entry/instrument/detector/data@units
```

Keep frequently changing resource locations in runtime registration rather than
hard-coding them into reusable pipeline YAML. `AppendSource` can load a complete
source into `ProcessingData`; `AppendProcessingData` can load selected values
with explicit metadata.

The CLI, Python runner, and runtime service all use the same registry contract.
Restricted servers additionally constrain allowed source types and filesystem
roots.
