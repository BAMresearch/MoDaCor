# Extending MoDaCor

This section covers the extension points that are already used in the current
codebase:

- `ProcessStep` subclasses under `src/modacor/modules/...` for pipeline
  computation.
- `IoSource` and `IoSink` subclasses under `src/modacor/io/...` for external
  data access and export.
- the contributor workflow expected for tests, linting, and docs updates.

Use the pages below as the current maintainer guide rather than as a future
roadmap.

```{toctree}
:maxdepth: 1

module_author_guide
io_source_sink_guide
contribution_checklist
```
