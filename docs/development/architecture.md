# Package architecture

Dependency direction is intentional:

```text
geometry and models   reusable numerical/physics kernels
          │
          ▼
io                    format and transport adapters
          │
          ▼
modules               pipeline-facing adaptation
          │
          ▼
runner                 graph loading and local execution
          │
          ▼
server and client      long-lived remote operation
```

`dataclasses` supplies shared containers and step contracts. Reusable numerical
or physical calculations belong in `geometry` or `models`; file-format access
belongs in `io`; `modules` adapts those capabilities to `ProcessingData` and
pipeline configuration. Avoid imports that reverse this direction.

The server calls the shared runner rather than maintaining another scientific
execution implementation. The client is a supported transport facade and does
not import server internals.

For complete step boundaries and metadata requirements, read the
[module author guide](module-author-guide.md). For I/O contracts, read the
[source and sink guide](io-source-sink-guide.md).
