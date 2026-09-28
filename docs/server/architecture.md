# Client-server architecture

The server wraps the normal processing framework in long-lived session state:

```text
Python client / CLI / HTTP caller
              │
              ▼
       FastAPI HTTP service
              │
              ▼
   Runtime service and session manager
      ├── Pipeline and runner
      ├── source/sink registries
      ├── latest ProcessingData snapshot
      ├── trace and run history
      └── buffer/chunked-output resources
```

A `PipelineSession` owns one pipeline specification, runtime registrations,
latest processing state, trace configuration, run history, and error state. A
per-session lock allows one active run while independent sessions can operate
concurrently.

The HTTP layer validates requests and translates service errors. Scientific
execution remains in the shared runner, so local and remote results follow the
same step contracts. The synchronous Python client is the preferred
programmatic interface; the CLI is convenient for operators, and raw HTTP/OpenAPI
supports other control systems.

The service is in-memory unless an operation writes an external result or
chunked-output resource. Restart and recovery semantics are therefore explicit;
session definitions are not a substitute for facility configuration storage.
