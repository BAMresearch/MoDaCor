# Command-line reference

Run `modacor --help` or `modacor <command> --help` for the exact options
installed with your version.

## `modacor run`

Execute one local pipeline. Required input is `--pipeline PATH`. Repeatable
source/sink registration options include `--hdf-source`, `--yaml-source`, and
`--csv-sink`. Tracing uses `--trace`, `--trace-watch`, snapshot controls, and
report limits. Execution can stop after a step. HDF output uses `--write-hdf`,
`--run-name`, repeatable `--write-path`, or `--write-all-processing-data`.

## `modacor serve`

Start the runtime HTTP service. Principal options select host, port, trusted or
restricted runtime policy, allowed read/write roots, session limits, maximum
pipeline size, and maximum buffer-upload size.

## `modacor session`

Manage a running service with `--url` and one of:

- `list`, `create`, `delete`, `status`, or `last-error`;
- `set-source`, `delete-source`, `set-sample`;
- `set-sink`, `delete-sink`;
- `process` or `dry-run` with `full`, `partial`, or `auto` mode;
- `reset`;
- `runs`; or
- `templates`.

Payload-rich or chunked-output operations are more conveniently expressed with
the [Python client](python-api/client.md) or exact [HTTP API](server-api.md).
