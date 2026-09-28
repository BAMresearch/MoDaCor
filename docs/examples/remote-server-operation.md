# Remote server operation

Remote operation retains the same YAML graph but moves pipeline state, source
registrations, results, and trace history into a server session.

Start a trusted local service in an environment with the server extra:

```bash
uv pip install "modacor[server]"
modacor serve --host 127.0.0.1 --port 8000
```

Create a client and session:

```python
from modacor.client import RuntimeClient

runtime = RuntimeClient("http://127.0.0.1:8000")
runtime.wait_until_ready()
session = runtime.replace_session(
    "synthetic-example",
    pipeline_yaml=pipeline_yaml,
    trace={"enabled": True, "watch": {"sample": ["signal"]}},
)
```

For array delivery without a shared filesystem, register a buffer source and
upload its payload:

```python
session.register_source(
    {"ref": "input", "type": "buffer", "location": "buffer://session"}
)
source = session.source_buffer("input")
source.put_array("sample/signal/signal", counts)
source.put_attrs("sample/signal/signal", {"units": "count"})
```

The server-oriented form of the pipeline begins with
`AppendProcessingData(signal_location="input::sample/signal/signal")`. The
correction steps that follow are the same Poisson and normalization operations
used locally. A buffer or persistent sink can publish the result.

Trigger and inspect a run with:

```python
plan = session.dry_run(mode="full")
run = session.process(mode="full", run_name="synthetic")
history = session.runs()
```

Use a full buffer example from the server test suite as the executable contract
when adapting this pattern. For persistent services, read
[Security and runtime policy](../server/security-and-runtime-policy.md) before
binding beyond localhost.
