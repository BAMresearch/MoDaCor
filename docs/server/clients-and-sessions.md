# Clients and sessions

Use `RuntimeClient` for supported programmatic access:

```python
from modacor.client import RuntimeClient

runtime = RuntimeClient("http://127.0.0.1:8000")
runtime.wait_until_ready()

session = runtime.create_session(
    session_id="demo",
    name="Synthetic demo",
    pipeline={"yaml_text": pipeline_yaml},
    trace={"enabled": True, "watch": {"sample": ["signal"]}},
)
```

The returned `SessionClient` scopes operations to one session. It can inspect or
delete the session, register sources and sinks, upload/download buffer data,
process or dry-run, reset or recover, inspect run history and the latest error,
and construct live-plot URLs.

Use stable session IDs for a known instrument/workflow combination. A session's
pipeline is instantiated at creation; replace or recreate the session after a
pipeline or custom-step definition changes.

The CLI exposes the same lifecycle for shell operation:

```bash
modacor session --url http://127.0.0.1:8000 list
```

See the [Python client reference](../reference/python-api/client.md),
[CLI reference](../reference/cli.md), and exact
[server API](../reference/server-api.md).
