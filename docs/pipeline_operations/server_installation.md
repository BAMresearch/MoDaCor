# Server installation

The runtime service is the long-lived HTTP API used for session-based
processing. Install the server extras when MoDaCor needs to accept files over
time, reuse cached session state, or be driven by an instrument control system.

For a released package:

```bash
pip install "modacor[server]"
```

For a source checkout:

```bash
pip install -e ".[server]"
```

Some pipelines need additional optional extras. Install them alongside
`server` when the pipeline uses those modules:

- `plotting` for runtime Plotly visualization sinks.
- `attenuation` for detector-efficiency and attenuator corrections backed by
  attenuation coefficients.
- `masks` for mask morphology helpers.

For example:

```bash
pip install -e ".[server,plotting,attenuation,masks]"
```

## Start a local service

Use the CLI entry point for local notebooks and trusted single-user workflows:

```bash
modacor serve --host 127.0.0.1 --port 8000
```

The service exposes:

- `GET /v1/health` for liveness.
- `GET /v1/readiness` for runtime readiness and high-level session metrics.
- `GET /docs` for FastAPI's generated interactive endpoint browser.

## Start a restricted service

Use the restricted runtime policy for network-facing or containerized services.
It disables arbitrary pipeline YAML paths, disables filesystem module discovery,
disables arbitrary custom IO class imports, and requires source/sink paths to
stay under configured roots.

```bash
modacor serve \
  --host 0.0.0.0 \
  --port 8000 \
  --runtime-policy restricted \
  --read-root /srv/modacor/pipelines \
  --read-root /data \
  --write-root /srv/modacor/output \
  --max-sessions 8 \
  --max-pipeline-yaml-bytes 262144 \
  --max-buffer-upload-bytes 104857600
```

The runtime service does not provide its own authentication or TLS termination.
Put network-facing deployments behind the facility's authenticated proxy or
service mesh.

## Create a session from the CLI

Once the server is running, `modacor session` commands call the HTTP API:

```bash
modacor session --url http://127.0.0.1:8000 create \
  --session-id i22-saxs \
  --name "I22 SAXS" \
  --pipeline-yaml-path /srv/modacor/pipelines/I22_SAXS_solids_operando.yaml \
  --trace

modacor session --url http://127.0.0.1:8000 set-source \
  --session-id i22-saxs \
  --ref sample \
  --type hdf \
  --location /data/current_sample_modacor.nxs

modacor session --url http://127.0.0.1:8000 process \
  --session-id i22-saxs \
  --mode full \
  --run-name first-sample
```

For a restricted service, submit `pipeline.yaml_text` through the REST API or
run session creation from a trusted local deployment that permits
`pipeline_yaml_path`.

The endpoint shapes, state machine, partial-rerun behavior, and error payloads
are documented in [Runtime Service API](runtime_service_api.md).
