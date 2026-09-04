# DLS I22 notebook server pattern

This example shows how a notebook can drive the MoDaCor runtime service for a
DLS I22 operando SAXS/WAXS series. The full working notebook is a workspace
artifact, not a file shipped by the MoDaCor repository, so the snippets below
are reduced patterns with neutral paths.

Use this workflow when many similar measurements should reuse server-side
session state: the first sample runs in `full` mode, later samples update only
the `sample` source and run in `auto` mode.

## Notebook configuration

Keep paths and detector choices in one notebook cell. The SAXS and WAXS
pipelines are configured independently, but can share one runtime server.

```python
from pathlib import Path

PROJECT_DIR = Path("/path/to/i22_modacor_work")
DATA_ROOT = Path("/data/i22/experiment")
PROCESSING_DIR = PROJECT_DIR / "example_files" / "processing"
OUTPUT_DIR = PROJECT_DIR / "modacor_output"

PIPELINE_PATHS = {
    "SAXS": PROJECT_DIR / "processing_pipelines" / "I22_SAXS_solids_operando.yaml",
    "WAXS": PROJECT_DIR / "processing_pipelines" / "I22_WAXS_solids_operando.yaml",
}
CALIBRATION_FILES = {
    "SAXS": PROCESSING_DIR / "SAXS_calibration.nxs",
    "WAXS": PROCESSING_DIR / "WAXS_calibration.nxs",
}
MASK_FILES = {
    "SAXS": PROCESSING_DIR / "SAXS_mask.nxs",
    "WAXS": PROCESSING_DIR / "WAXS_mask.nxs",
}
BACKGROUND_FILES = {
    "SAXS": DATA_ROOT / "background.nxs",
    "WAXS": DATA_ROOT / "background.nxs",
}

DETECTORS_TO_RUN = ["SAXS", "WAXS"]
SESSION_IDS = {detector: f"i22-{detector.lower()}-server-batch" for detector in DETECTORS_TO_RUN}

SERVER_HOST = "127.0.0.1"
SERVER_PORT = 8901
BASE_URL = f"http://{SERVER_HOST}:{SERVER_PORT}"

TRACE_ENABLED = True
TRACE_WATCH = {"sample": ["signal"], "background": ["signal"]}

PLOT_SINK_REF = "plots"
RESULT_HDF_SINK_REF = "result_hdf"
SAMPLE_OUTPUT_DATA_PATHS = ["/sample/signal", "/sample/Q"]
```

If you want to test process-level concurrency, use separate ports and one
server process per detector. For most notebook work, one local server is easier
to inspect and stop.

## Preview the configured pipelines

Before creating runtime sessions, load the YAML files once in the notebook and
preview the configured graphs. This catches missing modules and YAML mistakes
before the server process is involved.

```python
from IPython.display import Markdown, display

from modacor.runner.pipeline import Pipeline


pipelines = {}
for detector, pipeline_path in PIPELINE_PATHS.items():
    pipeline = Pipeline.from_yaml_file(yaml_file=pipeline_path)
    pipeline.prepare()
    pipelines[detector] = pipeline
    display(Markdown(f"### {detector}\n\n```mermaid\n{pipeline.to_mermaid(direction='TD')}\n```"))
    print(f"{detector}: {len(pipeline.graph)} configured step(s) from {pipeline_path}")
```

## HTTP helpers

Small helpers keep the notebook readable while still making the REST endpoints
explicit.

```python
import json
import requests


def api_url(path: str) -> str:
    return BASE_URL.rstrip("/") + path


def api_request(method: str, path: str, *, payload: dict | None = None, expected=(200, 201, 202, 204)):
    response = requests.request(method, api_url(path), json=payload, timeout=120)
    if response.status_code not in expected:
        try:
            message = json.dumps(response.json(), indent=2)
        except ValueError:
            message = response.text
        raise RuntimeError(f"{method.upper()} {path} failed with HTTP {response.status_code}:\n{message}")
    if response.status_code == 204 or not response.content:
        return None
    return response.json()


def readiness_ok(timeout: float = 1.0) -> bool:
    try:
        response = requests.get(api_url("/v1/readiness"), timeout=timeout)
        return response.ok and bool(response.json().get("ready", False))
    except requests.RequestException:
        return False
```

## Start the server from a notebook

Starting the server from `sys.executable` keeps the runtime process in the same
Python environment as the notebook kernel. For compressed I22 HDF5 data, pass
the `hdf5plugin` plugin path through the server environment when available.

```python
import atexit
import os
import subprocess
import sys
import time


SERVER_PROCESS = None


def server_environment() -> dict[str, str]:
    env = os.environ.copy()
    try:
        import hdf5plugin
    except ImportError:
        return env
    if hasattr(hdf5plugin, "PLUGINS_PATH"):
        env.setdefault("HDF5_PLUGIN_PATH", hdf5plugin.PLUGINS_PATH)
    return env


def start_server(timeout_s: float = 45.0):
    global SERVER_PROCESS
    if readiness_ok():
        return None

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    log_path = OUTPUT_DIR / "modacor_server.log"
    command = [
        sys.executable,
        "-m",
        "modacor.cli",
        "serve",
        "--host",
        SERVER_HOST,
        "--port",
        str(SERVER_PORT),
    ]
    log_file = open(log_path, "a", buffering=1)
    SERVER_PROCESS = subprocess.Popen(command, stdout=log_file, stderr=subprocess.STDOUT, env=server_environment())

    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if readiness_ok():
            return SERVER_PROCESS
        if SERVER_PROCESS.poll() is not None:
            raise RuntimeError(f"MoDaCor server exited with code {SERVER_PROCESS.returncode}. See {log_path}.")
        time.sleep(0.5)
    raise TimeoutError(f"MoDaCor server did not become ready at {BASE_URL}")


def stop_server():
    global SERVER_PROCESS
    if SERVER_PROCESS is None or SERVER_PROCESS.poll() is not None:
        return
    SERVER_PROCESS.terminate()
    try:
        SERVER_PROCESS.wait(timeout=10)
    except subprocess.TimeoutExpired:
        SERVER_PROCESS.kill()
        SERVER_PROCESS.wait(timeout=10)
    SERVER_PROCESS = None


atexit.register(stop_server)
start_server()
api_request("GET", "/v1/readiness")
```

## Create detector sessions

Recreate sessions when the pipeline YAML or trace configuration changes. This
example uses `yaml_path`, which is appropriate for a trusted local notebook
server. For a restricted service, send `yaml_text` instead.

```python
def delete_session_if_exists(session_id: str) -> None:
    response = requests.delete(api_url(f"/v1/sessions/{session_id}"), timeout=30)
    if response.status_code not in (204, 404):
        raise RuntimeError(f"DELETE session failed with HTTP {response.status_code}: {response.text}")


def create_server_session(detector: str):
    session_id = SESSION_IDS[detector]
    delete_session_if_exists(session_id)
    payload = {
        "session_id": session_id,
        "name": f"I22 {detector} server batch",
        "pipeline": {"yaml_path": str(PIPELINE_PATHS[detector])},
        "trace": {
            "enabled": TRACE_ENABLED,
            "watch": TRACE_WATCH if TRACE_ENABLED else {},
            "record_only_on_change": True,
            "snapshot_processing_data": False,
            "snapshot_step_ids": [],
        },
        "auto_full_reset_on_partial_error": True,
    }
    return api_request("POST", "/v1/sessions", payload=payload, expected=(200, 201))


sessions = {detector: create_server_session(detector) for detector in DETECTORS_TO_RUN}
```

## Register sources and sinks

In the I22 pattern, `sample` changes for each measurement while `background`,
calibration, and mask files usually stay stable. Register runtime plot sinks
once, and update the result HDF sink path per sample.

```python
def source_registrations(detector: str, sample_path: Path):
    return [
        {"ref": "sample", "type": "hdf", "location": str(sample_path)},
        {"ref": "background", "type": "hdf", "location": str(BACKGROUND_FILES[detector])},
        {"ref": "saxs_calibration", "type": "hdf", "location": str(CALIBRATION_FILES["SAXS"])},
        {"ref": "saxs_mask", "type": "hdf", "location": str(MASK_FILES["SAXS"])},
        {"ref": "waxs_calibration", "type": "hdf", "location": str(CALIBRATION_FILES["WAXS"])},
        {"ref": "waxs_mask", "type": "hdf", "location": str(MASK_FILES["WAXS"])},
    ]


def register_sources(detector: str, sample_path: Path):
    return api_request(
        "PUT",
        f"/v1/sessions/{SESSION_IDS[detector]}/sources",
        payload={"sources": source_registrations(detector, sample_path)},
    )


def register_plot_sink(detector: str):
    return api_request(
        "PUT",
        f"/v1/sessions/{SESSION_IDS[detector]}/sinks",
        payload={"sinks": [{"ref": PLOT_SINK_REF, "type": "plotly_json", "location": "buffer://session"}]},
    )


def register_result_hdf_sink(detector: str, output_path: Path):
    return api_request(
        "POST",
        f"/v1/sessions/{SESSION_IDS[detector]}/sinks/patch",
        payload={"ref": RESULT_HDF_SINK_REF, "type": "hdf", "location": str(output_path)},
    )
```

## Process a series

The first sample seeds server state with a full run. Later samples use
`changed_sources=["sample"]` in `auto` mode so the service can plan a partial
rerun and fall back to a full rerun if needed.

```python
run_results = []
session_has_processing_state = {detector: False for detector in DETECTORS_TO_RUN}


def process_detector_sample(detector: str, sample_path: Path, index: int):
    session_id = SESSION_IDS[detector]
    run_name = f"{sample_path.stem}_{detector.lower()}"
    output_path = OUTPUT_DIR / f"{sample_path.stem}_{detector.lower()}_server_result.h5"
    mode = "auto" if session_has_processing_state[detector] else "full"

    api_request(
        "PUT",
        f"/v1/sessions/{session_id}/sources",
        payload={"sources": [{"ref": "sample", "type": "hdf", "location": str(sample_path)}]},
    )
    register_result_hdf_sink(detector, output_path)

    payload = {
        "mode": mode,
        "run_name": run_name,
        "rollback_snapshot": False,
        "write_hdf": {
            "path": str(output_path),
            "data_paths": SAMPLE_OUTPUT_DATA_PATHS,
        },
    }
    if mode == "auto":
        payload["changed_sources"] = ["sample"]

    result = api_request("POST", f"/v1/sessions/{session_id}/process", payload=payload)
    session_has_processing_state[detector] = True
    return {
        "detector": detector,
        "index": index,
        "sample": str(sample_path),
        "output": str(output_path),
        "mode": mode,
        "effective_mode": result.get("effective_mode"),
        "run_id": result.get("run_id"),
        "status": result.get("status"),
    }


sample_files = sorted((DATA_ROOT / "modacor_preprocessed").glob("*_modacor.nxs"))
if not sample_files:
    raise FileNotFoundError("No preprocessed I22 sample files found.")

for detector in DETECTORS_TO_RUN:
    register_sources(detector, sample_files[0])
    register_plot_sink(detector)

for index, sample_path in enumerate(sample_files):
    for detector in DETECTORS_TO_RUN:
        item = process_detector_sample(detector, sample_path, index)
        run_results.append(item)
        print(
            f"{item['detector']}: {item['status']} mode={item['mode']} "
            f"effective_mode={item['effective_mode']} run_id={item['run_id']}"
        )
```

When processing should continue after individual sample failures, catch the
exception around `process_detector_sample(...)`, inspect
`GET /v1/sessions/{session_id}/errors/latest`, and recreate the failed detector
session before the next sample if rollback snapshots are disabled.

## Preprocessed input shape

The notebook pattern keeps the original NeXus/HDF5 measurement files untouched.
For each measurement it writes a compact MoDaCor-facing file that:

- externally links `/entry1` from the original master file;
- adds broadcast-ready normalization arrays under `/modacor/normalization`;
- stores scalar calibration values under `/modacor/calibration`;
- reduces the beamstop-diode channel over its 2,000-sample axis to per-frame
  mean, standard deviation, SEM, and valid-sample count values;
- reshapes detector count time and transmission arrays to the detector-divisor
  layout used by the pipelines.

Detector geometry is deliberately not precomputed in the notebook. The SAXS and
WAXS YAML files point at NeXus calibration files, and MoDaCor resolves detector
coordinates and scattering geometry through `PixelCoordinates3D` and
`XSGeometryFromPixelCoordinates`.

## Correction pattern

The SAXS and WAXS pipelines share the same broad structure:

1. load sample, background, calibration-shape, and mask data;
2. attach Poisson uncertainties;
3. mask invalid raw counts;
4. normalize sample and background by beamstop-diode intensity;
5. normalize by detector count time;
6. average frame stacks with weights;
7. subtract the corrected background;
8. compute static pixel coordinates and scattering geometry from calibration
   NeXus metadata;
9. index pixels for azimuthal integration;
10. attach static maps and combine instrument, sample raw-count, and background
    raw-count masks;
11. apply solid-angle, detector-efficiency, polarization, and absolute-scale
    corrections;
12. publish live 2D and I(Q) plots and write HDF outputs.

The WAXS flow additionally applies an aluminium attenuator-plate correction
before polarization correction. In the current operando YAML, that correction
divides by the angle-dependent aluminium transmission and avoids a second
division by the scalar `/modacor/normalization/transmission` value.

## Detector-specific integration settings

The SAXS pipeline uses azimuthal integration from `0.03` to `3.36 1/nm` with
500 logarithmic Q bins.

The WAXS pipeline uses azimuthal integration from about `4.424` to
`54.717 1/nm` with 1502 linear Q bins.

Both pipelines use the calibration wavelength from the detector-specific NeXus
calibration file, silicon detector-efficiency correction with `0.32 mm`
thickness, linear polarization factor `0.9`, and an absolute intensity factor
stored in the preprocessed sample file.
