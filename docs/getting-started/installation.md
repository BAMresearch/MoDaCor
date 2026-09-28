# Installation

MoDaCor requires Python 3.12 or newer. The recommended workflow uses
[uv](https://docs.astral.sh/uv/) to obtain Python and manage the environment.

## Released package

```bash
uv python install 3.12
uv venv --python 3.12
source .venv/bin/activate
uv pip install modacor
```

On Windows PowerShell, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

## Source checkout

From the repository root:

```bash
uv venv --python 3.12 .venv-dev
source .venv-dev/bin/activate
uv pip install -e ".[tests,lint,docs]"
```

## Optional capabilities

Install extras only when a workflow requires them:

```bash
uv pip install "modacor[server]"       # HTTP runtime service
uv pip install "modacor[tiled]"        # Tiled source and sink
uv pip install "modacor[plotting]"     # Plotly live-output sinks
uv pip install "modacor[attenuation]"  # attenuation-coefficient models
uv pip install "modacor[masks]"        # morphology helpers
```

Extras may be combined, for example `modacor[server,plotting]`.

Continue with the [Quickstart](quickstart.md), which needs only the base
installation.
